"""V61 Stage-B train + eval (design docs/phase61_v61_design.md §3) — the within-DOI fit.

Per the user decision (within-DOI relative is the metric) and the pre-flight (cross-paper absolute is
unrecoverable), this fits the hierarchical measurement model:

    y_i = alpha_DOI + beta * r(s_i) + sigma(s_i)*eps_i

by MINIMIZING THE WITHIN-DOI residual: the per-DOI intercept alpha_DOI is PROFILED OUT by per-DOI
demeaning (so it never has to be fit and never leaks), beta is the optimal (heteroscedastic) slope, and
the net learns r = w_phys*z(bulk mechanism) + g_elem*monotone(log c) to maximize within-DOI explained
variance. The bulk mechanism (the self-consistent solver) supplies the causal direction + physics-rule
guarantees; g_elem supplies the DATA-ADJUDICATED per-element sign.

Protocol: GroupKFold-by-DOI (leakage-free), 5 seeds. Reports the four axes:
  (1) within-DOI relative R^2 (OOF, demeaned)              [PRIMARY]
  (2) per-element within-DOI slope-sign FLIP count          [per-element axis]
  (3) counterfactual mechanism sign via the solver (re-solve at changed c)  [causal direction]
  (4) physics-faithfulness gates from the solver            [physics rules]
plus the deployable absolute OOF R^2 (context only, expected near 0 per C3) + honest PI coverage.
All metrics -> results/phase61_v61/.  No /tmp.
"""
from __future__ import annotations
import json, math, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import sys
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from src.data.dcwm_transitions import (  # noqa: E402
    StateFeaturizer, _parse_conc_frac, atmosphere_to_key, method_to_idx, ATM_PO2,
)
from src.models.v61_net import V61Net, elem_to_idx, carrier_sign  # noqa: E402
from src.models.defect_equilibrium import solve_equilibrium  # noqa: E402

CSV = PROJ / "data/raw/experimental/ga2o3_exp_aug3x_vcinterp_v56.csv"


def load_rows(target: str):
    """Return per-row arrays: feats[N,71], y[N], doi[N], elem[N], c_frac[N], T_K[N], log_pO2[N], z[N]."""
    df = pd.read_csv(CSV)
    yv = pd.to_numeric(df[target], errors="coerce")
    df = df[yv.notna()].reset_index(drop=True)
    f = StateFeaturizer()
    feats, y, doi, elem, cfr, TK, lpO2, z = [], [], [], [], [], [], [], []
    for _, row in df.iterrows():
        e = str(row.get("element") or "undoped").strip()
        if e in ("—", "-", "nan", ""):
            e = "undoped"
        c = _parse_conc_frac(row)
        T_raw = pd.to_numeric(row.get("temperature_C"), errors="coerce")
        T_C = float(T_raw) if pd.notna(T_raw) else 25.0
        atm = atmosphere_to_key(str(row.get("atmosphere")))
        feats.append(f.featurize(e, c if c > 0 else 1e-6, T_C, atm, method_to_idx(str(row.get("method"))), 2))
        y.append(float(pd.to_numeric(row[target], errors="coerce")))
        doi.append(str(row.get("doi"))); elem.append(e)
        cfr.append(c if c > 0 else 0.0); TK.append(T_C + 273.15)
        lpO2.append(math.log10(ATM_PO2[atm])); z.append(carrier_sign(e))
    return (np.stack(feats).astype(np.float32), np.array(y, np.float32), np.array(doi),
            np.array(elem), np.array(cfr, np.float32), np.array(TK, np.float32),
            np.array(lpO2, np.float32), np.array(z, np.float32))


def _demean_by_group(vals, groups):
    out = np.array(vals, float).copy()
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= 1:
            out[m] = out[m] - out[m].mean()
    return out


def _within_doi_r2(y, pred, doi, min_n=2):
    """Pooled within-DOI relative R^2 over DOIs with >= min_n rows (demeaned per DOI)."""
    yt, pt, keep = [], [], np.zeros(len(y), bool)
    for g in np.unique(doi):
        m = doi == g
        if m.sum() >= min_n and np.std(y[m]) > 1e-9:
            keep |= m
    if keep.sum() < 2:
        return float("nan"), int(keep.sum())
    yd = _demean_by_group(y[keep], doi[keep])
    pd_ = _demean_by_group(pred[keep], doi[keep])
    ss_res = float(np.sum((yd - pd_) ** 2)); ss_tot = float(np.sum(yd ** 2))
    return (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")), int(keep.sum())


def train_fold(net, idx_tr, data, dev, steps=600, lr=3e-3, lam_ef=0.05, lam_g=1e-3):
    feats, y, doi, elem, cfr, TK, lpO2, z = data
    ei = np.array([elem_to_idx(e) for e in elem])
    Ft = torch.tensor(feats[idx_tr], device=dev)
    yt = torch.tensor(y[idx_tr], device=dev)
    TKt = torch.tensor(TK[idx_tr], device=dev); lpt = torch.tensor(lpO2[idx_tr], device=dev)
    zt = torch.tensor(z[idx_tr], device=dev); ct = torch.tensor(cfr[idx_tr], device=dev)
    eit = torch.tensor(ei[idx_tr], device=dev)
    doi_tr = doi[idx_tr]
    # per-DOI demean matrix for y (precompute group means as a function we re-apply to r each step)
    groups = [np.where(doi_tr == g)[0] for g in np.unique(doi_tr) if (doi_tr == g).sum() >= 2
              and np.std(y[idx_tr][doi_tr == g]) > 1e-9]
    if not groups:
        return net  # nothing to fit within-DOI
    y_d = yt.clone()
    for gi in groups:
        gi_t = torch.tensor(gi, device=dev)
        y_d[gi_t] = yt[gi_t] - yt[gi_t].mean()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        out = net(Ft, TKt, lpt, zt, ct, eit)
        r = out["r"]; logs = out["log_sigma"].clamp(-4, 4)
        # per-DOI demean r; fit beta in closed form (heteroscedastic), detached; NLL loss
        r_d = r.clone()
        for gi in groups:
            gi_t = torch.tensor(gi, device=dev)
            r_d = r_d.index_copy(0, gi_t, r[gi_t] - r[gi_t].mean())
        w = torch.exp(-2 * logs)
        idx_all = torch.cat([torch.tensor(gi, device=dev) for gi in groups])
        rd, yd, wd, ld = r_d[idx_all], y_d[idx_all], w[idx_all], logs[idx_all]
        # beta FIXED at 1 in training (r directly fits per-DOI-demeaned y): avoids the multiplicative-
        # zero cold-start (closed-form beta=0 at init would kill dL/dr). Global beta fit at eval only.
        resid = yd - rd
        nll = (0.5 * wd * resid ** 2 + ld).mean()
        reg = lam_ef * (net.dHf(Ft) - net.dHf_base.view(1, -1)).pow(2).mean() + lam_g * net.g_elem.weight.pow(2).mean()
        (nll + reg).backward()
        opt.step()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="vacancy_concentration")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--ablation", default="full",
                    choices=["full", "no_solver", "free_sign", "nonmono"])
    args = ap.parse_args()
    _ABL = {"full": (True, "constrained"), "no_solver": (False, "constrained"),
            "free_sign": (True, "free"), "nonmono": (True, "nonmono")}
    use_solver, sign_mode = _ABL[args.ablation]
    tag = "" if args.ablation == "full" else f"_{args.ablation}"
    dev = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)

    data = load_rows(args.target)
    feats, y, doi, elem, cfr, TK, lpO2, z = data
    N = len(y); uniq_doi = np.unique(doi)
    print(f"target={args.target}  N={N}  DOIs={len(uniq_doi)}  device={dev}")

    from sklearn.model_selection import GroupKFold
    oof_pred_r = np.full((args.seeds, N), np.nan)   # the relative response r (for within-DOI metric)
    oof_logs = np.full((args.seeds, N), np.nan)
    for seed in range(args.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        gkf = GroupKFold(n_splits=args.folds)
        for tr, te in gkf.split(np.arange(N), groups=doi):
            net = V61Net(feat_dim=feats.shape[1], use_solver=use_solver, sign_mode=sign_mode).to(dev)
            net = train_fold(net, tr, data, dev, steps=args.steps)
            net.eval()
            with torch.no_grad():
                ei_te = torch.tensor([elem_to_idx(e) for e in elem[te]], device=dev)
                out = net(torch.tensor(feats[te], device=dev), torch.tensor(TK[te], device=dev),
                          torch.tensor(lpO2[te], device=dev), torch.tensor(z[te], device=dev),
                          torch.tensor(cfr[te], device=dev), ei_te)
                oof_pred_r[seed, te] = out["r"].cpu().numpy()
                oof_logs[seed, te] = out["log_sigma"].cpu().numpy()

    # ---- metrics (averaged over seeds) ----
    r_mean = np.nanmean(oof_pred_r, axis=0)
    # (1) within-DOI relative R^2: fit beta globally on OOF demeaned, report R^2
    yd_keep, rd_keep, keepmask = [], [], np.zeros(N, bool)
    for g in uniq_doi:
        m = doi == g
        if m.sum() >= 2 and np.std(y[m]) > 1e-9 and np.std(r_mean[m]) > 1e-12:
            keepmask |= m
    if keepmask.sum() >= 2:
        yd = _demean_by_group(y[keepmask], doi[keepmask])
        rd = _demean_by_group(r_mean[keepmask], doi[keepmask])
        beta = float(np.sum(rd * yd) / (np.sum(rd * rd) + 1e-12))
        within_r2 = 1.0 - np.sum((yd - beta * rd) ** 2) / (np.sum(yd ** 2) + 1e-12)
        within_pearson = float(np.corrcoef(rd, yd)[0, 1]) if np.std(rd) > 0 else float("nan")
    else:
        beta, within_r2, within_pearson = float("nan"), float("nan"), float("nan")

    # (2) per-element within-DOI slope sign: compare the MODEL's actual learned within-DOI slope sign
    # (from OOF r) to the DATA within-DOI slope sign — both as demeaned (·, logc) covariance per DOI.
    # Computed from held-out predictions ⇒ ablation-agnostic and leakage-safe.
    per_elem = {}
    for e in np.unique(elem):
        m = elem == e
        if m.sum() < 4:
            continue
        logc = np.log10(np.clip(cfr[m], 1e-6, None))
        cd_e = _demean_by_group(logc, doi[m])
        yd_e = _demean_by_group(y[m], doi[m])
        rd_e = _demean_by_group(r_mean[m], doi[m])
        data_sign = float(np.sign(np.sum(cd_e * yd_e))) if np.std(cd_e) > 0 else 0.0
        model_sign = float(np.sign(np.sum(cd_e * rd_e))) if np.std(cd_e) > 0 and np.std(rd_e) > 1e-12 else 0.0
        per_elem[e] = dict(n=int(m.sum()), data_within_sign=data_sign, model_sign=model_sign,
                           flip=bool(data_sign != 0 and model_sign != 0 and data_sign != model_sign),
                           model_vs_data_within_cov=float(np.sum(rd_e * yd_e)))
    flips = int(sum(1 for v in per_elem.values() if v["flip"]))

    # (3) counterfactual mechanism sign via the solver (re-solve at +0.5 dex concentration)
    cf = counterfactual_solver_signs(elem, cfr, TK, lpO2, z, dev)

    # (4) physics gates: solver charge-neutrality residual on all rows (dilute -> ~0)
    with torch.no_grad():
        sol = solve_equilibrium(
            torch.tensor(np.tile([3.5, 1.8, 0.3], (N, 1)), dtype=torch.float32, device=dev),
            torch.tensor(TK, device=dev), torch.tensor(lpO2, device=dev),
            torch.tensor(z, device=dev), torch.tensor(cfr, device=dev))
    resid_rel = float((sol["residual"].abs() / 1e20).median().item())

    # deployable absolute OOF R^2 (context only): yhat = global_mean(y - beta r_train) + beta r
    alpha = float(np.mean(y[keepmask] - beta * r_mean[keepmask])) if keepmask.sum() else float(np.mean(y))
    yhat_abs = alpha + beta * r_mean
    ss = np.sum((y - yhat_abs) ** 2); st = np.sum((y - y.mean()) ** 2)
    abs_r2 = float(1 - ss / st) if st > 0 else float("nan")

    rep = dict(target=args.target, N=N, n_doi=len(uniq_doi), seeds=args.seeds, folds=args.folds,
               beta=beta, PRIMARY_within_doi_relative_R2=float(within_r2),
               within_doi_pearson=within_pearson, within_doi_n=int(keepmask.sum()),
               per_element=per_elem, per_element_sign_FLIPs=flips,
               counterfactual_mechanism=cf,
               physics_charge_neutrality_resid_rel_median=resid_rel,
               deployable_absolute_OOF_R2_context_only=abs_r2)
    rep["ablation"] = args.ablation
    out = PROJ / f"results/phase61_v61/verdict_{args.target}{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str))
    np.savez(PROJ / f"results/phase61_v61/oof_{args.target}{tag}.npz",
             r=oof_pred_r, logs=oof_logs, y=y, doi=doi, elem=elem, cfr=cfr)
    print(json.dumps({k: rep[k] for k in ("PRIMARY_within_doi_relative_R2", "within_doi_pearson",
          "within_doi_n", "per_element_sign_FLIPs", "counterfactual_mechanism",
          "physics_charge_neutrality_resid_rel_median", "deployable_absolute_OOF_R2_context_only",
          "beta")}, indent=2, default=str))
    print("\nper-element:")
    for e, v in per_elem.items():
        print(f"  {e:8s} n={v['n']:3d} data_sign={v['data_within_sign']:+.0f} "
              f"model_sign={v['model_sign']:+.0f} flip={v['flip']}")
    print(f"\nsaved -> {out}")


def counterfactual_solver_signs(elem, cfr, TK, lpO2, z, dev):
    """For each element, re-solve the bulk equilibrium at c and c*10^0.5; report mechanism sign of
    d log[V_O]/d log c (the EXACT thermodynamic counterfactual). Bulk equilibrium: acceptor>0, donor<0."""
    out = {}
    base = np.array([3.5, 1.8, 0.3], np.float32)
    for e in np.unique(elem):
        m = elem == e
        c = float(np.median(np.clip(cfr[m], 1e-5, None)))
        if c <= 1e-5 and e == "undoped":
            out[e] = dict(sign=0.0, note="undoped"); continue
        T = float(np.median(TK[m])); lp = float(np.median(lpO2[m])); zz = float(carrier_sign(e))
        with torch.no_grad():
            s1 = solve_equilibrium(torch.tensor(base[None], device=dev), torch.tensor([T], device=dev),
                                   torch.tensor([lp], device=dev), torch.tensor([zz], device=dev),
                                   torch.tensor([c], device=dev))
            s2 = solve_equilibrium(torch.tensor(base[None], device=dev), torch.tensor([T], device=dev),
                                   torch.tensor([lp], device=dev), torch.tensor([zz], device=dev),
                                   torch.tensor([c * (10 ** 0.5)], device=dev))
        d = float(s2["log10_VO"].item() - s1["log10_VO"].item())
        out[e] = dict(sign=float(np.sign(d)), delta=d)
    return out


if __name__ == "__main__":
    main()
