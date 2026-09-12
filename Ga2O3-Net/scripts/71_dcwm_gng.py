"""Phase 60 V60-DCWM — Go/No-Go + expansion orchestrator (Stage gen -> A -> B -> C), per target.

GNG criterion (docs/phase60_dcwm_jepa_design.md DECISIONS): after Stage-A (WM pretrain on
closed-form+DFT transitions) + Stage-B (Δ-over-WM-readout on the experimental labels, GroupKFold-by-DOI),
does the counterfactual sweep THROUGH THE WORLD MODEL lift `LEARNED_LAW` off 0 (vs project sign) while
within-element-R² is NOT regressed vs V55-Ext UNDER THE SAME GroupKFold-by-DOI protocol (fair floor)?

Single-target (V44 lesson): VC and PDR trained separately. Usage:
  PYTHONPATH=. python scripts/71_dcwm_gng.py --stage all --target vacancy_concentration --gpu 0
  PYTHONPATH=. python scripts/71_dcwm_gng.py --stage all --target photo_dark_ratio --gpu 1 --rho-lip 0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.data.dcwm_transitions import (StateFeaturizer, TransitionGenerator, ACTION_DIM, ATM_PO2,
                                       method_to_idx, atmosphere_to_key, C_REF)
from src.models.dcwm import DCWM
from src.training.dcwm_trainer import train_stageA, fit_stageB

OUT = PROJ / "results/phase60_dcwm"
CSV = PROJ / "data/raw/experimental/ga2o3_exp_aug3x_vcinterp_v55.csv"
TT = {"vacancy_concentration": "vc", "photo_dark_ratio": "pdr"}
# FAIR floor = V55-Ext within-element R² under THE SAME GroupKFold-by-DOI protocol
# (results/phase59_eval_*.json baseline_A). VC's older +0.605 was StratifiedKFold-leaky.
FAIR_FLOOR = {"vacancy_concentration": -8.051, "photo_dark_ratio": 0.065}

from scripts.eval_physics_diag_table import (ACCEPTOR_ELEMENTS, DONOR_ELEMENTS, SUPER_DONOR_ELEMENTS)
PROBE_ELEMENTS = sorted(set(ACCEPTOR_ELEMENTS) | set(DONOR_ELEMENTS) | set(SUPER_DONOR_ELEMENTS))
PROC_CONDITIONS = [
    ("RF magnetron sputtering", "Ar", 500.0), ("RF magnetron sputtering", "Ar", 900.0),
    ("RF magnetron sputtering", "O2", 700.0), ("RF magnetron sputtering", "Ar:O2=1:1", 700.0),
    ("MOCVD", "O2", 700.0), ("PLD", "Ar", 600.0),
]


def _expected_sign(elem, target):
    """Project sign convention. VC: acceptor-1, donor+1. PDR flips (acceptor+1, donor-1)."""
    flip = (target == "photo_dark_ratio")
    if elem in ACCEPTOR_ELEMENTS:
        return +1 if flip else -1
    if elem in DONOR_ELEMENTS or elem in SUPER_DONOR_ELEMENTS:
        return -1 if flip else +1
    return 0


def _paths(target, tag=""):
    tt = TT[target]
    sfx = f"_{tag}" if tag else ""
    # ablations reuse the base-target transition cache (same physics), only ckpt/oof/verdict differ
    return dict(cache=OUT / f"transitions_{tt}.npz", ckpt=PROJ / f"checkpoints/dcwm_stageA_{tt}{sfx}.pt",
                oof=OUT / f"oof_{tt}{sfx}.csv", vj=OUT / f"gng_verdict_{tt}{sfx}.json",
                vm=OUT / f"gng_verdict_{tt}{sfx}.md", log=OUT / f"stageA_log_{tt}{sfx}.csv", tt=tt + sfx)


def _build_model(feat, device):
    return DCWM(in_dim=feat.dim, action_dim=ACTION_DIM, d=64, a_emb=32, enc_hidden=192,
                n_blocks=4, dropout=0.1, sig_slices=256, sig_quad=17).to(device)


def _load_model(target, device, tag=""):
    P = _paths(target, tag)
    ck = torch.load(P["ckpt"], map_location=device, weights_only=False)
    feat = StateFeaturizer(mace_proj_dim=ck["mace_proj_dim"], use_z_expert=ck["use_z_expert"])
    model = _build_model(feat, device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    return model, feat


# ──────────────────────────────────────────────────────────────────────────────
def stage_gen(target, n=1_000_000, chunk=50_000, seed=7):
    OUT.mkdir(parents=True, exist_ok=True)
    P = _paths(target)
    feat = StateFeaturizer()
    from src.data.dcwm_transitions import load_experimental_recipes
    exp_recipes = load_experimental_recipes(str(CSV))
    gen = TransitionGenerator(feat, seed=seed, exp_recipes=exp_recipes, p_exp=0.5, target=target)
    print(f"[gen:{P['tt']}] dim={feat.dim}; {len(exp_recipes)} exp recipes; generating {n}...", flush=True)
    keys = ["x_s", "x_t", "a", "ym_s", "yb_s", "ym_t", "yb_t"]
    acc = {k: [] for k in keys}; done = 0; t0 = time.time()
    while done < n:
        b = gen.sample_batch(min(chunk, n - done))
        for k in keys:
            acc[k].append(b[k])
        done += min(chunk, n - done)
    data = {k: np.concatenate(acc[k], 0) for k in keys}
    np.savez(P["cache"], **data)
    print(f"[gen:{P['tt']}] saved {P['cache']} ({P['cache'].stat().st_size/1e6:.0f} MB) {time.time()-t0:.0f}s",
          flush=True)


def stage_A(target, gpu=0, steps=40000, batch=4096, lam_sig=0.05, mu_cyc=0.1, tag=""):
    P = _paths(target, tag)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu")
    feat = StateFeaturizer()
    cache = dict(np.load(P["cache"]))
    from src.data.dcwm_transitions import load_experimental_recipes, featurize_recipes
    exp_X = featurize_recipes(load_experimental_recipes(str(CSV)), feat)
    model = _build_model(feat, device)
    Xall = torch.tensor(np.concatenate([cache["x_s"][:200000], cache["x_t"][:200000], exp_X], 0),
                        device=device)
    model.encoder.std.fit(Xall)
    model.set_target_stats(cache["ym_s"].mean(), cache["ym_s"].std(),
                           cache["yb_s"].mean(), cache["yb_s"].std())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[A:{P['tt']}] device={device} trainable={n_train} steps={steps}", flush=True)
    train_stageA(model, cache, device, steps=steps, batch=batch, lam_sig=lam_sig, mu_cyc=mu_cyc,
                 log_path=str(P["log"]), exp_X=exp_X)
    P["ckpt"].parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_dim": feat.dim, "action_dim": ACTION_DIM,
                "mace_proj_dim": feat.mace_proj_dim, "use_z_expert": feat.use_z_expert}, P["ckpt"])
    print(f"[A:{P['tt']}] saved {P['ckpt']}", flush=True)


def stage_B(target, gpu=0, n_folds=10, n_seeds=5, rho_lip=0.0, tag=""):
    P = _paths(target, tag)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu")
    model, feat = _load_model(target, device, tag)
    from src.data.dcwm_transitions import load_experimental_states
    exp = load_experimental_states(str(CSV), target, feat)
    print(f"[B:{P['tt']}] rows={len(exp['y'])} DOIs={len(set(exp['doi']))} rho_lip={rho_lip}", flush=True)
    res = fit_stageB(model, exp, device, n_folds=n_folds, n_seeds=n_seeds, rho_lip=rho_lip)
    pd.DataFrame(dict(doi=res["doi"], element=res["element"], conc=res["conc"], y_true=res["y"],
                      y_pred=res["oof_pred"], base_meas=res["base_meas"])).to_csv(P["oof"], index=False)
    model.delta_meas.load_state_dict(res["global_delta_state"])
    ck = torch.load(P["ckpt"], map_location=device, weights_only=False)
    ck["state_dict"] = model.state_dict(); torch.save(ck, P["ckpt"])
    print(f"[B:{P['tt']}] saved {P['oof']} + baked Δ", flush=True)


# ── eval ──────────────────────────────────────────────────────────────────────
def within_element_vs_null(df):
    d = df.dropna(subset=["y_true", "y_pred", "element"]).copy()
    if len(d) < 4:
        return dict(gain=float("nan"), n=len(d))
    a, b = np.polyfit(d.y_pred.values, d.y_true.values, 1); d["pp"] = a * d.y_pred + b
    d["t"] = d.y_true - d.groupby("element").y_true.transform("mean")
    d["p"] = d.pp - d.groupby("element").pp.transform("mean")
    ss = float(((d.t - d.p) ** 2).sum()); tot = float((d.t ** 2).sum())
    return dict(gain=float(1 - ss / tot) if tot > 1e-12 else float("nan"), n=int(len(d)))


def data_referenced_within_doi(df, n_boot=1000, seed=42):
    from scipy import stats
    a, b = np.polyfit(df.y_pred, df.y_true, 1); df = df.assign(pp=a * df.y_pred + b)
    rng = np.random.default_rng(seed)
    learned = signok = flip = ambig = noref = 0; rows = []
    for (doi, el), g in df.groupby(["doi", "element"]):
        g = g[g.conc > 0]
        if g.conc.nunique() < 3:
            continue
        lc = np.log10(g.conc.values)
        rp = stats.spearmanr(lc, g.pp.values).statistic if g.pp.std() > 1e-9 else np.nan
        rt = stats.spearmanr(lc, g.y_true.values).statistic if g.y_true.std() > 1e-9 else np.nan
        if not (np.isfinite(rt) and abs(rt) > 1e-9 and np.isfinite(rp)):
            noref += 1; continue
        boots = []
        for _ in range(n_boot):
            idx = rng.choice(len(lc), len(lc), replace=True)
            if len(set(lc[idx])) < 2:
                continue
            rb = stats.spearmanr(lc[idx], g.pp.values[idx]).statistic
            if np.isfinite(rb):
                boots.append(rb)
        ci_lo = np.percentile(boots, 2.5) if len(boots) >= 100 else np.nan
        ci_hi = np.percentile(boots, 97.5) if len(boots) >= 100 else np.nan
        ci_conf = np.isfinite(ci_lo) and (ci_lo > 0 or ci_hi < 0)
        match = np.sign(rp) == np.sign(rt)
        v = ("LEARNED" if (match and ci_conf) else "SIGN_OK" if match else "FLIP" if ci_conf else "AMBIGUOUS")
        learned += v == "LEARNED"; signok += v == "SIGN_OK"; flip += v == "FLIP"; ambig += v == "AMBIGUOUS"
        rows.append(dict(doi=str(doi)[:40], element=el, rho_true=round(float(rt), 2),
                         rho_pred=round(float(rp), 2), verdict=v))
    return dict(LEARNED=learned, SIGN_OK=signok, FLIP=flip, AMBIGUOUS=ambig, NO_REF=noref, detail=rows)


def counterfactual_sweep(model, feat, device, target, apply_delta=False, n_grid=7,
                         c_min=1e-3, c_max=8e-2, flat_tol=0.15, flat_span=0.10):
    grid = np.logspace(math.log10(c_min), math.log10(c_max), n_grid); logc = np.log10(grid)
    c0 = float(10 ** ((math.log10(c_min) + math.log10(c_max)) / 2))
    from scipy import stats
    rows = []
    for elem in PROBE_ELEMENTS:
        exp = _expected_sign(elem, target)
        rhos, spans, monos, law_hits = [], [], [], []
        for (meth, atm, T) in PROC_CONDITIONS:
            base = feat.featurize(elem, c0, T, atmosphere_to_key(atm), method_to_idx(meth), 2)
            xs = torch.tensor(np.tile(base, (n_grid, 1)), dtype=torch.float32, device=device)
            acts = np.zeros((n_grid, ACTION_DIM), np.float32); acts[:, 0] = logc - math.log10(c0)
            preds = model.counterfactual_measured(xs, torch.tensor(acts, device=device),
                                                  apply_delta=apply_delta).detach().cpu().numpy()
            rho = float(stats.spearmanr(logc, preds).statistic) if np.std(preds) > 1e-9 else 0.0
            dpred = np.diff(preds); dom = np.sign(np.sum(dpred)) if np.any(dpred) else 0.0
            monos.append(float(np.mean(np.sign(dpred) == dom)) if dom != 0 else 0.0)
            rhos.append(rho); spans.append(float(preds.max() - preds.min()))
            law_hits.append(1.0 if (exp != 0 and np.sign(rho) == exp) else 0.0)
        rho_med = float(np.median(rhos)); span_med = float(np.median(spans))
        mono_med = float(np.median(monos)); law_frac = float(np.mean(law_hits)) if exp != 0 else float("nan")
        if exp == 0:
            v = "OK_FLAT" if abs(rho_med) < flat_tol else "UNEXPECTED"
        elif span_med < flat_span:
            v = "FLAT"
        elif abs(rho_med) < flat_tol:
            v = "NONMONOTONIC"
        elif np.sign(rho_med) == exp and mono_med >= 0.7 and (math.isnan(law_frac) or law_frac >= 0.6):
            v = "LEARNED_LAW"
        elif np.sign(rho_med) == exp:
            v = "RIGHT_SIGN_FRAGILE"
        else:
            v = "WRONG_DIRECTION"
        rows.append(dict(element=elem, expected=exp, rho_med=round(rho_med, 3), span_med=round(span_med, 3),
                         mono_med=round(mono_med, 2),
                         law_frac=None if math.isnan(law_frac) else round(law_frac, 2), verdict=v))
    graded = [r for r in rows if r["expected"] != 0]
    summ = dict(n_graded=len(graded),
                LEARNED_LAW=sum(r["verdict"] == "LEARNED_LAW" for r in graded),
                RIGHT_SIGN_FRAGILE=sum(r["verdict"] == "RIGHT_SIGN_FRAGILE" for r in graded),
                NONMONOTONIC=sum(r["verdict"] == "NONMONOTONIC" for r in graded),
                WRONG_DIRECTION=sum(r["verdict"] == "WRONG_DIRECTION" for r in graded))
    return summ, rows


def stage_C(target, gpu=0, tag=""):
    P = _paths(target, tag)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu")
    model, feat = _load_model(target, device, tag)
    df = pd.read_csv(P["oof"])
    mag = within_element_vs_null(df)
    wdoi = data_referenced_within_doi(df)
    cf_int, _ = counterfactual_sweep(model, feat, device, target, apply_delta=False)
    cf_dep, cf_rows = counterfactual_sweep(model, feat, device, target, apply_delta=True)
    d = df.dropna(subset=["y_true", "y_pred"])
    r2_raw = float(1 - ((d.y_true - d.y_pred) ** 2).sum() / ((d.y_true - d.y_true.mean()) ** 2).sum()) \
        if len(d) >= 2 else float("nan")
    floor = FAIR_FLOOR[target]
    pass_mag = bool(np.isfinite(mag["gain"]) and mag["gain"] >= floor)
    stretch = bool(np.isfinite(mag["gain"]) and mag["gain"] >= max(0.0, floor))
    pass_law = bool(cf_dep["LEARNED_LAW"] >= 1)
    verdict = "PASS" if (pass_mag and pass_law) else "FAIL"
    out = dict(gng_verdict=verdict, target=target,
               criterion=dict(magnitude_floor_GKF=floor, law_floor=1),
               magnitude=dict(within_element_r2=mag["gain"], n=mag.get("n"), pass_=pass_mag,
                              stretch_beats_floor=stretch,
                              vs_v55ext_GKF=round(mag["gain"] - floor, 4) if np.isfinite(mag["gain"]) else None),
               law_deployed=dict(**{k: cf_dep[k] for k in cf_dep}, pass_=pass_law, detail=cf_rows),
               law_intrinsic={k: cf_int[k] for k in cf_int},
               within_doi_data_referenced={k: wdoi[k] for k in wdoi},
               oof_r2_raw=r2_raw, oof_n=int(len(d)))
    P["vj"].write_text(json.dumps(out, indent=2, default=str))
    md = [f"# V60-DCWM Go/No-Go — {target} ({P['tt']}): **{verdict}**", "",
          f"- **Magnitude** within-element R² (Platt, GroupKFold-by-DOI) = **{mag['gain']:+.3f}** "
          f"(fair floor V55-Ext@GKF {floor}; Δ={mag['gain']-floor:+.3f}) -> {'PASS' if pass_mag else 'FAIL'}; "
          f"beats-floor {'YES' if stretch else 'no'}  [n={mag.get('n')}]",
          f"- **Law (deployed)** counterfactual LEARNED_LAW = **{cf_dep['LEARNED_LAW']}/{cf_dep['n_graded']}** "
          f"(floor ≥1; V55-Ext=0) -> {'PASS' if pass_law else 'FAIL'} "
          f"(RSF={cf_dep['RIGHT_SIGN_FRAGILE']}, WRONG={cf_dep['WRONG_DIRECTION']})",
          f"- Law (WM intrinsic) LEARNED_LAW = {cf_int['LEARNED_LAW']}/{cf_int['n_graded']}",
          f"- **Within-DOI vs MEASURED** (deployed, CI-gated): LEARNED={wdoi['LEARNED']}, "
          f"SIGN_OK={wdoi['SIGN_OK']}, FLIP={wdoi['FLIP']}, AMBIGUOUS={wdoi['AMBIGUOUS']}, NO_REF={wdoi['NO_REF']}",
          f"- OOF aggregate R² (raw) = {r2_raw:+.3f}  [n={len(d)}]", "",
          "| elem | exp | rho_med | span | mono | law_frac | verdict (deployed) |",
          "|---|---|---|---|---|---|---|"]
    for r in cf_rows:
        md.append(f"| {r['element']} | {r['expected']:+d} | {r['rho_med']:+.2f} | {r['span_med']:.2f} "
                  f"| {r['mono_med']:.2f} | {r['law_frac']} | {r['verdict']} |")
    P["vm"].write_text("\n".join(md)); print("\n".join(md))
    print(f"\nwrote {P['vj']} -> GNG({P['tt']}) = {verdict}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["gen", "A", "B", "C", "all"])
    ap.add_argument("--target", default="vacancy_concentration",
                    choices=["vacancy_concentration", "photo_dark_ratio"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--ntrans", type=int, default=1_000_000)
    ap.add_argument("--rho-lip", type=float, default=0.0, dest="rho_lip")
    ap.add_argument("--lam-sig", type=float, default=0.05, dest="lam_sig")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    P = _paths(args.target, args.tag)
    if args.stage in ("gen", "all"):
        if not P["cache"].exists():
            stage_gen(args.target, n=args.ntrans)
        else:
            print(f"[gen] {P['cache']} exists; skipping", flush=True)
    if args.stage in ("A", "all"):
        stage_A(args.target, gpu=args.gpu, steps=args.steps, lam_sig=args.lam_sig, tag=args.tag)
    if args.stage in ("B", "all"):
        stage_B(args.target, gpu=args.gpu, rho_lip=args.rho_lip, tag=args.tag)
    if args.stage in ("C", "all"):
        stage_C(args.target, gpu=args.gpu, tag=args.tag)


if __name__ == "__main__":
    main()
