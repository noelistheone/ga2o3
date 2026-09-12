"""Phase 63 — cross-doping-TYPE physics-law test suite for the world-model surrogate.

The decisive Phase-63 test (the user's focus = physics-law learning across doping types). Evaluates whether
the surrogate, trained ONLY on D in {1,2}, recovers the SOLVER's physical LAWS on HELD-OUT high-cardinality
(D in {3,4}) and unseen-type configs. We compare the surrogate's DIRECTIONAL response to the V63 solver
(ground truth) -- research-validated framing (counterfactual law recovery G1 >=8/9; directional rho>0 on
held-out type, NO R^2 floor on magnitude which is data-limited).

Gates:
  G1  counterfactual law recovery (9 physics rules) -- surrogate Delta-sign matches solver Delta-sign on
      >=80% of decisive configs => rule LEARNED. LEARNED_LAW = #rules / 9. Pass >=8/9.
  L1  monotonicity: d log[V_O]/d log c >= 0 for a cation dopant, on held-out D in {3,4} (% pass).
  L5  permutation invariance: max |f(perm(recipe)) - f(recipe)| (sum/mean/attn ~exact; slot fails).
  L6  c->0 reduction: f(D, one dopant c->0) ~= f(D-1) (mean |Delta log[V_O]|).
  G5  doping-TYPE held-out: retrain (sum) EXCLUDING acceptor+acceptor configs; test the compensation law
      + directional rho on acceptor+acceptor held-out.
All metrics -> results/phase63/law_tests.json. Re-reads standardize.json. GPU.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from src.data.codoping_oracle import (build_batch, label_with_oracle, sample_recipes,            # noqa: E402
                                       DONORS, ACCEPTORS, ISOVALENT, D_MAX)
from src.models.v63_world_model import V63WorldModel, SlotMLP                                    # noqa: E402
from src.models.v61_net import carrier_sign                                                      # noqa: E402
from scipy.stats import spearmanr                                                                # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
WMDIR = PROJ / "results/phase63/worldmodel"
ST = json.load(open(WMDIR / "standardize.json"))
MEAN = torch.tensor(ST["mean"]); STD = torch.tensor(ST["std"]); TGT = ST["targets"]
RNG = np.random.default_rng(0)


def load_model(name):
    if name == "slot":
        m = SlotMLP(d_max=D_MAX)
    elif name == "sum_gated":
        m = V63WorldModel(agg="sum", conc_gate=True)
    else:
        m = V63WorldModel(agg=name)
    m = m.to(DEV)
    m.load_state_dict(torch.load(WMDIR / f"wm_{name}.pt", map_location=DEV)); m.eval()
    return m


@torch.no_grad()
def surrogate(model, recipes, T, lpo2):
    rec = build_batch(recipes, T, lpo2)
    out = model(rec["tokens"].to(DEV), rec["mask"].to(DEV),
                rec["T"].to(DEV), rec["lpo2"].to(DEV))
    pred = torch.stack([out["E_F"], out["log10_VO"], out["bound_frac"]], dim=1).cpu() * STD + MEAN
    return {t: pred[:, i] for i, t in enumerate(TGT)}


def oracle(recipes, T, lpo2):
    rec = build_batch(recipes, T, lpo2)
    lab = label_with_oracle(rec, device=DEV, chunk=5000)
    return lab


# ---- recipe sampling (returns python lists so perturbations are easy) ---------------------------
def sample_lists(n, d_choices, mode="any"):
    """mode: any | has_donor | has_acceptor | donor_only | acceptor_only"""
    recipes, T, lp = [], [], []
    for _ in range(n):
        D = int(RNG.choice(d_choices))
        if mode == "donor_only":
            pool = DONORS + ISOVALENT
        elif mode == "acceptor_only":
            pool = ACCEPTORS + ISOVALENT
        else:
            pool = DONORS + ACCEPTORS + ISOVALENT
        elems = list(RNG.choice(pool, size=min(D, len(pool)), replace=False))
        if mode == "has_donor" and not any(carrier_sign(e) > 0 for e in elems):
            elems[0] = str(RNG.choice(DONORS))
        if mode == "has_acceptor" and not any(carrier_sign(e) < 0 for e in elems):
            elems[0] = str(RNG.choice(ACCEPTORS))
        rc = [(e, float(10 ** RNG.uniform(-3.5, -1.5))) for e in elems]
        recipes.append(rc); T.append(float(RNG.uniform(700, 1200))); lp.append(float(RNG.uniform(-7, 0)))
    return recipes, T, lp


def _scale_role(recipes, role, factor):
    out = []
    for rc in recipes:
        rc2 = []
        done = False
        for (e, c) in rc:
            s = carrier_sign(e)
            if not done and ((role == "donor" and s > 0) or (role == "acceptor" and s < 0)):
                rc2.append((e, c * factor)); done = True
            else:
                rc2.append((e, c))
        out.append(rc2)
    return out


def _add(recipes, pool):
    out = []
    for rc in recipes:
        if len(rc) < D_MAX:
            e = str(RNG.choice(pool))
            out.append(rc + [(e, float(10 ** RNG.uniform(-3.0, -1.7)))])
        else:
            out.append(rc)
    return out


def cf_rule(model, base, T, lp, pert_recipes, pertT, pertLP, obs, eps=0.02):
    """Sign-agreement of surrogate Delta vs solver Delta on decisive configs (|solver Delta|>eps)."""
    i = TGT.index(obs)
    s0 = surrogate(model, base, T, lp)[obs]; s1 = surrogate(model, pert_recipes, pertT, pertLP)[obs]
    o0 = oracle(base, T, lp)[obs]; o1 = oracle(pert_recipes, pertT, pertLP)[obs]
    ds = (s1 - s0); do = (o1 - o0)
    dec = do.abs() > eps
    if dec.sum() < 5:
        return None
    agree = (torch.sign(ds[dec]) == torch.sign(do[dec])).float().mean().item()
    return dict(agree=round(agree, 3), n_decisive=int(dec.sum()),
                solver_mean_dsign=round(float(torch.sign(do[dec]).float().mean()), 2))


def run_G1(model):
    """9 counterfactual physics rules; LEARNED if surrogate matches solver sign >=80% on decisive configs."""
    rules = {}
    base_d, Td, lpd = sample_lists(800, (2, 3, 4), "has_donor")
    base_a, Ta, lpa = sample_lists(800, (2, 3, 4), "has_acceptor")
    base_any, Tn, lpn = sample_lists(800, (3, 4), "any")
    base_d2, Td2, lpd2 = sample_lists(800, (2,), "donor_only")
    base_a2, Ta2, lpa2 = sample_lists(800, (2,), "acceptor_only")
    base_b2, Tb2, lpb2 = sample_lists(800, (2,), "any")

    rules["R1_donor_up_EF_up"] = cf_rule(model, base_d, Td, lpd, _scale_role(base_d, "donor", 4.0), Td, lpd, "E_F")
    rules["R2_acceptor_up_EF_down"] = cf_rule(model, base_a, Ta, lpa, _scale_role(base_a, "acceptor", 4.0), Ta, lpa, "E_F")
    rules["R3_acceptor_up_VO_up"] = cf_rule(model, base_a, Ta, lpa, _scale_role(base_a, "acceptor", 4.0), Ta, lpa, "log10_VO")
    rules["R4_donor_up_VO_down"] = cf_rule(model, base_d, Td, lpd, _scale_role(base_d, "donor", 4.0), Td, lpd, "log10_VO")
    rules["R5_pO2_up_VO_down"] = cf_rule(model, base_any, Tn, lpn, base_any, Tn, [x + 3 for x in lpn], "log10_VO")
    rules["R6_T_up_VO_up"] = cf_rule(model, base_any, Tn, lpn, base_any, [x + 200 for x in Tn], lpn, "log10_VO")
    rules["R7_add_acceptor_EF_down"] = cf_rule(model, base_d2, Td2, lpd2, _add(base_d2, ACCEPTORS), Td2, lpd2, "E_F")
    rules["R8_add_donor_EF_up"] = cf_rule(model, base_a2, Ta2, lpa2, _add(base_a2, DONORS), Ta2, lpa2, "E_F")
    # R9 isovalent add perturbs E_F LESS than a donor add (both surrogate & solver) -> learned if surrogate agrees
    iso = _add(base_b2, ISOVALENT); don = _add(base_b2, DONORS)
    s_iso = surrogate(model, iso, Tb2, lpb2)["E_F"]; s_don = surrogate(model, don, Tb2, lpb2)["E_F"]
    s_base = surrogate(model, base_b2, Tb2, lpb2)["E_F"]
    o_iso = oracle(iso, Tb2, lpb2)["E_F"]; o_don = oracle(don, Tb2, lpb2)["E_F"]; o_base = oracle(base_b2, Tb2, lpb2)["E_F"]
    dec = (o_don - o_base).abs() > 0.02
    learned9 = ((s_iso - s_base).abs()[dec] < (s_don - s_base).abs()[dec]).float().mean().item()
    rules["R9_isovalent_less_than_donor"] = dict(agree=round(learned9, 3), n_decisive=int(dec.sum()))

    passed = sum(1 for r in rules.values() if r and r["agree"] >= 0.80)
    valid = sum(1 for r in rules.values() if r)
    return dict(rules=rules, learned=passed, of=valid, pass_=bool(passed >= 8))


def run_L1_monotonicity(model, n=600):
    base, T, lp = sample_lists(n, (3, 4), "any")
    i = TGT.index("log10_VO")
    out = []
    for fac in [1.0, 3.0]:
        out.append(surrogate(model, _scale_role(base, "donor", fac), T, lp)["log10_VO"])
    # monotonicity wrt a cation dopant conc: use the FIRST dopant scaled up (any cation)
    up = [[(e, c * (3.0 if k == 0 else 1.0)) for k, (e, c) in enumerate(rc)] for rc in base]
    s0 = surrogate(model, base, T, lp)["log10_VO"]; s1 = surrogate(model, up, T, lp)["log10_VO"]
    o0 = oracle(base, T, lp)["log10_VO"]; o1 = oracle(up, T, lp)["log10_VO"]
    dec = (o1 - o0).abs() > 0.02
    mono_surr = (s1 - s0)[dec] >= -1e-3
    return dict(name="L1_monotonicity_VO_vs_conc_D34",
                surrogate_pct_monotone=round(float(mono_surr.float().mean()), 3),
                solver_pct_monotone=round(float(((o1 - o0)[dec] >= -1e-3).float().mean()), 3),
                n_decisive=int(dec.sum()), pass_=bool(float(mono_surr.float().mean()) >= 0.90))


def run_L5_perm(model, n=400):
    base, T, lp = sample_lists(n, (3, 4), "any")
    perm = []
    for rc in base:
        idx = list(RNG.permutation(len(rc)))
        perm.append([rc[k] for k in idx])
    s0 = surrogate(model, base, T, lp); s1 = surrogate(model, perm, T, lp)
    dev = max(float((s1[t] - s0[t]).abs().max()) for t in TGT)
    return dict(name="L5_permutation_invariance", max_dev=round(dev, 6), pass_=bool(dev < 1e-3))


def run_L6_reduction(model, n=400):
    base, T, lp = sample_lists(n, (3, 4), "any")     # D>=3
    drop = [rc[:-1] for rc in base]                  # D-1 (remove last dopant)
    tiny = [rc[:-1] + [(rc[-1][0], 1e-7)] for rc in base]  # last dopant c->0
    s_drop = surrogate(model, drop, T, lp)["log10_VO"]; s_tiny = surrogate(model, tiny, T, lp)["log10_VO"]
    return dict(name="L6_c_to_0_reduction", mean_abs_dlogVO=round(float((s_tiny - s_drop).abs().mean()), 4),
                pass_=bool(float((s_tiny - s_drop).abs().mean()) < 0.10))


# ---- minimal inline trainer for the doping-TYPE held-out experiment (G5) ------------------------
def _train_sum(rec, lab, n_epoch=60, bs=4096):
    Y = torch.stack([lab[t] for t in TGT], dim=1)
    mean = Y.mean(0, keepdim=True); std = Y.std(0, keepdim=True).clamp_min(1e-6)
    Ys = ((Y - mean) / std).to(DEV)
    tok, msk, T, lp = rec["tokens"].to(DEV), rec["mask"].to(DEV), rec["T"].to(DEV), rec["lpo2"].to(DEV)
    model = V63WorldModel(agg="sum").to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epoch)
    n = tok.shape[0]; g = torch.Generator(device=DEV).manual_seed(0)
    for _ in range(n_epoch):
        perm = torch.randperm(n, generator=g, device=DEV)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            out = model(tok[idx], msk[idx], T[idx], lp[idx])
            pred = torch.stack([out["E_F"], out["log10_VO"], out["bound_frac"]], dim=1)
            loss = nn.functional.mse_loss(pred, Ys[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return model, mean, std


def run_G5_type_heldout():
    """Train sum-model EXCLUDING acceptor+acceptor configs; test compensation + directional rho on AA held-out."""
    global RNG
    RNG = np.random.default_rng(777)        # fresh seed -> G5 reproducible regardless of model-loop order
    torch.manual_seed(777)                  # seed model init too (compensation metric is init-sensitive)
    g = torch.Generator().manual_seed(11)
    rec = sample_recipes(120000, g, d_choices=(1, 2, 3, 4), force_da=0.4)
    # exclude any config with >=2 acceptors
    nacc = (rec["z"] < 0).sum(dim=1)
    keep = nacc < 2
    rec_f = {k: (v[keep] if torch.is_tensor(v) and v.shape[0] == keep.shape[0] else v) for k, v in rec.items()}
    lab_f = label_with_oracle(rec_f, device=DEV, chunk=5000)
    model, mean, std = _train_sum(rec_f, lab_f)
    # held-out test: acceptor+acceptor (+ optional extra) configs
    aa, Taa, lpaa = [], [], []
    for _ in range(1500):
        a1, a2 = RNG.choice(ACCEPTORS, size=2, replace=False)
        rc = [(str(a1), float(10 ** RNG.uniform(-3, -1.5))), (str(a2), float(10 ** RNG.uniform(-3, -1.5)))]
        aa.append(rc); Taa.append(float(RNG.uniform(700, 1200))); lpaa.append(float(RNG.uniform(-7, 0)))
    rec_aa = build_batch(aa, Taa, lpaa)
    with torch.no_grad():
        out = model(rec_aa["tokens"].to(DEV), rec_aa["mask"].to(DEV), rec_aa["T"].to(DEV), rec_aa["lpo2"].to(DEV))
        ps = (torch.stack([out["E_F"], out["log10_VO"], out["bound_frac"]], dim=1).cpu() * std + mean)
    lab_aa = label_with_oracle(rec_aa, device=DEV, chunk=5000)
    rho_vo = float(spearmanr(lab_aa["log10_VO"].numpy(), ps[:, 1].numpy()).correlation)
    rho_ef = float(spearmanr(lab_aa["E_F"].numpy(), ps[:, 0].numpy()).correlation)
    # compensation law: scale up the 2nd acceptor -> [V_O] should rise (solver direction), surrogate agrees?
    aa_up = [[(e, c * (4.0 if k == 1 else 1.0)) for k, (e, c) in enumerate(rc)] for rc in aa]
    rec_up = build_batch(aa_up, Taa, lpaa)
    with torch.no_grad():
        ou = model(rec_up["tokens"].to(DEV), rec_up["mask"].to(DEV), rec_up["T"].to(DEV), rec_up["lpo2"].to(DEV))
        su = (torch.stack([ou["E_F"], ou["log10_VO"], ou["bound_frac"]], dim=1).cpu() * std + mean)
    lab_up = label_with_oracle(rec_up, device=DEV, chunk=5000)
    do = lab_up["log10_VO"] - lab_aa["log10_VO"]; dsur = su[:, 1] - ps[:, 1]
    dec = do.abs() > 0.02
    comp_agree = float((torch.sign(dsur[dec]) == torch.sign(do[dec])).float().mean())
    return dict(name="G5_doping_type_heldout_acceptor_acceptor",
                trained_excluding="acceptor+acceptor configs", n_train=int(keep.sum()),
                heldout_rho_logVO=round(rho_vo, 3), heldout_rho_EF=round(rho_ef, 3),
                compensation_sign_agreement=round(comp_agree, 3),
                pass_=bool(rho_vo > 0 and comp_agree >= 0.80))


def main():
    t0 = time.time()
    rep = {"device": DEV, "models": {}}
    for name in ["sum", "mean", "attn", "slot", "sum_gated"]:
        m = load_model(name)
        g1 = run_G1(m)
        rep["models"][name] = dict(
            G1_counterfactual_law=g1,
            L1=run_L1_monotonicity(m), L5=run_L5_perm(m), L6=run_L6_reduction(m))
        print(f"[{time.time()-t0:.0f}s] {name:5s}  G1 learned={g1['learned']}/{g1['of']}  "
              f"L1mono={rep['models'][name]['L1']['surrogate_pct_monotone']}  "
              f"L5dev={rep['models'][name]['L5']['max_dev']}  L6red={rep['models'][name]['L6']['mean_abs_dlogVO']}")
    print(f"[{time.time()-t0:.0f}s] running G5 doping-type held-out (retrain excluding acceptor+acceptor)...")
    rep["G5_type_heldout"] = run_G5_type_heldout()
    print(f"[{time.time()-t0:.0f}s] G5: {rep['G5_type_heldout']}")
    out = PROJ / "results/phase63/law_tests.json"
    out.write_text(json.dumps(rep, indent=2))
    print(f"[{time.time()-t0:.0f}s] DONE -> {out}")


if __name__ == "__main__":
    main()
