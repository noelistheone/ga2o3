"""Phase 63 — per-doping-TYPE performance of the world-model surrogate vs the V63 solver (ground truth).

Breaks the aggregate metrics down by the doping types the user named: single element (donor/acceptor/
isovalent), binary co-doping (donor+acceptor, donor+donor, acceptor+acceptor, donor+isovalent), and multi-
dopant (ternary / quaternary, incl. a donor+acceptor ternary). For each type, on a fresh oracle-labeled test
set, reports: log[V_O] R^2 + Spearman rho (the TREND), E_F R^2 + rho, bound_frac R^2 (pairing), MAE. Marks
in-distribution (D in {1,2}, the training cardinality) vs OOD (D in {3,4}, held-out cardinality). Models:
sum (best law recovery) and mean (best magnitude); slot (non-invariant control) on a few types for contrast.
Writes results/phase63/per_type_performance.json. GPU, deterministic.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from src.data.codoping_oracle import build_batch, label_with_oracle, DONORS, ACCEPTORS, ISOVALENT, D_MAX  # noqa: E402
from src.models.v63_world_model import V63WorldModel, SlotMLP                                            # noqa: E402
from scipy.stats import spearmanr                                                                        # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
WM = PROJ / "results/phase63/worldmodel"
ST = json.load(open(WM / "standardize.json"))
MEAN = torch.tensor(ST["mean"]); STD = torch.tensor(ST["std"]); TGT = ST["targets"]
RNG = np.random.default_rng(2024)
N = 3000


def load(name):
    m = SlotMLP(d_max=D_MAX) if name == "slot" else V63WorldModel(agg=name)
    m = m.to(DEV); m.load_state_dict(torch.load(WM / f"wm_{name}.pt", map_location=DEV)); m.eval()
    return m


def conc():
    return float(10 ** RNG.uniform(-3.5, -1.5))


def make(type_spec, n=N):
    """type_spec: list of role tokens 'd'/'a'/'i'/'any' -> n recipes of that composition."""
    recipes, T, lp = [], [], []
    for _ in range(n):
        elems = []
        for role in type_spec:
            if role == "d": elems.append(str(RNG.choice(DONORS)))
            elif role == "a": elems.append(str(RNG.choice(ACCEPTORS)))
            elif role == "i": elems.append(str(RNG.choice(ISOVALENT)))
            else: elems.append(str(RNG.choice(DONORS + ACCEPTORS + ISOVALENT)))
        # ensure distinct elements
        seen = set(); uniq = []
        for e in elems:
            while e in seen:
                e = str(RNG.choice(DONORS + ACCEPTORS + ISOVALENT))
            seen.add(e); uniq.append(e)
        recipes.append([(e, conc()) for e in uniq])
        T.append(float(RNG.uniform(700, 1200))); lp.append(float(RNG.uniform(-7, 0)))
    return recipes, T, lp


@torch.no_grad()
def predict(model, recipes, T, lp):
    rec = build_batch(recipes, T, lp)
    o = model(rec["tokens"].to(DEV), rec["mask"].to(DEV), rec["T"].to(DEV), rec["lpo2"].to(DEV))
    return (torch.stack([o["E_F"], o["log10_VO"], o["bound_frac"]], 1).cpu() * STD + MEAN)


def r2(y, p):
    y = np.asarray(y); p = np.asarray(p); tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / tot) if tot > 0 else float("nan")


def evaluate(model, recipes, T, lp, lab):
    pred = predict(model, recipes, T, lp)
    yvo = lab["log10_VO"].numpy(); yef = lab["E_F"].numpy(); ybf = lab["bound_frac"].numpy()
    out = dict(
        logVO_R2=round(r2(yvo, pred[:, 1].numpy()), 3),
        logVO_rho=round(float(spearmanr(yvo, pred[:, 1].numpy()).correlation), 3),
        logVO_MAE=round(float(np.abs(yvo - pred[:, 1].numpy()).mean()), 3),
        EF_R2=round(r2(yef, pred[:, 0].numpy()), 3),
        EF_rho=round(float(spearmanr(yef, pred[:, 0].numpy()).correlation), 3))
    if ybf.std() > 1e-6:
        out["boundfrac_R2"] = round(r2(ybf, pred[:, 2].numpy()), 3)
    return out


TYPES = [
    ("single_donor",        ["d"],          "in-dist (D1)"),
    ("single_acceptor",     ["a"],          "in-dist (D1)"),
    ("single_isovalent",    ["i"],          "in-dist (D1)"),
    ("binary_donor+acceptor", ["d", "a"],   "in-dist (D2)"),
    ("binary_donor+donor",  ["d", "d"],     "in-dist (D2)"),
    ("binary_acceptor+acceptor", ["a", "a"], "in-dist (D2)"),
    ("binary_donor+isovalent", ["d", "i"],  "in-dist (D2)"),
    ("ternary_random",      ["any", "any", "any"], "OOD (D3)"),
    ("ternary_donor+acceptor+x", ["d", "a", "any"], "OOD (D3)"),
    ("quaternary_random",   ["any", "any", "any", "any"], "OOD (D4)"),
]


def main():
    models = {n: load(n) for n in ["sum", "mean", "slot"]}
    rep = {"device": DEV, "n_per_type": N,
           "note": "metrics are surrogate-vs-V63-solver (ground truth). Models trained ONLY on D in {1,2}; "
                   "D3/D4 are held-out cardinality (OOD). rho = trend/ranking fidelity (the user's 'trend'). "
                   "Surrogate ceiling is the solver; this is LAW/trend learning, not experimental magnitude.",
           "types": {}}
    print(f"{'doping type':28s} {'split':13s} | {'model':5s} {'logVO_R2':>8s} {'logVO_rho':>9s} {'EF_R2':>6s} {'bf_R2':>6s}")
    for key, spec, split in TYPES:
        recipes, T, lp = make(spec)
        lab = label_with_oracle(build_batch(recipes, T, lp), device=DEV, chunk=5000)
        rep["types"][key] = {"split": split, "models": {}}
        report_models = ["sum", "mean"] + (["slot"] if "binary" in key or "ternary_random" in key else [])
        for mn in report_models:
            ev = evaluate(models[mn], recipes, T, lp, lab)
            rep["types"][key]["models"][mn] = ev
            print(f"{key:28s} {split:13s} | {mn:5s} {ev['logVO_R2']:8.3f} {ev['logVO_rho']:9.3f} "
                  f"{ev['EF_R2']:6.3f} {ev.get('boundfrac_R2', float('nan')):6.3f}")
    out = PROJ / "results/phase63/per_type_performance.json"
    out.write_text(json.dumps(rep, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
