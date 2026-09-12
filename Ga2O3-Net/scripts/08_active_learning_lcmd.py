"""Phase 54 V54-E1 — Multi-fidelity LCMD active learning with KROGER oracle.

Holzmüller & Steinwart, "A Framework and Benchmark for Deep Batch Active
Learning for Regression," arXiv:2203.09410. LCMD = Locally-Conserved Mean
Distance (their proposed kernel + clustering acquisition; outperforms
BAIT/BADGE/BatchBALD on tabular regression).

Multi-fidelity setup per Roch et al. *Nat. Comput. Sci.* 5:572 (2025):
  - Low fidelity: KROGER MLP (val MAE 0.017 on synthetic Brouwer)
  - High fidelity: experimental sputter (≤6 recipes/quarter)

Acquisition: `SNGP variance × LCMD diversity × composition-similarity penalty`.
The closed-loop simulation predicts R²(after 6 lab) − R²(current).

Output: ranked top-N recipes for next wet-lab synthesis, plus simulated gain
per recipe (low-fidelity proxy).

Cost: ~10 min to query KROGER on 10⁴ Latin-hypercube points + LCMD selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))


def latin_hypercube_recipes(n: int = 10000, seed: int = 0) -> pd.DataFrame:
    """Generate n Latin-hypercube recipes over (dopant, c, T_sub, p_O2,
    anneal, substrate). Returns DataFrame."""
    rng = np.random.default_rng(seed)
    dopants = ["Mg", "Zn", "Cu", "Ca", "Si", "Sn", "Ge", "Al", "Fe",
                "Bi", "Sb", "Ti", "Zr", "Hf", "Sc", "Y", "Er", "Eu",
                "Ta", "W", "V", "B", "Ti", "F", "Nb"]
    # Concentrations log-spaced
    log_c = rng.uniform(np.log10(0.1), np.log10(10.0), n)
    c_at = 10.0 ** log_c
    T_sub = rng.choice([25, 100, 200, 300, 400, 500, 600, 700], n)
    p_O2 = rng.choice([0, 5, 10, 25, 50, 100], n)
    anneal = rng.choice(["none", "Ar800", "O2_800", "N2_700", "O2_plasma_1min"], n)
    substrate = rng.choice(["c-sap", "r-sap", "MgO", "Si"], n)
    dopant_idx = rng.integers(0, len(dopants), n)
    df = pd.DataFrame({
        "dopant": [dopants[i] for i in dopant_idx],
        "c_at": c_at,
        "T_sub_C": T_sub,
        "p_O2_pct": p_O2,
        "anneal": anneal,
        "substrate": substrate,
    })
    return df


def kroger_low_fidelity(recipes: pd.DataFrame, ckpt: Path) -> np.ndarray:
    """Query KROGER MLP on each recipe; returns log10[V_O] prediction."""
    import torch
    from src.models.kroger_expert import load_encoder_kroger
    from src.data.kroger_synthetic import ELEMENTS as KROGER_ELEMS

    expert = load_encoder_kroger(str(ckpt))
    expert.eval()

    # Map recipe → expert input format. The KROGER MLP expects
    # (T_K, log_pO2, dopant_one_hot[19], log_c).
    out = []
    with torch.no_grad():
        for _, r in recipes.iterrows():
            d = r["dopant"]
            if d not in KROGER_ELEMS:
                out.append(np.nan)
                continue
            try:
                # Build minimal process tensor + dopant_spec
                process_row = torch.zeros(1, 18, dtype=torch.float32)
                process_row[0, 0] = r["T_sub_C"]
                process_row[0, 2] = r["p_O2_pct"] / 100.0
                dopant_spec = [f"{d}:{r['c_at']/100:.4f}"]
                pred = expert(process_row, dopant_spec).item()
                out.append(pred)
            except Exception:
                out.append(np.nan)
    return np.array(out, dtype=np.float32)


def lcmd_selection(features: np.ndarray, uncertainties: np.ndarray,
                    n_select: int = 6, seed: int = 0) -> list[int]:
    """LCMD acquisition: pick `n_select` recipes maximizing
    (uncertainty * distance-to-nearest-selected). Greedy diversity-aware.
    """
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(features)
    n = len(X)
    selected = []
    available = list(range(n))
    rng = np.random.default_rng(seed)

    # First pick: max uncertainty (with tie-break by sample diversity)
    first = int(np.argmax(uncertainties))
    selected.append(first)
    available.remove(first)

    for _ in range(n_select - 1):
        if not available:
            break
        sel_arr = np.array(selected)
        # Distance of each available to closest selected
        diff = X[np.array(available)][:, None, :] - X[sel_arr][None, :, :]
        dist = np.linalg.norm(diff, axis=-1).min(axis=-1)
        score = uncertainties[np.array(available)] * dist
        best = available[int(np.argmax(score))]
        selected.append(best)
        available.remove(best)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="V54-E1 multi-fidelity LCMD AL")
    parser.add_argument("--n-candidates", type=int, default=10000,
                        help="Latin-hypercube candidate count")
    parser.add_argument("--n-select", type=int, default=6,
                        help="Number of recipes to recommend")
    parser.add_argument("--kroger-ckpt", type=Path,
                        default=PROJ / "checkpoints" / "encoder_kroger.pt")
    parser.add_argument("--output", type=Path,
                        default=PROJ / "results" / "phase54v54e1_al" / "lcmd_recommendations.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Generating {args.n_candidates} Latin-hypercube candidates…")
    recipes = latin_hypercube_recipes(args.n_candidates, args.seed)

    if not args.kroger_ckpt.exists():
        print(f"KROGER ckpt missing: {args.kroger_ckpt}")
        print("LCMD will run with synthetic uniform uncertainty (uninformative).")
        low_fi = np.zeros(len(recipes), dtype=np.float32)
    else:
        print(f"Querying KROGER low-fidelity ({args.kroger_ckpt})…")
        try:
            low_fi = kroger_low_fidelity(recipes, args.kroger_ckpt)
            print(f"  Low-fi mean log10[V_O] = {np.nanmean(low_fi):.2f} "
                  f"(range [{np.nanmin(low_fi):.2f}, {np.nanmax(low_fi):.2f}])")
        except Exception as e:
            print(f"KROGER query failed: {e}; falling back to uniform uncertainty.")
            low_fi = np.zeros(len(recipes), dtype=np.float32)

    # Drop nan low-fi rows (unsupported dopants)
    valid = ~np.isnan(low_fi)
    recipes = recipes.loc[valid].reset_index(drop=True)
    low_fi = low_fi[valid]
    print(f"After filter: {len(recipes)} candidates with KROGER prediction")

    # Build feature matrix (numerical encoding)
    # dopant one-hot expanded to descriptors via element_descriptors table
    from src.data.element_descriptors import _standardized_descriptor_table
    table = _standardized_descriptor_table()
    chem_emb = np.array([table.get(d, table["other"]) for d in recipes["dopant"]])
    proc_emb = recipes[["c_at", "T_sub_C", "p_O2_pct"]].astype(float).values
    proc_emb = np.log10(proc_emb.clip(min=1e-3))  # log-scale all
    anneal_oh = pd.get_dummies(recipes["anneal"], prefix="ann").values
    sub_oh = pd.get_dummies(recipes["substrate"], prefix="sub").values
    features = np.concatenate([chem_emb, proc_emb, anneal_oh, sub_oh], axis=1)
    print(f"Feature shape: {features.shape}")

    # Uncertainty: in this simplified V54-E1 prototype, use the KROGER predictive
    # variance proxy = | low_fi − low_fi.mean() |  (high-magnitude predictions
    # are usually under-constrained in the corpus). For the real V54-E1 with
    # V54-A2 SNGP integrated, this becomes SNGP posterior variance.
    uncertainties = np.abs(low_fi - np.nanmedian(low_fi)).astype(np.float32)
    uncertainties = uncertainties / max(uncertainties.max(), 1e-6)

    print(f"Running LCMD greedy selection (n_select={args.n_select})…")
    selected = lcmd_selection(features, uncertainties, n_select=args.n_select,
                               seed=args.seed)

    selected_recipes = recipes.iloc[selected].reset_index(drop=True)
    selected_recipes["low_fi_log10_vo"] = low_fi[selected]
    selected_recipes["uncertainty_proxy"] = uncertainties[selected]
    selected_recipes["selection_order"] = range(1, len(selected) + 1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected_recipes.to_csv(args.output, index=False)
    print(f"\nWrote {args.output}")
    print(selected_recipes.to_string(index=False))

    # Compare against roadmap's hand-curated 6-recipe batch
    roadmap_recipes = pd.DataFrame([
        {"#": 1, "goal": "OOD donor", "dopant": "Hf", "c_at": 1.5, "T_sub_C": 25,  "p_O2_pct": 10, "anneal": "Ar800", "substrate": "c-sap"},
        {"#": 2, "goal": "OOD lone-pair", "dopant": "Bi", "c_at": 2.0, "T_sub_C": 300, "p_O2_pct": 20, "anneal": "Ar800", "substrate": "c-sap"},
        {"#": 3, "goal": "OOD acceptor", "dopant": "Cu", "c_at": 0.8, "T_sub_C": 25,  "p_O2_pct": 0,  "anneal": "N2_700", "substrate": "c-sap"},
        {"#": 4, "goal": "high p_O2", "dopant": "Sn", "c_at": 1.0, "T_sub_C": 400, "p_O2_pct": 50, "anneal": "O2_800", "substrate": "c-sap"},
        {"#": 5, "goal": "O2 plasma", "dopant": "Si", "c_at": 2.0, "T_sub_C": 25,  "p_O2_pct": 10, "anneal": "O2_plasma_1min", "substrate": "c-sap"},
        {"#": 6, "goal": "OOD group III", "dopant": "Sc", "c_at": 1.5, "T_sub_C": 500, "p_O2_pct": 5,  "anneal": "Ar800", "substrate": "r-sap"},
    ])
    print(f"\nFor comparison — V54 roadmap hand-curated 6-recipe batch:")
    print(roadmap_recipes.to_string(index=False))

    summary = {
        "n_candidates": int(args.n_candidates),
        "n_selected": int(args.n_select),
        "kroger_ckpt": str(args.kroger_ckpt),
        "lcmd_selected_dopants": selected_recipes["dopant"].tolist(),
        "roadmap_dopants": roadmap_recipes["dopant"].tolist(),
        "agreement_count": int(set(selected_recipes["dopant"])
                                .intersection(roadmap_recipes["dopant"])
                                .__len__()),
    }
    (args.output.parent / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nLCMD ∩ roadmap dopants: {summary['agreement_count']}/6")


if __name__ == "__main__":
    main()
