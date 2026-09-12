"""Phase 54 V54-C1 — PySR symbolic regression on ElementHyperNet output.

Extracts per-element (dopant_term) outputs from a trained V52b/V53-ζ/V54-A1
bundle's ElementHyperNet, then searches for a closed-form symbolic expression
as a function of physical descriptors (EN, ionic radius, oxidation state,
lone-pair, Walsh-D, optionally Mat2vec/UMA PCs).

Outputs `data/processed/pysr_dopant_term.json` with the Pareto-front formula
(picked by complexity vs. error), used by V54-C1 as a soft constraint via
_pysr_dopant_soft_loss in finetune_trainer.

Recommended workflow:
    1. Train V52b or V53-ζ to convergence first.
    2. Run this script with --bundle-dir pointing to the trained bundle.
    3. Inspect the formula in pysr_dopant_term.json; if RMSE < 0.25 in
       log10[V_O] units, integrate via V54-C1 config.

Cost: ~30 min single-seed PySR; ~5 h × 10 seeds for stable formula.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))


def _gather_hypernet_per_element(bundle_dir: Path) -> pd.DataFrame:
    """Eval the ElementHyperNet output for each known element on a representative
    process context.

    Returns DataFrame with columns:
        element, EN, radius, ox_state, lone_pair, walsh_D, hardness,
        chi_mm, r_mm, ox_off, hypernet_out
    """
    import torch
    from src.models.ga2o3_net import Ga2O3Net
    from src.data.element_descriptors import _RAW_DESCRIPTORS

    ckpt = bundle_dir / "stage3_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {ckpt}")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    # The stage3_model.pt format varies — try common shapes
    # (state_dict | full model | dict with 'model_state_dict')
    if isinstance(state, dict):
        if "model_state_dict" in state:
            sd = state["model_state_dict"]
        elif "model_state_dicts" in state:
            # Ensemble bundle (V53+): list of per-seed state dicts. Pick the
            # first seed — ElementHyperNet weights should be similar across
            # seeds (and PySR fits patterns, not exact values).
            sds = state["model_state_dicts"]
            sd = sds[0] if isinstance(sds, list) else next(iter(sds.values()))
        elif any(k.endswith(".weight") or k.endswith(".bias") for k in state.keys()):
            sd = state
        else:
            raise RuntimeError(
                f"Unrecognized checkpoint format: {list(state.keys())[:5]}"
            )
    elif isinstance(state, torch.nn.Module):
        sd = state.state_dict()
    else:
        raise RuntimeError(f"Unrecognized checkpoint type: {type(state)}")

    # Filter to ElementHyperNet weights only
    hn_state = {k.split("element_hypernet.", 1)[1]: v
                for k, v in sd.items() if "element_hypernet." in k}
    if not hn_state:
        raise RuntimeError(
            "No 'element_hypernet.*' tensors in checkpoint. "
            "Did you train with brouwer_use_hypernet=true?"
        )

    # Rebuild a stand-alone ElementHyperNet matching the checkpoint shapes
    from src.models.element_hypernet import ElementHyperNet
    # Infer out_dim from the final layer's bias size and desc_dim from first weight
    final_bias_key = next(k for k in hn_state if k.endswith("net.4.bias"))
    out_dim = int(hn_state[final_bias_key].shape[0])
    first_w_key = next(k for k in hn_state if k.endswith("net.0.weight"))
    desc_dim = int(hn_state[first_w_key].shape[1])
    hidden = int(hn_state[first_w_key].shape[0])
    mid_w_key = next(k for k in hn_state if k.endswith("net.2.weight"))
    mid = int(hn_state[mid_w_key].shape[0])
    hn = ElementHyperNet(desc_dim=desc_dim, hidden=hidden, mid=mid, out_dim=out_dim)
    hn.load_state_dict(hn_state, strict=False)
    hn.eval()

    rows = []
    elements = [e for e in _RAW_DESCRIPTORS.keys()
                if e not in ("undoped", "other", "H")]
    with torch.no_grad():
        for e in elements:
            raw = _RAW_DESCRIPTORS[e]
            # Build a dummy 1-element "spec" — ElementHyperNet.forward expects
            # ([B, K, D] descriptors, [B, K] weights). We feed B=1, K=1 with
            # full weight on this element.
            from src.data.element_descriptors import _standardized_descriptor_table
            table = _standardized_descriptor_table()
            desc = torch.tensor(table[e], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            w = torch.tensor([[1.0]], dtype=torch.float32)
            out = hn(desc, w).squeeze(0).cpu().numpy()
            row = {
                "element": e,
                "EN": raw[0], "radius": raw[1], "ox_state": int(raw[3]) + 3,
                "lone_pair": raw[5], "walsh_D": raw[6],
                "hardness": raw[7], "chi_mm": raw[8],
                "r_mm": raw[9], "ox_off": raw[10],
            }
            for i, v in enumerate(out.tolist()):
                row[f"hypernet_out_{i}"] = float(v)
            rows.append(row)
    return pd.DataFrame(rows)


def _run_pysr(features: pd.DataFrame, target_col: str, n_seeds: int = 1,
              maxsize: int = 25, niterations: int = 200) -> list[dict]:
    """Run PySR n_seeds times; return Pareto fronts."""
    from pysr import PySRRegressor

    feature_cols = ["EN", "radius", "ox_state", "lone_pair",
                    "walsh_D", "hardness", "chi_mm", "r_mm", "ox_off"]
    X = features[feature_cols].astype(float).values
    y = features[target_col].astype(float).values

    runs = []
    for seed in range(n_seeds):
        model = PySRRegressor(
            niterations=niterations,
            populations=20,
            population_size=33,
            maxsize=maxsize,
            unary_operators=["exp", "log", "square"],
            binary_operators=["+", "-", "*", "/"],
            model_selection="best",
            random_state=seed,
            deterministic=True,
            parallelism="serial",  # disables Julia parallelism for determinism
            progress=False,
            verbosity=0,
        )
        model.fit(X, y)
        # Best-by-loss formula
        best = model.get_best()
        # Pareto front (complexity vs loss)
        pareto = []
        if hasattr(model, "equations_") and model.equations_ is not None:
            for _, eq in model.equations_.iterrows():
                pareto.append({
                    "complexity": int(eq["complexity"]),
                    "loss": float(eq["loss"]),
                    "equation": str(eq["equation"]),
                })
        runs.append({
            "seed": seed,
            "best_equation": str(best["equation"]) if best is not None else None,
            "best_loss": float(best["loss"]) if best is not None else None,
            "best_complexity": int(best["complexity"]) if best is not None else None,
            "pareto": pareto,
        })
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="V54-C1 PySR symbolic dopant_term")
    parser.add_argument("--bundle-dir", required=True, type=Path,
                        help="Trained V52b/V53-ζ/V54-A1 bundle directory")
    parser.add_argument("--target-col", default="hypernet_out_0",
                        help="Which ElementHyperNet output to fit "
                             "(out_0 = log_VO offset; out_1 = a_e for V53-γ1)")
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Number of PySR seeds; 10 for stable formula")
    parser.add_argument("--niterations", type=int, default=200)
    parser.add_argument("--maxsize", type=int, default=25)
    parser.add_argument("--output", type=Path,
                        default=PROJ / "data" / "processed" / "pysr_dopant_term.json")
    args = parser.parse_args()

    print(f"Extracting ElementHyperNet outputs from {args.bundle_dir.name}…")
    features = _gather_hypernet_per_element(args.bundle_dir)
    print(features.to_string(index=False))

    if args.target_col not in features.columns:
        raise RuntimeError(
            f"Target col {args.target_col!r} not in extracted features: "
            f"{list(features.columns)}"
        )

    print(f"\nRunning PySR ({args.n_seeds} seeds × {args.niterations} iters)…")
    runs = _run_pysr(features, args.target_col,
                     n_seeds=args.n_seeds,
                     maxsize=args.maxsize,
                     niterations=args.niterations)

    # Aggregate: best loss across seeds + Pareto union
    best_run = min(runs, key=lambda r: r["best_loss"] if r["best_loss"] is not None else 1e9)
    summary = {
        "bundle_dir": str(args.bundle_dir),
        "target_col": args.target_col,
        "n_seeds": args.n_seeds,
        "elements_used": features["element"].tolist(),
        "best_run_seed": best_run["seed"],
        "best_equation": best_run["best_equation"],
        "best_loss": best_run["best_loss"],
        "best_complexity": best_run["best_complexity"],
        "rmse_log10_vo": float(np.sqrt(best_run["best_loss"])) if best_run["best_loss"] else None,
        "all_runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.output}")
    print(f"  Best equation : {summary['best_equation']}")
    print(f"  Best loss     : {summary['best_loss']:.4f}" if summary['best_loss'] else "  No formula")
    print(f"  RMSE log10[V_O]: {summary['rmse_log10_vo']:.4f}"
          if summary['rmse_log10_vo'] else "  No RMSE")
    print(f"  Complexity    : {summary['best_complexity']}")
    print(f"\nV54-C1 gate: RMSE ≤ 0.25 to accept this formula as soft prior.")


if __name__ == "__main__":
    main()
