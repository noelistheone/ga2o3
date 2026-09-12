"""Phase 51+ — Local concentration-sensitivity test (finite-difference).

For each labeled sputter sample, perturb the dopant concentration by ±δ
(default 10%) without changing anything else, run model inference, and
check whether the predicted log[V_O] **direction** (sign of
pred(c+δ) - pred(c-δ)) matches the literature LIT_RHO sign for that
element.

This tests the model's **local concentration response**, finer-grained
than within-DOI rank tests because it isolates the conc gradient at each
specific (sample, atmosphere, temperature, method) combination.

Cost: ~3-5 min per bundle (load model + N forward passes per labeled
sample). No retraining.

Usage:
    conda activate ga2o3
    python scripts/eval_concentration_sensitivity.py results/<bundle> \\
        [--delta 0.10] [--target vacancy_concentration]
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.data.dopant_spec import DopantSpec, parse_spec
from src.data.experimental_dataset import (
    Ga2O3ExpDataset, collate_fn, build_process_tensor, _parse_row_spec,
)
from src.data.graph_builder import get_graph_for_spec, structure_to_graph
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Mirror the literature LIT_RHO from eval_physics_diag_table
LIT_RHO_VC = {
    "Mg": -0.95, "Zn": -0.90, "Cu": -0.85,
    "Sn": +0.95, "Si": +0.95, "Ti": +0.80, "Ge": +0.85,
    "Sb": +0.95, "Ta": +0.95, "Bi": +0.85,
}


def _resolve_config_path(bundle_dir: Path, fallback_default: str) -> Path:
    """Bundle-saved multimodal.yaml takes priority; if absent (V5-era bundles
    didn't save configs alongside), fall back to a known default."""
    p = bundle_dir / "multimodal.yaml"
    if p.exists():
        return p
    return PROJ / fallback_default


def _load_model(bundle_dir: Path, device: torch.device,
                mm_config_override: str | None = None):
    """Load V5/V18-style Ga2O3Net from bundle. Reuses V5 driver's load_model."""
    bundle_path = bundle_dir / "stage3_model.pt"
    if not bundle_path.exists():
        raise FileNotFoundError(f"stage3_model.pt missing in {bundle_dir}")
    if mm_config_override:
        mm_path = Path(mm_config_override)
        if not mm_path.is_absolute():
            mm_path = PROJ / mm_path
    else:
        mm_path = _resolve_config_path(
            bundle_dir,
            fallback_default="config/multimodal_v2_film_method_offset_distill.yaml",
        )
    if not mm_path.exists():
        raise FileNotFoundError(f"multimodal.yaml not found at {mm_path}")
    spec = importlib.util.spec_from_file_location(
        "v5_driver", str(PROJ / "scripts" / "04_finetune_predict.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.load_model(str(bundle_path), str(mm_path)).to(device)
    model.eval()
    return model, mm_path


def _build_perturbed_inputs(row: pd.Series, factor: float,
                              structures_dir: str, prefer_ordered: bool):
    """Reconstruct (graph, dopant_spec_str, process_tensor) with concentration
    multiplied by `factor` (1.0 = original)."""
    spec = _parse_row_spec(row)
    if spec.is_undoped or spec.total_conc <= 0:
        return None
    new_conc = float(np.clip(spec.total_conc * factor, 1e-4, 0.49))
    if len(spec.components) == 1:
        new_spec_str = f"{spec.components[0].formula}:{new_conc:.6f}"
    else:
        # Multi-component: scale all proportionally
        per_factor = new_conc / spec.total_conc
        parts = [f"{c.formula}:{c.conc * per_factor:.6f}" for c in spec.components]
        new_spec_str = ",".join(parts)
    new_spec = parse_spec(new_spec_str)
    graph = get_graph_for_spec(
        new_spec, structures_dir=structures_dir,
        prefer_ordered=prefer_ordered,
    )
    return graph, new_spec_str, new_spec


def _build_process_for_row(row: pd.Series, dataset: Ga2O3ExpDataset) -> torch.Tensor:
    """Use dataset.__getitem__ on the matching row index. The dataset
    already has process_layout / scaler logic baked in.
    """
    sidx = int(row.get("sample_idx", -1))
    if sidx >= 0 and sidx < len(dataset):
        # Cheap path — sample_idx maps directly to dataset position
        return dataset[sidx]["process"]
    # Fallback: scan
    for i in range(len(dataset)):
        d_row = dataset.df.iloc[i]
        if (str(d_row.get("element", "")) == str(row.get("element", "")) and
            float(d_row.get("concentration_at%", -999)) == float(row.get("concentration_at%", -1000)) and
            str(d_row.get("atmosphere", "")) == str(row.get("atmosphere", ""))):
            return dataset[i]["process"]
    raise RuntimeError(f"Row sample_idx={sidx} not found in dataset (len={len(dataset)})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bundle_dir")
    p.add_argument("--full-csv",
                   default="data/raw/experimental/ga2o3_exp_aug3x_vcinterp.csv")
    p.add_argument("--target", default="vacancy_concentration",
                   choices=["vacancy_concentration", "photo_dark_ratio"])
    p.add_argument("--delta", type=float, default=0.10,
                   help="Concentration perturbation factor (0.10 = ±10%)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--mm-config", default=None,
                   help="Override multimodal.yaml path. Default: bundle/multimodal.yaml "
                        "if present, else config/multimodal_v2_film_method_offset_distill.yaml")
    p.add_argument("--fu-config", default=None,
                   help="Override fusion config path. Default: bundle/fusion_head128.yaml "
                        "if present, else config/fusion_head128_phase43_v5_distill_vc.yaml")
    args = p.parse_args()

    bundle = Path(args.bundle_dir)
    if not bundle.is_absolute():
        bundle = PROJ / bundle

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # Load model
    logger.info(f"Loading model from {bundle}...")
    model, mm_path = _load_model(bundle, device, args.mm_config)
    logger.info(f"Using multimodal config: {mm_path}")

    # Load full CSV + filter
    csv_path = (PROJ / args.full_csv) if not Path(args.full_csv).is_absolute() else Path(args.full_csv)
    df = pd.read_csv(csv_path)
    if "usable_flag" in df.columns:
        df = df[df["usable_flag"] != "exotic_skip"].reset_index(drop=True)
    df = df[~df["element"].isin(["N", "H", "Nb", "In"])].reset_index(drop=True)
    df["sample_idx"] = np.arange(len(df))

    # Restrict to sputter labeled rows
    df = df[df["method"].fillna("").astype(str).str.lower().str.contains("sputter")]
    target_col = args.target
    if target_col not in df.columns:
        raise RuntimeError(f"target column {target_col} not in CSV")
    df = df[df[target_col].notna()].copy()
    df = df[df["element"].isin(LIT_RHO_VC.keys())].copy()

    # Multi-component rows are skipped — too risky for clean test
    df["_is_single_elem"] = df["element"].apply(lambda e: "+" not in str(e))
    df = df[df._is_single_elem].copy()
    logger.info(f"Eligible sputter labeled samples for sensitivity test: {len(df)}")

    # Build dataset to get process tensors
    mm_cfg = yaml.safe_load(mm_path.read_text())
    if args.fu_config:
        fu_path = Path(args.fu_config)
        if not fu_path.is_absolute():
            fu_path = PROJ / fu_path
    elif (bundle / "fusion_head128.yaml").exists():
        fu_path = bundle / "fusion_head128.yaml"
    else:
        fu_path = PROJ / "config" / "fusion_head128_phase43_v5_distill_vc.yaml"
    fu_cfg = yaml.safe_load(fu_path.read_text())
    prefer_ordered = mm_cfg.get("structure_stream", {}).get("prefer_ordered", False)
    structures_dir = str(PROJ / fu_cfg["paths"]["structures_dir"])
    logger.info(f"Using fusion config: {fu_path}")

    dataset = Ga2O3ExpDataset(
        csv_path=str(csv_path),
        structures_dir=structures_dir,
        target_cols=fu_cfg["targets"],
        augment=False,
        process_layout=mm_cfg.get("process_stream", {}).get("layout", "v2"),
        prefer_ordered=prefer_ordered,
    )

    # Run finite-difference per row
    rows_out = []
    delta = args.delta
    factors = (1.0 - delta, 1.0, 1.0 + delta)
    factor_labels = ("low", "center", "high")

    with torch.no_grad():
        for _, row in df.iterrows():
            elem = str(row["element"])
            if elem not in LIT_RHO_VC:
                continue
            try:
                process = _build_process_for_row(row, dataset).unsqueeze(0).to(device)
                preds = {}
                for fac, lab in zip(factors, factor_labels):
                    res = _build_perturbed_inputs(row, fac, structures_dir, prefer_ordered)
                    if res is None:
                        preds[lab] = None
                        continue
                    graph, spec_str, spec_obj = res
                    # PyG batch
                    from torch_geometric.data import Batch
                    batch_g = Batch.from_data_list([graph]).to(device)
                    out = model(batch_g, dopant_specs=[spec_str], process=process)
                    # V5 bundles trained with --single-target produce out
                    # with size 1 even though fu_cfg lists 2 targets.
                    if out.shape[1] == 1:
                        target_idx = 0
                    else:
                        target_idx = fu_cfg["targets"].index(target_col)
                    preds[lab] = float(out[0, target_idx].item())
                if preds.get("low") is None or preds.get("high") is None:
                    continue
                d_pred = preds["high"] - preds["low"]
                lit_sign = +1 if LIT_RHO_VC[elem] > 0 else -1
                pred_sign = +1 if d_pred > 0 else -1
                sign_match = "OK" if pred_sign == lit_sign else "FLIP"
                rows_out.append(dict(
                    sample_idx=int(row["sample_idx"]),
                    element=elem,
                    conc_at_pct=float(row["concentration_at%"]),
                    atmosphere=row["atmosphere"],
                    pred_low=preds["low"],
                    pred_center=preds["center"],
                    pred_high=preds["high"],
                    d_pred=d_pred,
                    expected_lit_sign=lit_sign,
                    pred_sign=pred_sign,
                    sign_match=sign_match,
                ))
            except Exception as e:
                import traceback
                logger.warning(f"sample {row.get('sample_idx', '?')} failed: "
                               f"{type(e).__name__}: {e}")
                logger.debug(traceback.format_exc())

    out_df = pd.DataFrame(rows_out)
    print(f"\n{'='*80}\nFINITE-DIFFERENCE CONCENTRATION SENSITIVITY — {bundle.name}\n{'='*80}\n")
    print(f"Perturbation: ±{delta*100:.0f}% concentration; tested {len(out_df)} samples")

    if len(out_df) == 0:
        print("\n(no samples successfully tested — see warnings above)")
        return

    # Per-element pass-rate
    by_elem = out_df.groupby("element").agg(
        n=("sign_match", "size"),
        ok=("sign_match", lambda s: (s == "OK").sum()),
    ).reset_index()
    by_elem["pass_rate"] = by_elem["ok"] / by_elem["n"]
    print(f"\nPer-element local-slope sign-match:")
    print(by_elem.to_string(index=False))

    overall_pass = float((out_df["sign_match"] == "OK").mean()) if len(out_df) else 0.0
    print(f"\nOverall pass rate: {overall_pass*100:.1f}% ({(out_df['sign_match']=='OK').sum()}/{len(out_df)})")

    # Save
    out_path = bundle / f"finite_diff_concentration_sensitivity_{target_col}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
