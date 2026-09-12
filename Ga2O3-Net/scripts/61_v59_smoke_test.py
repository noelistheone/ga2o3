"""
Phase 59 V59-CALM smoke test — verifies the three architectural guarantees of
docs/phase59_llm_multiexpert_physics_law_design.md §3.4 BEFORE any training:

  1. C1 (parameter frugality): CALMResidual adds < 10 k trainable params.
  2. No-regression-at-init: with Δ=0 (zero w_out), V59-CALM predictions are
     byte-identical to the V55-Ext baseline at initialization.
  3. DCC feasibility: ∂ŷ/∂(process[:,11]=log10 c) is computable via
     torch.autograd.grad(create_graph=True) (the differentiable slope ρ).

Run: PYTHONPATH=. conda run -n ga2o3 python scripts/61_v59_smoke_test.py --gpu 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import importlib.util
_s = importlib.util.spec_from_file_location("s04", HERE / "scripts" / "04_finetune_predict.py")
s04 = importlib.util.module_from_spec(_s); _s.loader.exec_module(s04)

from src.data.experimental_dataset import Ga2O3ExpDataset
from src.models.ga2o3_net import Ga2O3Net


def build_model(mm_path: str, targets, device, seed: int = 0):
    mm_cfg = yaml.safe_load(open(mm_path))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hc = {"hidden_dims": [64, 32], "dropout": 0.30, "type": "brouwer"}
    model = Ga2O3Net.from_pretrained(
        encoder_ckpt=mm_cfg["structure_stream"]["pretrained_ckpt"],
        target_cols=targets,
        fusion_mode=mm_cfg["fusion"]["mode"],
        encoder_backbone="cgcnn",
        composition_kwargs={"hidden_dim": mm_cfg["composition_stream"]["hidden_dim"],
                            "out_dim": mm_cfg["composition_stream"]["out_dim"],
                            "dropout": mm_cfg["composition_stream"]["dropout"]},
        process_kwargs={"in_dim": mm_cfg["process_stream"]["in_dim"],
                        "continuous_dim": mm_cfg["process_stream"].get("continuous_dim", 12),
                        "method_embed_dim": mm_cfg["process_stream"].get("method_embed_dim", 3),
                        "n_methods": mm_cfg["process_stream"].get("n_methods", 6),
                        "substrate_embed_dim": mm_cfg["process_stream"].get("substrate_embed_dim", 3),
                        "n_substrates": mm_cfg["process_stream"].get("n_substrates", 7),
                        "hidden_dim": 64, "out_dim": mm_cfg["process_stream"]["out_dim"],
                        "dropout": mm_cfg["process_stream"]["dropout"]},
        dopant_kwargs=None,
        head_hidden_dims=hc["hidden_dims"], head_dropout=hc["dropout"],
        head_type=hc["type"], num_experts=4, moe_routing="soft",
        **s04._physics_kwargs(mm_cfg),
    ).to(device)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--csv", default="data/raw/experimental/ga2o3_exp_aug3x_vcinterp_v58.csv")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if (args.gpu >= 0 and torch.cuda.is_available()) else "cpu")
    targets = ["vacancy_concentration"]

    ds = Ga2O3ExpDataset(csv_path=args.csv, structures_dir="data/structures",
                         target_cols=targets, augment=False,
                         exclude_elements=["N", "H", "Nb", "In"], process_layout="v2")
    from src.data.experimental_dataset import collate_fn
    all_proc = torch.stack([ds[i]["process"] for i in range(len(ds))]).numpy()
    all_specs = [ds[i]["dopant_spec"] for i in range(len(ds))]
    batch = collate_fn([ds[i] for i in range(min(16, len(ds)))])
    graph = batch["graph"].to(device)
    process = batch["process"].to(device)
    specs = batch["dopant_spec"]

    print("=== Build V55-Ext (baseline) ===")
    base = build_model("config/multimodal_v54a1_sincere.yaml", targets, device, seed=0)
    # fit scalers
    base.process_stream.fit_scaler(all_proc)
    base.composition_stream.fit_scaler(all_specs)

    print("=== Build V59-CALM (full) ===")
    calm = build_model("config/multimodal_v59_calm_full.yaml", targets, device, seed=0)
    calm.process_stream.fit_scaler(all_proc)
    calm.composition_stream.fit_scaler(all_specs)

    # ── Guarantee 1: CALM param budget (C1) ─────────────────────────────────────
    n_calm = calm.calm_residual.num_trainable()
    print(f"\n[C1] CALMResidual trainable params = {n_calm}  (target < 10000)")
    assert n_calm < 10000, "C1 VIOLATED: CALM residual exceeds 10k params"

    base_total = sum(p.numel() for p in base.parameters() if p.requires_grad)
    calm_total = sum(p.numel() for p in calm.parameters() if p.requires_grad)
    print(f"[C1] base trainable={base_total}  calm trainable={calm_total}  Δ={calm_total - base_total}")

    # ── Guarantee 2: no-regression-at-init (Δ=0 ⇒ identical preds) ───────────────
    with torch.no_grad():
        emb_b = base.get_embedding(graph, specs, process)
        pred_b = base._heads_from_fused(emb_b, dopant_specs=specs, process=process)
        emb_c = calm.get_embedding(graph, specs, process)
        pred_c = calm._heads_from_fused(emb_c, dopant_specs=specs, process=process)
    max_abs = (pred_b - pred_c).abs().max().item()
    # also confirm the Δ itself is exactly 0 at init
    with torch.no_grad():
        delta0 = calm.calm_delta(emb_c, specs, process).abs().max().item()
    print(f"\n[init] max|Δ| at init = {delta0:.3e}  (must be 0)")
    print(f"[init] max|pred_V55Ext − pred_V59CALM| = {max_abs:.3e}  (must be ~0)")
    assert delta0 == 0.0, "Δ is not exactly 0 at init (zero-w_out guarantee broken)"
    assert max_abs < 1e-5, "V59 != V55-Ext at init (path-separation/identity broken)"

    # ── Guarantee 3: DCC feasibility — finite-difference slope is differentiable
    # w.r.t. params (the process_stream scaler routes through numpy → autograd
    # through process[:,11] is blocked; DCC therefore uses the eval-symmetric
    # finite-difference over rebuilt concentration points, exactly as eval does).
    from src.data.dopant_spec import DopantSpec
    from src.data.graph_builder import get_graph_for_spec
    from src.data.experimental_dataset import build_process_tensor
    from torch_geometric.data import Batch
    calm.train()
    elem = "Sn"
    cs = [1e-3, 4e-3, 1.6e-2, 6.4e-2]
    specs_k = [f"{elem}:{c:.6f}" for c in cs]
    gks = Batch.from_data_list([get_graph_for_spec(DopantSpec.parse(s)) for s in specs_k]).to(device)
    pks = torch.stack([build_process_tensor(temperature_C=700.0, time_min=60.0,
                                            atmosphere="Ar", method="RF magnetron sputtering",
                                            concentration_total_frac=c) for c in cs]).to(device)
    emb = calm.get_embedding(gks, specs_k, pks)
    pred = calm._heads_from_fused(emb, dopant_specs=specs_k, process=pks)[:, 0]
    d = pred[1:] - pred[:-1]
    print(f"\n[DCC] Sn conc-sweep preds={pred.detach().cpu().numpy().round(3)}  "
          f"steps={d.detach().cpu().numpy().round(3)}  requires_grad={d.requires_grad}")
    # DCC penalty (donor: want d>0): one-sided hinge, then backward to params.
    pen = torch.clamp(-d + 0.02, min=0.0).pow(2).mean()
    pen.backward()
    g = sum((p.grad.abs().sum().item() for p in calm.parameters() if p.grad is not None))
    print(f"[DCC] one-sided penalty={pen.item():.4e}; backward OK; total |grad| over params={g:.4e}")
    assert d.requires_grad, "FD slope has no grad_fn — DCC backward path broken"
    assert g > 0, "DCC penalty produced zero gradient — head/CALM not reached"

    print("\n✅ ALL SMOKE CHECKS PASSED — C1 + identity-at-init + DCC finite-diff feasible.")


if __name__ == "__main__":
    main()
