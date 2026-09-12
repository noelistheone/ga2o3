"""Counterfactual concentration-sweep — the model-intrinsic physics-law probe (no labels).

The cleanest test of whether the model LEARNED the doping concentration law: hold the
process vector fixed (same method/atmosphere/temperature), sweep ONLY the dopant
concentration on a dense grid through the trained model, and read off the predicted
response direction + monotonicity. Because everything except concentration is held
constant, the predicted slope isolates the learned concentration response — with NO
measurement noise and NO label requirement (works for every element, incl. Tier-1C
elements that have no measured multi-concentration data).

Physics law being tested (β-Ga2O3, Kröger-Vink charge compensation):
  - V_O (vacancy_concentration): acceptors (Mg/Zn/Cu) SUPPRESS V_O with rising conc
    (slope<0); donors/super-donors (Si/Sn/Ge/Ti/Ta/W) ENHANCE V_O (slope>0).
  - PDR (photo_dark_ratio): inverse direction (acceptors raise PDR, donors lower it),
    since PDR is anti-correlated with the V_O trap density.
Isovalent/anion (Al/Fe/Cr/Bi/Sb/F): ≈ flat (no first-order charge-compensation effect).

Verdict per element: LEARNED_LAW (correct sign + monotonic + non-trivial magnitude),
FLAT (no learned concentration response — |slope|≈0), WRONG_DIRECTION (confidently
opposite to the law). A model that merely memorized per-paper offsets or follows valence
ordering without a concentration response shows FLAT; only a model that internalized the
law shows LEARNED_LAW across elements.

Usage:
  PYTHONPATH=. python scripts/eval_counterfactual_sweep.py \
     --ckpt results/<bundle>/stage3_model.pt --config config/<multimodal>.yaml \
     --target {vacancy_concentration|photo_dark_ratio} --out <json>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parents[1]

from scripts.eval_physics_diag_table import (  # noqa: E402
    ACCEPTOR_ELEMENTS, DONOR_ELEMENTS, SUPER_DONOR_ELEMENTS, ISOVALENT_ELEMENTS,
)

# Elements to probe: labeled + Tier-1C, restricted to those with a generatable graph.
PROBE_ELEMENTS = sorted(
    set(ACCEPTOR_ELEMENTS) | set(DONOR_ELEMENTS) | set(SUPER_DONOR_ELEMENTS)
    | {"Fe", "Cr", "Bi", "Sb", "Al", "B", "V", "Ti", "Ge"}
)


def _expected_slope_sign(elem: str, target: str) -> int:
    flip = (target == "photo_dark_ratio")
    if elem in ACCEPTOR_ELEMENTS:
        return +1 if flip else -1
    if elem in DONOR_ELEMENTS or elem in SUPER_DONOR_ELEMENTS:
        return -1 if flip else +1
    return 0  # isovalent / anion → ≈ flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--target", required=True, choices=["vacancy_concentration", "photo_dark_ratio"])
    ap.add_argument("--n-grid", type=int, default=10)
    ap.add_argument("--n-mc", type=int, default=20)
    ap.add_argument("--c-min", type=float, default=1e-3)
    ap.add_argument("--c-max", type=float, default=8e-2)
    ap.add_argument("--temperature-c", type=float, default=700.0)
    ap.add_argument("--atmosphere", default="Ar")
    ap.add_argument("--method", default="RF magnetron sputtering")
    ap.add_argument("--flat-tol", type=float, default=0.15,
                    help="|Spearman rho| below this on the sweep = FLAT (no learned response)")
    ap.add_argument("--gpu", type=int, default=0, help="GPU index (-1 = CPU). Phase 59 bug #9.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    s = importlib.util.spec_from_file_location("s04", str(PROJ / "scripts/04_finetune_predict.py"))
    s04 = importlib.util.module_from_spec(s); s.loader.exec_module(s04)
    from src.data.dopant_spec import DopantSpec
    from src.data.graph_builder import get_graph_for_spec
    from src.data.experimental_dataset import build_process_tensor

    import torch
    model = s04.load_model(args.ckpt, args.config)
    device = torch.device(f"cuda:{args.gpu}" if (args.gpu >= 0 and torch.cuda.is_available())
                          else "cpu")
    try:
        model.to(device)
    except Exception:
        device = torch.device("cpu")
    tgt_idx = None
    try:
        tgt_idx = list(model.target_cols).index(args.target)
    except Exception:
        pass

    # Robustness: sweep over a GRID of representative process conditions (not one slice),
    # so a single unrepresentative process can't drive the verdict. Report the modal
    # direction + the fraction of conditions where the law holds.
    PROC_CONDITIONS = [
        ("RF magnetron sputtering", "Ar", 500.0),
        ("RF magnetron sputtering", "Ar", 900.0),
        ("RF magnetron sputtering", "O2", 700.0),
        ("RF magnetron sputtering", "Ar:O2=1:1", 700.0),
        ("MOCVD", "O2", 700.0),
        ("PLD", "Ar", 600.0),
    ]
    grid = np.logspace(np.log10(args.c_min), np.log10(args.c_max), args.n_grid)
    logc = np.log10(grid)
    # Phase 59 fix (v59-impl-verify bug #1): build the process tensor PER concentration
    # (concentration_total_frac=c) so the log10-concentration slot process[:,11] varies
    # with the sweep — matching exactly how the DCC training loss varies it. The old code
    # built one process per CONDITION (process[:,11] pinned at the −4 floor), which left the
    # process→proc_emb→head concentration channel un-probed and broke train/eval symmetry.
    rows = []
    for elem in PROBE_ELEMENTS:
        exp = _expected_slope_sign(elem, args.target)
        # build the per-concentration graphs once (graph is process-independent)
        specs, graphs, ok = [], [], True
        for c in grid:
            spec_str = f"{elem}:{c:.6f}"
            try:
                graphs.append(get_graph_for_spec(DopantSpec.parse(spec_str)))
                specs.append(spec_str)
            except Exception:
                ok = False; break
        if not ok:
            rows.append(dict(element=elem, status="ungeneratable")); continue
        # sweep concentration at EACH process condition
        rhos, spans, monos = [], [], []
        for (meth, atm, T) in PROC_CONDITIONS:
            preds = []
            for spec_str, g, c in zip(specs, graphs, grid):
                proc = build_process_tensor(temperature_C=T, time_min=60.0, atmosphere=atm,
                                            method=meth,
                                            concentration_total_frac=float(c)).unsqueeze(0).to(device)
                mean, _ = model.mc_predict(g.to(device), [spec_str], proc, n_passes=args.n_mc)
                if torch.is_tensor(mean):
                    mean = mean.detach().cpu()
                arr = np.asarray(mean).flatten()
                preds.append(float(arr[tgt_idx]) if (tgt_idx is not None and arr.size > 1) else float(arr[0]))
            preds = np.asarray(preds)
            r = float(stats.spearmanr(logc, preds).statistic) if np.std(preds) > 1e-9 else 0.0
            d = np.diff(preds); dom = np.sign(np.sum(d)) if np.any(d) else 0.0
            rhos.append(r); spans.append(float(preds.max() - preds.min()))
            monos.append(float(np.mean(np.sign(d) == dom)) if dom != 0 else 0.0)
        rho = float(np.median(rhos))            # median direction across process conditions
        span = float(np.median(spans))
        mono_frac = float(np.median(monos))
        # fraction of process conditions whose sweep matches the textbook law direction
        law_frac = float(np.mean([np.sign(r) == exp for r in rhos])) if exp != 0 else float("nan")
        cls = ("acceptor" if elem in ACCEPTOR_ELEMENTS else
               "donor" if elem in DONOR_ELEMENTS else
               "super_donor" if elem in SUPER_DONOR_ELEMENTS else
               "isovalent/anion")
        FLAT_SPAN = 0.10  # below this standardized-span = genuinely no response
        if exp == 0:
            verdict = "OK_FLAT" if abs(rho) < args.flat_tol else "UNEXPECTED_RESPONSE"
        elif span < FLAT_SPAN:
            verdict = "FLAT"               # model produces ~no concentration response at all
        elif abs(rho) < args.flat_tol:
            verdict = "NONMONOTONIC"        # responds in magnitude but no consistent direction
        elif np.sign(rho) == exp and mono_frac >= 0.7 and (np.isnan(law_frac) or law_frac >= 0.6):
            verdict = "LEARNED_LAW"         # correct direction, monotonic, robust across processes
        elif np.sign(rho) == exp:
            verdict = "RIGHT_SIGN_FRAGILE"  # right median direction but non-mono or process-fragile
        else:
            verdict = "WRONG_DIRECTION"
        rows.append(dict(element=elem, status="ok", valence_class=cls,
                         expected_sign=exp, sweep_rho_median=round(rho, 3),
                         law_frac_across_processes=None if np.isnan(law_frac) else round(law_frac, 2),
                         monotonic_frac=round(mono_frac, 2), span_std_median=round(span, 3),
                         verdict=verdict))

    graded = [r for r in rows if r.get("status") == "ok" and r["expected_sign"] != 0]
    summ = dict(
        target=args.target, n_probed=len(rows),
        n_graded=len(graded),
        LEARNED_LAW=sum(r["verdict"] == "LEARNED_LAW" for r in graded),
        RIGHT_SIGN_FRAGILE=sum(r["verdict"] == "RIGHT_SIGN_FRAGILE" for r in graded),
        FLAT=sum(r["verdict"] == "FLAT" for r in graded),
        NONMONOTONIC=sum(r["verdict"] == "NONMONOTONIC" for r in graded),
        WRONG_DIRECTION=sum(r["verdict"] == "WRONG_DIRECTION" for r in graded),
        isovalent_OK_FLAT=sum(r.get("verdict") == "OK_FLAT" for r in rows),
        isovalent_UNEXPECTED=sum(r.get("verdict") == "UNEXPECTED_RESPONSE" for r in rows),
        # Phase 59 fix (bug #3): aggregate monotonicity so the §9 gate can read the
        # design's "RIGHT_SIGN_FRAGILE + mono increase" fallback (condition 2).
        mono_frac_graded_mean=(round(float(np.mean([r["monotonic_frac"] for r in graded])), 3)
                               if graded else None),
        right_sign_count=sum((r["verdict"] in ("LEARNED_LAW", "RIGHT_SIGN_FRAGILE")) for r in graded),
    )
    out = dict(ckpt=args.ckpt, config=args.config,
               process=dict(method=args.method, atmosphere=args.atmosphere,
                            temperature_c=args.temperature_c),
               summary=summ, per_element=rows)
    print(json.dumps(summ, indent=2))
    print("\n[per-element counterfactual concentration sweep]")
    for r in rows:
        if r.get("status") == "ok":
            print(f"  {r['element']:4s} {r['valence_class']:16s} exp={r['expected_sign']:+d} "
                  f"rho_med={r['sweep_rho_median']:+.2f} law_frac={r['law_frac_across_processes']} "
                  f"mono={r['monotonic_frac']:.2f} span={r['span_std_median']:.2f} → {r['verdict']}")
        else:
            print(f"  {r['element']:4s} ({r['status']})")
    outpath = Path(args.out) if args.out else Path(args.ckpt).parent / f"counterfactual_sweep_{args.target}.json"
    outpath.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
