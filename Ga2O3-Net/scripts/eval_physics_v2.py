"""Physics-rule-learning evaluation — v2 (corrected rubric).

Supersedes the within-DOI FLIP gate + Tier-1C of `eval_sputter_physics.py` for
frontier decisions. Built from the 2026-05-30 validity review
(docs/physics_rubric_validity_review.md). Goal: genuinely measure whether the model
learned the MEASURED doping→(V_O / PDR) physical law, not textbook-sign compliance.

What changed vs v1 (and WHY):
  (1) DATA-REFERENCED within-DOI verdict. v1 compared sign(rho_pred) to a hardcoded
      textbook class sign and IGNORED the measured slope rho_true — so it penalized
      models that correctly fit data disagreeing with the textbook. v2 judges the model
      against rho_true (the measured concentration response); the textbook sign is reported
      separately as a data-consistency flag, never as the model's grade.
  (2) DEDUPLICATE + real-rows-only. v1 inflated group n ~3x with augmented/duplicate rows
      and let interpolated rows FABRICATE a slope on flat real data. v2 drops interp/aug
      rows, collapses to unique (doi, element, concentration) cells (mean pred), and gates
      on n_unique_real_conc >= 3.
  (3) CI-GATED + MAGNITUDE-AWARE. v1's integer FLIP count ignored the bootstrap CI (which
      flagged ~all groups ambiguous) and ignored slope magnitude (a flat prediction scored
      "OK"). v2 counts a FLIP only when the bootstrap CI confidently excludes 0 on the wrong
      side, and credits LEARNED only when direction AND magnitude match the data.
  (4) TIER-1C vs TRUTH. v1's baseline was the model's own prediction mean; for n_labeled=0
      elements the verdict was unfalsifiable. v2 anchors to the true-label mean and labels
      no-ground-truth verdicts as "prior-plausibility (unfalsifiable)".
  (5) NULL-BASELINE. A valence-class-mean predictor (no concentration response) is scored
      through the same within-element R²; the model must BEAT it to claim it learned the
      concentration law (vs merely valence ordering).
  (6) HONEST R². Raw R² reported beside the in-sample-calibrated Platt R².
  (7) SAFE MERGE. OOF↔CSV merged with a dopant_label==element assertion (catches the v1
      positional off-by-one); merges against the bundle's OWN training CSV.

The model-intrinsic counterfactual concentration sweep (the cleanest law probe, needs no
labels) is a companion: scripts/eval_counterfactual_sweep.py.

Usage:
  PYTHONPATH=. python scripts/eval_physics_v2.py <bundle_dir> --target {vacancy_concentration|photo_dark_ratio} [--csv <csv>] [--out <md>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from scripts.eval_physics_diag_table import (
    ACCEPTOR_ELEMENTS, DONOR_ELEMENTS, SUPER_DONOR_ELEMENTS,
    _expected_class, fit_sputter_platt, load_oof_with_meta,
)

PROJ = Path(__file__).resolve().parents[1]


# ── helpers ────────────────────────────────────────────────────────────────
def _spearman(x, y):
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(stats.spearmanr(x, y).statistic)


def _theil_sen(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    sl = [(y[j] - y[i]) / (x[j] - x[i])
          for i in range(len(x)) for j in range(i + 1, len(x))
          if abs(x[j] - x[i]) > 1e-12]
    return float(np.median(sl)) if sl else np.nan


def _textbook_sign(elem: str, target: str) -> int:
    """Class-based expected concentration-response sign (REPORTED, not graded)."""
    flip = (target == "photo_dark_ratio")
    if elem in ACCEPTOR_ELEMENTS:
        return +1 if flip else -1
    if elem in DONOR_ELEMENTS or elem in SUPER_DONOR_ELEMENTS:
        return -1 if flip else +1
    return 0


def _dedup_real(df: pd.DataFrame, target: str, notes_col: str | None) -> pd.DataFrame:
    """Drop interpolated/augmented rows; collapse to unique (doi,element,conc) cells.

    Returns one row per (doi, element, rounded-concentration) with mean pred / mean
    pred_platt and the measured true value. Removes the v1 ~3x duplication + interp-
    fabricated slopes.
    """
    pred = f"{target}_pred"; pred_pl = f"{target}_pred_platt"; true = f"{target}_true"
    w = df.copy()
    # drop interpolated / augmented synthetic rows from the slope test
    if notes_col and notes_col in w.columns:
        is_interp = w[notes_col].astype(str).str.contains("interp|augment|aug3x", case=False, na=False)
        w = w[~is_interp]
    w["c"] = pd.to_numeric(w.get("concentration_at%"), errors="coerce")
    w = w[w.c > 0].dropna(subset=["doi", "element", pred])
    w["cbin"] = w.c.round(6)
    agg = {pred: "mean"}
    if pred_pl in w.columns:
        agg[pred_pl] = "mean"
    if true in w.columns:
        agg[true] = "mean"
    cell = w.groupby(["doi", "element", "cbin"], as_index=False).agg(agg)
    cell["logc"] = np.log10(cell.cbin)
    return cell


# ── (1) data-referenced within-DOI concentration-law test ───────────────────
def within_doi_law_test(cell: pd.DataFrame, target: str,
                        min_unique_conc: int = 3, n_boot: int = 1000,
                        seed: int = 42) -> tuple[pd.DataFrame, dict]:
    pred_pl = f"{target}_pred_platt"
    pred = pred_pl if pred_pl in cell.columns else f"{target}_pred"
    true = f"{target}_true"
    rng = np.random.default_rng(seed)
    rows = []
    for (doi, elem), g in cell.groupby(["doi", "element"]):
        g = g.sort_values("logc")
        n_uc = g.logc.nunique()
        if n_uc < min_unique_conc:
            continue
        has_true = true in g.columns and g[true].notna().sum() >= min_unique_conc \
            and float(np.nanstd(g[true])) > 1e-9
        rho_pred = _spearman(g.logc.values, g[pred].values)
        ts_pred = _theil_sen(g.logc.values, g[pred].values)
        tb = _textbook_sign(elem, target)
        rec = dict(doi=str(doi)[:55], element=elem, n_unique_conc=int(n_uc),
                   rho_pred=rho_pred, ts_pred=ts_pred, textbook_sign=tb)
        # bootstrap CI on rho_pred over unique-conc cells
        gp = g[[pred]].assign(logc=g.logc).reset_index(drop=True)
        boots = []
        for _ in range(n_boot):
            idx = rng.choice(len(gp), len(gp), replace=True)
            sub = gp.iloc[idx]
            if sub.logc.nunique() < 2:
                continue
            rb = _spearman(sub.logc.values, sub[pred].values)
            if np.isfinite(rb):
                boots.append(rb)
        ci_lo = float(np.percentile(boots, 2.5)) if len(boots) >= 100 else np.nan
        ci_hi = float(np.percentile(boots, 97.5)) if len(boots) >= 100 else np.nan
        ci_confident = np.isfinite(ci_lo) and (ci_lo > 0 or ci_hi < 0)
        rec.update(rho_pred_ci_lo=ci_lo, rho_pred_ci_hi=ci_hi, ci_confident=bool(ci_confident))

        rho_true = _spearman(g.logc.values, g[true].values) if has_true else np.nan
        # A group can pass has_true (>=3 non-NaN, nonzero full-group std) yet still yield a
        # non-finite rho_true after dedup-to-unique-conc (e.g. constant true across the unique
        # concentrations / all-ties). Such groups have NO usable measured slope → NO_REF,
        # never FLIP. (Bug caught in v2 self-test: matlet.2017.08.032 Mg.)
        if has_true and np.isfinite(rho_true):
            ts_true = _theil_sen(g.logc.values, g[true].values)
            rec.update(rho_true=rho_true, ts_true=ts_true)
            match = np.isfinite(rho_pred) and np.sign(rho_pred) == np.sign(rho_true)
            mag_ok = (np.isfinite(ts_pred) and np.isfinite(ts_true) and abs(ts_true) > 1e-9
                      and (1/3) <= abs(ts_pred) / abs(ts_true) <= 3 and np.sign(ts_pred) == np.sign(ts_true))
            if not np.isfinite(rho_pred) or abs(rho_pred) < 1e-9:
                verdict = "AMBIGUOUS"
            elif match and ci_confident and mag_ok:
                verdict = "LEARNED"
            elif match and ci_confident:
                verdict = "SIGN_OK"          # right direction, magnitude off
            elif match:
                verdict = "SIGN_OK_WEAK"      # right direction, CI ambiguous
            elif ci_confident:
                verdict = "FLIP"              # confidently wrong vs MEASURED data
            else:
                verdict = "AMBIGUOUS"
            rec["data_vs_textbook"] = ("consistent" if (np.isfinite(rho_true) and tb != 0
                                       and np.sign(rho_true) == tb) else "data_contradicts_textbook"
                                       if (np.isfinite(rho_true) and tb != 0) else "n/a")
        else:
            # No measured multi-conc reference → cannot grade physics-learning from data.
            rec.update(rho_true=np.nan, ts_true=np.nan,
                       data_vs_textbook="no_measured_reference")
            verdict = "NO_REF"
        rec["verdict"] = verdict
        rows.append(rec)
    df = pd.DataFrame(rows).sort_values(["element", "doi"]).reset_index(drop=True) if rows else pd.DataFrame()
    def cnt(v): return int((df["verdict"] == v).sum()) if len(df) else 0
    summ = dict(
        n_groups=len(df),
        n_with_measured_ref=int((df["verdict"] != "NO_REF").sum()) if len(df) else 0,
        LEARNED=cnt("LEARNED"), SIGN_OK=cnt("SIGN_OK"), SIGN_OK_WEAK=cnt("SIGN_OK_WEAK"),
        FLIP=cnt("FLIP"), AMBIGUOUS=cnt("AMBIGUOUS"), NO_REF=cnt("NO_REF"),
    )
    return df, summ


# ── (5) within-element R² vs null (valence-class-mean) baseline ─────────────
def within_element_vs_null(df: pd.DataFrame, target: str) -> dict:
    """Within-element R² for the model vs a class-mean null (no conc response).

    The model must BEAT the null to claim it learned concentration physics rather
    than only valence ordering. Computed on labeled rows.
    """
    pred_pl = f"{target}_pred_platt"; pred = pred_pl if pred_pl in df.columns else f"{target}_pred"
    true = f"{target}_true"
    lab = df.dropna(subset=[true, pred, "element"]).copy()
    if len(lab) < 4:
        return dict(model_within_r2=np.nan, null_within_r2=np.nan, gain=np.nan, n=len(lab))
    # within-element: subtract per-element mean from both true and pred
    lab["t_dm"] = lab[true] - lab.groupby("element")[true].transform("mean")
    lab["p_dm"] = lab[pred] - lab.groupby("element")[pred].transform("mean")
    ss_res = float(((lab.t_dm - lab.p_dm) ** 2).sum())
    ss_tot = float((lab.t_dm ** 2).sum())
    model_r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    # null = class mean → within-element prediction is the element mean → p_dm == 0
    null_r2 = 0.0  # by construction the class-mean null has within-element R²=0
    return dict(model_within_r2=float(model_r2), null_within_r2=null_r2,
                gain=float(model_r2 - null_r2), n=int(len(lab)),
                interpretation="model_within_r2 > 0 means it explains within-element "
                               "(concentration) variance the valence-class-mean null cannot")


# ── (4) Tier-1C anchored to TRUE-label mean ─────────────────────────────────
def tier1c_vs_truth(df: pd.DataFrame, target: str, std_scale: float = 1.0) -> pd.DataFrame:
    pred_pl = f"{target}_pred_platt"; pred = pred_pl if pred_pl in df.columns else f"{target}_pred"
    true = f"{target}_true"; std = f"{target}_std"
    flip = (target == "photo_dark_ratio")
    w = df.dropna(subset=[pred]).copy()
    lab = w.dropna(subset=[true])
    if len(lab) < 2:
        return pd.DataFrame()
    baseline = float(lab[true].mean())  # GROUND-TRUTH mean (v1 used model's own pred mean)
    rows = []
    for elem in sorted(w.element.dropna().unique(), key=lambda e: -len(w[w.element == e])):
        sub = w[w.element == elem]
        if elem == "undoped":
            continue
        n_lab = int(sub[true].notna().sum())
        pmean = float(sub[pred].mean())
        pstd = float(sub[std].mean()) * std_scale if std in sub.columns else 0.0
        cls = _expected_class(elem)
        exp = ("above" if (cls in ("donor", "super_donor")) ^ flip else
               "below" if cls == "acceptor" else "≈")
        # honest verdict: only grade against truth if labels exist
        if n_lab == 0:
            # unfalsifiable: report direction vs the model's plausibility only
            side = "above" if pmean > baseline else "below"
            verdict = f"prior_plausibility({side})"
        else:
            d = pmean - baseline
            tol = 0.20
            if exp == "≈":
                verdict = "OK" if abs(d) < tol else "OFFSET" if abs(d) > tol + pstd else "ambiguous"
            elif exp == "above":
                verdict = "OK" if pmean - pstd > baseline + tol else "FLIP" if pmean + pstd < baseline - tol else "ambiguous"
            else:
                verdict = "OK" if pmean + pstd < baseline - tol else "FLIP" if pmean - pstd > baseline + tol else "ambiguous"
        rows.append(dict(element=elem, n_labeled=n_lab, pred_mean=round(pmean, 3),
                         baseline_true_mean=round(baseline, 3), d_vs_truth=round(pmean - baseline, 3),
                         expected=exp, verdict=verdict))
    return pd.DataFrame(rows)


def _oof_aggregate_r2(df: pd.DataFrame, target: str, platt: bool) -> tuple[float, int]:
    col = f"{target}_pred_platt" if platt else f"{target}_pred"
    true = f"{target}_true"
    s = df.dropna(subset=[col, true])
    if len(s) < 2:
        return np.nan, 0
    t = s[true].values; p = s[col].values
    return float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()), len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--target", required=True, choices=["vacancy_concentration", "photo_dark_ratio"])
    ap.add_argument("--csv", default=None)
    ap.add_argument("--scope", default="all", choices=["all", "sputter"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bundle = Path(args.bundle)
    df = load_oof_with_meta(bundle, args.csv, target=args.target)
    # bring in notes column for interp filtering (load_oof_with_meta drops it) — re-merge
    if args.csv or True:
        from scripts.eval_physics_diag_table import _resolve_csv_from_bundle
        csv = args.csv or _resolve_csv_from_bundle(bundle)
        if csv:
            src = pd.read_csv(csv)
            if "usable_flag" in src.columns:
                src = src[src["usable_flag"] != "exotic_skip"].reset_index(drop=True)
            src = src[~src["element"].isin(["N", "H", "Nb", "In"])].reset_index(drop=True)
            src["sample_idx"] = np.arange(len(src))
            for c in ["notes", "concentration_at%"]:
                if c in src.columns and c not in df.columns:
                    df = df.merge(src[["sample_idx", c]], on="sample_idx", how="left")
    # SAFE-MERGE assertion (catch the v1 positional off-by-one)
    n_mis = 0
    if "element" in df.columns and "dopant_label" in df.columns:
        m = df.dropna(subset=["element", "dopant_label"])
        m = m[m.dopant_label != "undoped"]
        n_mis = int((m.element != m.dopant_label).sum())

    if args.scope == "sputter" and "is_sputter" in df.columns:
        df = df[df.is_sputter]

    slope, intercept = fit_sputter_platt(df, target=args.target)
    df[f"{args.target}_pred_platt"] = df[f"{args.target}_pred"] * slope + intercept

    cell = _dedup_real(df, args.target, "notes")
    wd_df, wd = within_doi_law_test(cell, args.target)
    we = within_element_vs_null(df, args.target)
    t1c = tier1c_vs_truth(df, args.target)
    r2_raw, n_raw = _oof_aggregate_r2(df, args.target, platt=False)
    r2_pl, n_pl = _oof_aggregate_r2(df, args.target, platt=True)

    out = {
        "bundle": str(bundle), "target": args.target, "scope": args.scope,
        "safe_merge_mismatches": n_mis,
        "oof_r2_raw": r2_raw, "oof_r2_platt_INSAMPLE": r2_pl, "n_labeled": n_raw,
        "platt_slope": slope, "platt_intercept": intercept,
        "within_doi_law": wd,
        "within_element_vs_null": we,
        "within_doi_groups": wd_df.to_dict("records") if len(wd_df) else [],
        "tier1c": t1c.to_dict("records") if len(t1c) else [],
    }
    # console + md
    print(json.dumps({k: v for k, v in out.items() if k not in ("within_doi_groups", "tier1c")}, indent=2, default=str))
    if len(wd_df):
        print("\n[within-DOI concentration-law (data-referenced)]")
        cols = [c for c in ["doi", "element", "n_unique_conc", "rho_true", "rho_pred",
                            "ts_pred", "ci_confident", "textbook_sign", "data_vs_textbook", "verdict"] if c in wd_df.columns]
        print(wd_df[cols].to_string(index=False))
    if len(t1c):
        print("\n[Tier-1C vs TRUTH]")
        print(t1c.to_string(index=False))

    outpath = Path(args.out) if args.out else bundle / f"physics_v2_{args.target}_{args.scope}.json"
    outpath.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {outpath}")
    if n_mis:
        print(f"  ⚠ {n_mis} dopant_label!=element mismatches (v1 positional-merge drift) — Tier-1C for affected rare elements unreliable")


if __name__ == "__main__":
    main()
