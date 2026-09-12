"""Phase 51+ — Enhanced sputter-physics evaluation.

Addresses six known weaknesses in `eval_physics_diag_table.py`:

  1. Within-DOI per-element slope ρ — the cleanest physics test (same paper,
     same atmosphere/temperature, only concentration varies). Currently
     missing entirely from evaluation.
  2. Default scope = sputter (not "all") — sputter is the actual prediction
     scope, "all" mixes methods and confounds Tier-1A signs.
  3. Outlier-robust per-element parity (IQR filter, optionally configurable).
     Single-point outliers (e.g. Sn=10.13 in "Original CSV") were silently
     flipping per-element correlations.
  4. R² decomposition into BETWEEN-element and WITHIN-element components.
     R² ≈ 0.6 alone is mostly cross-element ordering; this hides whether
     within-element concentration response is captured.
  5. Theil-Sen robust slope estimator alongside Spearman ρ — less sensitive
     to single-point flips.
  6. Tier-1C with ensemble-std interval check (not just point mean).
     N=1 super-donor verdicts depend on a single number; reporting the
     prediction interval makes the uncertainty explicit.

Usage:
    conda activate ga2o3
    python scripts/eval_sputter_physics.py results/<bundle_dir> \\
        [<full_csv>] [--target vacancy_concentration|photo_dark_ratio]

Output:
  results/<bundle_dir>/sputter_physics_eval.md  — full markdown report
  stdout — concise summary for quick comparison
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

# Re-use the existing loader and Platt fit
from scripts.eval_physics_diag_table import (
    load_oof_with_meta, fit_sputter_platt,
    ELEM_VALENCE, ACCEPTOR_ELEMENTS, DONOR_ELEMENTS, SUPER_DONOR_ELEMENTS,
    ISOVALENT_ELEMENTS, _expected_class, _atm_group, _target_lit_rho,
)


# ────────────────────── helpers ──────────────────────

def _spearman_safe(x, y):
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def _theil_sen_slope(x, y):
    """Robust Theil-Sen slope = median of all pairwise slopes."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 2:
        return float("nan")
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if abs(dx) < 1e-12:
                continue
            slopes.append((y[j] - y[i]) / dx)
    if not slopes:
        return float("nan")
    return float(np.median(slopes))


def _iqr_outlier_mask(values: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Tukey IQR outlier mask — True for kept (non-outlier) points."""
    if len(values) < 4:
        return np.ones_like(values, dtype=bool)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (values >= lo) & (values <= hi)


def _r2(t, p):
    if len(t) < 2 or np.var(t) < 1e-12:
        return float("nan")
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ────────────────────── core metrics ──────────────────────

def within_doi_slope_test(df_sp: pd.DataFrame, target: str
                           ) -> tuple[pd.DataFrame, dict]:
    """[The primary physics test V5 used as a TRAINING loss but never
    reported as an eval metric.]

    For every (DOI, element) sputter group with ≥3 rows, ≥2 unique
    concentrations, and at least 0.05 dec-conc spread: compute Spearman ρ
    of predicted log(V_O) vs log(concentration) and compare to lit sign.

    This is the cleanest "did the model learn within-paper concentration
    response" measure — same paper means same method, atmosphere, temp,
    so any slope difference is purely concentration-driven.
    """
    pred_col = f"{target}_pred_platt"
    true_col = f"{target}_true"
    work = df_sp.dropna(subset=["doi", "dopant_label", "concentration_at%",
                                 true_col, pred_col]).copy()
    work["c"] = pd.to_numeric(work["concentration_at%"], errors="coerce")
    work = work[work.c > 0]
    work["logc"] = np.log10(work.c)
    lit_rho = _target_lit_rho(target)

    rows = []
    for (doi, elem), grp in work.groupby(["doi", "dopant_label"]):
        if len(grp) < 3:
            continue
        n_unique_c = grp.c.nunique()
        if n_unique_c < 2:
            continue
        spread = grp.logc.max() - grp.logc.min()
        if spread < 0.05:
            continue
        rho_true = _spearman_safe(grp.logc, grp[true_col])
        rho_pred = _spearman_safe(grp.logc, grp[pred_col])
        ts_pred = _theil_sen_slope(grp.logc, grp[pred_col])
        v = ELEM_VALENCE.get(elem, 99)
        flip_pdr = (target == "photo_dark_ratio")
        if elem in ACCEPTOR_ELEMENTS:
            expected_sign = +1 if flip_pdr else -1
        elif elem in DONOR_ELEMENTS or elem in SUPER_DONOR_ELEMENTS:
            expected_sign = -1 if flip_pdr else +1
        else:
            expected_sign = 0
        if not np.isfinite(rho_pred):
            sign_ok = "(NaN)"
        elif expected_sign == 0:
            sign_ok = "—"
        elif np.sign(rho_pred) == np.sign(expected_sign):
            sign_ok = "OK"
        else:
            sign_ok = "**FLIP**"
        rows.append(dict(
            doi=str(doi)[:60], element=elem, valence=v,
            n=int(len(grp)), n_unique_c=int(n_unique_c),
            logc_spread=float(spread),
            rho_true=rho_true, rho_pred=rho_pred,
            ts_pred=ts_pred,
            rho_lit=lit_rho.get(elem, np.nan),
            sign_match=sign_ok,
        ))
    df = pd.DataFrame(rows).sort_values(["element", "doi"]).reset_index(drop=True)
    summary = dict(
        n_groups=len(df),
        ok=int((df["sign_match"] == "OK").sum()) if len(df) > 0 else 0,
        flip=int(df["sign_match"].str.contains("FLIP", regex=False).sum())
            if len(df) > 0 else 0,
        nan=int((df["sign_match"] == "(NaN)").sum()) if len(df) > 0 else 0,
        skipped=int((df["sign_match"] == "—").sum()) if len(df) > 0 else 0,
    )
    return df, summary


def bootstrap_within_doi_ci(df_sp: pd.DataFrame, target: str,
                             n_boot: int = 1000, seed: int = 42,
                             ci_pct: float = 95.0) -> pd.DataFrame:
    """Bootstrap CI for per-(DOI, element) ρ_pred — gives statistical
    significance to the within-DOI slope test.

    Without CI, "V5 2/4 OK vs V18b 3/4 OK" is just 4 binary outcomes —
    cannot distinguish real model improvement from noise. This block
    resamples with replacement *within each (DOI, element) group* to
    derive empirical CI on ρ_pred.
    """
    pred_col = f"{target}_pred_platt"
    true_col = f"{target}_true"
    rng = np.random.default_rng(seed)
    work = df_sp.dropna(subset=["doi", "dopant_label", "concentration_at%",
                                 pred_col, true_col]).copy()
    work["c"] = pd.to_numeric(work["concentration_at%"], errors="coerce")
    work = work[work.c > 0]
    work["logc"] = np.log10(work.c)
    lit_rho = _target_lit_rho(target)
    lo_pct = (100.0 - ci_pct) / 2.0
    hi_pct = 100.0 - lo_pct

    rows = []
    for (doi, elem), grp in work.groupby(["doi", "dopant_label"]):
        if len(grp) < 3 or grp.c.nunique() < 2:
            continue
        spread = grp.logc.max() - grp.logc.min()
        if spread < 0.05:
            continue
        n = len(grp)
        rho_boots = []
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            sub = grp.iloc[idx]
            if sub.c.nunique() < 2 or np.std(sub[pred_col]) < 1e-9:
                continue
            rho_b = _spearman_safe(sub.logc.to_numpy(),
                                    sub[pred_col].to_numpy())
            if np.isfinite(rho_b):
                rho_boots.append(rho_b)
        if len(rho_boots) < 100:
            continue
        rho_arr = np.asarray(rho_boots)
        rho_point = _spearman_safe(grp.logc.to_numpy(),
                                    grp[pred_col].to_numpy())
        # Sign-stability: how often does bootstrap agree with point sign?
        if abs(rho_point) > 1e-9:
            sign_stability = float(np.mean(np.sign(rho_arr) == np.sign(rho_point)))
        else:
            sign_stability = float("nan")
        rows.append(dict(
            doi=str(doi)[:55], element=elem, n=int(n),
            rho_point=rho_point,
            rho_lo=float(np.percentile(rho_arr, lo_pct)),
            rho_hi=float(np.percentile(rho_arr, hi_pct)),
            sign_stability=sign_stability,
            rho_lit=lit_rho.get(elem, np.nan),
        ))
    return pd.DataFrame(rows).sort_values(["element", "doi"]).reset_index(drop=True)


def mc_dropout_calibration(df_eval: pd.DataFrame, target: str,
                            target_coverage: float = 0.68) -> dict:
    """MC-dropout std calibration audit.

    Theoretical Gaussian: 68% of true values fall within pred ± 1σ.
    If observed coverage at 1σ != 68%, std is mis-calibrated — typical
    failure mode is OVER-estimation (coverage > 68%), making Tier-1C
    interval verdicts overly lenient ("ambiguous" verdicts mask real
    sign info).

    Returns:
      coverage_at_1sigma, coverage_at_2sigma — empirical coverage rates
      scale_for_68pct — scale factor s such that 68% of |err|/std ≤ s
                        (>1 means std is under-estimated — rare;
                         <1 means std is over-estimated — common)
      cal_error_1sigma, cal_error_2sigma — |obs - theoretical|
    """
    pred_col = f"{target}_pred_platt"
    true_col = f"{target}_true"
    std_col = f"{target}_std"
    if std_col not in df_eval.columns:
        return None
    sub = df_eval.dropna(subset=[pred_col, true_col, std_col])
    if len(sub) < 5:
        return None
    abs_err = np.abs(sub[pred_col].to_numpy() - sub[true_col].to_numpy())
    std = sub[std_col].to_numpy()
    std_safe = np.maximum(std, 1e-9)

    coverage_1 = float(np.mean(abs_err <= std_safe))
    coverage_2 = float(np.mean(abs_err <= 2 * std_safe))
    norm_err = abs_err / std_safe
    s_optimal = float(np.percentile(norm_err, target_coverage * 100.0))

    return dict(
        n=int(len(sub)),
        coverage_at_1sigma=coverage_1,
        coverage_at_2sigma=coverage_2,
        cal_error_1sigma=float(abs(coverage_1 - 0.68)),
        cal_error_2sigma=float(abs(coverage_2 - 0.95)),
        scale_for_68pct=s_optimal,
        is_overconfident=(coverage_1 < 0.68 - 0.05),  # std too small
        is_underconfident=(coverage_1 > 0.68 + 0.05),  # std too large
    )


def per_element_robust_parity(df_sp: pd.DataFrame, target: str
                                ) -> pd.DataFrame:
    """Per-element parity Pearson r WITH and WITHOUT IQR-1.5 outlier
    filtering on the *true* value distribution. Surfaces single-point
    outliers that silently flip slopes (e.g. the Sn=10.13 anomaly).
    """
    pred_col = f"{target}_pred_platt"
    true_col = f"{target}_true"
    work = df_sp.dropna(subset=[pred_col, true_col]).copy()
    rows = []
    for elem in sorted(work.dopant_label.unique(),
                       key=lambda e: -len(work[work.dopant_label == e])):
        sub = work[work.dopant_label == elem]
        if len(sub) < 3:
            continue
        y_t = sub[true_col].to_numpy()
        y_p = sub[pred_col].to_numpy()
        r_raw, r2_raw, mae_raw = _correlation_triplet(y_t, y_p)
        keep = _iqr_outlier_mask(y_t, k=1.5)
        n_clean = int(keep.sum())
        if n_clean >= 3 and n_clean < len(sub):
            r_clean, r2_clean, mae_clean = _correlation_triplet(
                y_t[keep], y_p[keep])
        else:
            r_clean, r2_clean, mae_clean = float("nan"), float("nan"), float("nan")
        rows.append(dict(
            element=elem, valence=ELEM_VALENCE.get(elem, 99),
            n=int(len(sub)),
            r_raw=r_raw, r2_raw=r2_raw, mae_raw=mae_raw,
            n_clean=n_clean,
            r_clean=r_clean, r2_clean=r2_clean, mae_clean=mae_clean,
            n_outliers_excluded=int(len(sub) - n_clean),
        ))
    return pd.DataFrame(rows).sort_values(["valence", "element"]).reset_index(drop=True)


def _correlation_triplet(t, p):
    if len(t) < 2 or np.std(t) < 1e-12:
        return float("nan"), float("nan"), float("nan")
    r = float(stats.pearsonr(t, p).statistic)
    r2 = _r2(t, p)
    mae = float(np.mean(np.abs(t - p)))
    return r, r2, mae


def r2_variance_decomposition(df_sp: pd.DataFrame, target: str
                                ) -> dict:
    """Decompose R² into:
      - between_r2: variance explained by per-element MEAN predictions vs
                    per-element MEAN truths
      - within_r2:  variance explained by deviations from per-element mean
                    (i.e. concentration response within element)

    A model can have R²=0.6 mostly from cross-element ordering (acceptor
    vs donor) while having within-element R² near 0 — meaning it doesn't
    capture the concentration response within any single element. This
    decomposition exposes that.
    """
    pred_col = f"{target}_pred_platt"
    true_col = f"{target}_true"
    work = df_sp.dropna(subset=[pred_col, true_col]).copy()
    if len(work) < 4:
        return dict(between_r2=float("nan"), within_r2=float("nan"))

    y_t = work[true_col].to_numpy(float)
    y_p = work[pred_col].to_numpy(float)
    overall_r2 = _r2(y_t, y_p)
    overall_r = float(stats.pearsonr(y_t, y_p).statistic) if np.std(y_t) > 0 else float("nan")
    overall_mae = float(np.mean(np.abs(y_t - y_p)))

    # Element means
    elem_means = work.groupby("dopant_label").agg(
        mean_t=(true_col, "mean"),
        mean_p=(pred_col, "mean"),
        n=(true_col, "count"),
    ).reset_index()
    if len(elem_means) >= 2:
        # Weight each element mean by its sample count
        w = elem_means["n"].to_numpy(float)
        mt = elem_means["mean_t"].to_numpy(float)
        mp = elem_means["mean_p"].to_numpy(float)
        # Weighted between-component R²
        ss_t = float(np.sum(w * (mt - np.average(mt, weights=w)) ** 2))
        ss_r = float(np.sum(w * (mp - mt) ** 2))
        between_r2 = 1.0 - ss_r / ss_t if ss_t > 1e-12 else float("nan")
    else:
        between_r2 = float("nan")

    # Within-element residuals
    work2 = work.merge(elem_means, on="dopant_label", how="left")
    res_t = work2[true_col].to_numpy(float) - work2["mean_t"].to_numpy(float)
    res_p = work2[pred_col].to_numpy(float) - work2["mean_p"].to_numpy(float)
    within_r2 = _r2(res_t, res_p)

    return dict(
        n=int(len(work)),
        overall_r2=overall_r2,
        overall_r=overall_r,
        overall_mae=overall_mae,
        n_elements=int(len(elem_means)),
        between_r2=between_r2,
        within_r2=within_r2,
    )


def tier1c_interval_check(df_full: pd.DataFrame, target: str,
                            std_scale: float = 1.0) -> pd.DataFrame:
    """Tier-1C with prediction-interval check (not just point mean).

    For each unsupervised element (no labels), compute:
      pred_mean, pred_std (from MC dropout / ensemble),
      d_vs_baseline = pred_mean - labeled_overall_mean

    Verdict:
      "above"     if pred_mean - pred_std > baseline + tol
      "below"     if pred_mean + pred_std < baseline - tol
      "ambiguous" if interval crosses baseline ± tol
      "≈"         if expected ≈ and predicted within tol of baseline
    """
    pred_col = f"{target}_pred_platt"
    std_col = f"{target}_std"
    true_col = f"{target}_true"
    flip_pdr = (target == "photo_dark_ratio")

    work = df_full.dropna(subset=[pred_col]).copy()
    labeled = work.dropna(subset=[true_col])
    if len(labeled) >= 2:
        baseline = float(labeled[pred_col].mean())
        baseline_label = f"labeled-overall mean (N={len(labeled)})"
    else:
        baseline = float(work[pred_col].median())
        baseline_label = f"global median (N={len(work)})"

    rows = []
    for elem in sorted(work.dopant_label.unique(),
                       key=lambda e: -len(work[work.dopant_label == e])):
        sub = work[work.dopant_label == elem]
        if elem == "undoped" or len(sub) < 1:
            continue
        cls = _expected_class(elem)
        n_total = int(len(sub))
        n_labeled = int(sub[true_col].notna().sum())
        pred_mean = float(sub[pred_col].mean())
        pred_std_raw = float(sub[std_col].mean()) if std_col in sub.columns else float("nan")
        pred_std = pred_std_raw * std_scale  # Apply calibration scale
        d = pred_mean - baseline

        if cls == "acceptor":
            expected = "above" if flip_pdr else "below"
        elif cls in ("donor", "super_donor"):
            expected = "below" if flip_pdr else "above"
        elif cls in ("isovalent", "anion"):
            expected = "≈"
        else:
            expected = "?"

        # Interval verdict (use pred_std as 1σ interval)
        tol = 0.20  # tolerance for "≈" verdict
        if not np.isfinite(pred_std):
            pred_std = 0.0
        if expected == "≈":
            if abs(d) < tol:
                verdict = "OK"
            elif abs(d) < tol + pred_std:
                verdict = "ambiguous"
            else:
                verdict = "**OFFSET**"
        elif expected == "above":
            if pred_mean - pred_std > baseline + tol:
                verdict = "OK"
            elif pred_mean + pred_std < baseline - tol:
                verdict = "**FLIP**"
            else:
                verdict = "ambiguous"
        elif expected == "below":
            if pred_mean + pred_std < baseline - tol:
                verdict = "OK"
            elif pred_mean - pred_std > baseline + tol:
                verdict = "**FLIP**"
            else:
                verdict = "ambiguous"
        else:
            verdict = "—"

        rows.append(dict(
            element=elem, valence=ELEM_VALENCE.get(elem, 99), class_=cls,
            n_total=n_total, n_labeled=n_labeled,
            pred_mean=pred_mean, pred_std=pred_std,
            d_vs_baseline=d, expected=expected, verdict=verdict,
        ))
    df = pd.DataFrame(rows).sort_values(["valence", "element"]).reset_index(drop=True)
    df.attrs["baseline"] = baseline
    df.attrs["baseline_label"] = baseline_label
    return df


# ────────────────────── reporting ──────────────────────

def _df_to_md(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return "_(no data)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append("(N/A)" if np.isnan(v) else
                             (f"{v:+.3f}" if abs(v) < 100 else f"{v:.3f}"))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bundle_dir")
    p.add_argument("full_csv", nargs="?", default=None)
    p.add_argument("--target", default="vacancy_concentration",
                   choices=["vacancy_concentration", "photo_dark_ratio"])
    p.add_argument("--scope", default="sputter", choices=["sputter", "all"],
                   help="Default sputter — sputter is the actual prediction "
                        "scope. 'all' available for diagnostic only.")
    p.add_argument("--no-write", action="store_true",
                   help="Print only, do not save markdown file")
    args = p.parse_args()

    bundle = Path(args.bundle_dir)
    if not bundle.is_absolute():
        bundle = PROJ / bundle

    df_full = load_oof_with_meta(bundle, args.full_csv, target=args.target)

    # Authoritative ground-truth: source CSV. If a label was retroactively
    # masked (e.g. dimension errors corrected in source), enforce NaN here
    # regardless of what was saved in OOF at training time. This makes
    # post-hoc data fixes effective without retraining.
    if args.full_csv is not None:
        src_df = pd.read_csv(args.full_csv)
        if "usable_flag" in src_df.columns:
            src_df = src_df[src_df["usable_flag"] != "exotic_skip"].reset_index(drop=True)
        src_df = src_df[~src_df["element"].isin(["N", "H", "Nb", "In"])].reset_index(drop=True)
        src_df["sample_idx"] = np.arange(len(src_df), dtype=int)
        true_col_src = args.target  # column in source CSV
        if true_col_src in src_df.columns:
            true_map = dict(zip(src_df["sample_idx"], src_df[true_col_src]))
            true_col = f"{args.target}_true"
            n_masked = 0
            for i, sidx in enumerate(df_full["sample_idx"].astype(int)):
                src_val = true_map.get(int(sidx))
                if src_val is None or pd.isna(src_val):
                    if pd.notna(df_full.iloc[i][true_col]):
                        df_full.at[df_full.index[i], true_col] = np.nan
                        n_masked += 1
            if n_masked > 0:
                print(f"[info] Masked {n_masked} OOF true labels per source CSV "
                      f"(post-hoc data fixes propagated)")

    slope, intercept = fit_sputter_platt(df_full, target=args.target)
    pred_col = f"{args.target}_pred"
    pred_platt = f"{args.target}_pred_platt"
    df_full[pred_platt] = df_full[pred_col] * slope + intercept

    df_sp = df_full[df_full.is_sputter].copy() if args.scope == "sputter" else df_full.copy()
    df_eval = df_sp.dropna(subset=[f"{args.target}_true"]).copy()

    # ── Run all metric blocks ──
    print(f"\n{'='*80}\n SPUTTER PHYSICS EVAL — {bundle.name}  (target={args.target}, scope={args.scope})\n{'='*80}")

    # 1. R² decomposition
    decomp = r2_variance_decomposition(df_eval, args.target)
    print(f"\n[Overall metrics, n_labeled={decomp.get('n', 0)}]")
    print(f"  Overall R²        = {decomp['overall_r2']:+.3f}")
    print(f"  Overall r         = {decomp['overall_r']:+.3f}")
    print(f"  Overall MAE       = {decomp['overall_mae']:.3f}")
    print(f"  Between-element R² = {decomp['between_r2']:+.3f}    "
          f"← cross-element ordering ({decomp.get('n_elements', 0)} elements)")
    print(f"  Within-element R²  = {decomp['within_r2']:+.3f}    "
          f"← within-element concentration response (THE physics test)")

    # 2. Within-DOI slope test (THE primary physics metric)
    wd_df, wd_summary = within_doi_slope_test(df_sp, args.target)
    print(f"\n[Within-DOI per-element slope (n_groups={wd_summary['n_groups']}, "
          f"OK={wd_summary['ok']}, FLIP={wd_summary['flip']})]")
    if len(wd_df) > 0:
        print(wd_df[["doi", "element", "n", "rho_true", "rho_pred",
                     "rho_lit", "sign_match"]].to_string(index=False))
    else:
        print("  (no DOIs with ≥3 sputter rows of same element + 2+ concs)")

    # 3. Robust per-element parity
    rp_df = per_element_robust_parity(df_eval, args.target)
    print(f"\n[Per-element parity (raw vs IQR-filtered)]")
    if len(rp_df) > 0:
        print(rp_df[["element", "n", "r_raw", "n_clean", "r_clean",
                     "n_outliers_excluded"]].to_string(index=False))

    # 4a. MC-dropout std calibration (compute scale factor for Tier-1C)
    cal = mc_dropout_calibration(df_eval, args.target)
    if cal is not None:
        print(f"\n[MC-dropout std calibration, n={cal['n']}]")
        print(f"  Coverage at 1σ:  {cal['coverage_at_1sigma']*100:.1f}% "
              f"(theoretical Gaussian: 68.0%)")
        print(f"  Coverage at 2σ:  {cal['coverage_at_2sigma']*100:.1f}% "
              f"(theoretical Gaussian: 95.0%)")
        if cal["is_overconfident"]:
            print(f"  → std under-estimates uncertainty (coverage at 1σ < 63%)")
        elif cal["is_underconfident"]:
            print(f"  → std OVER-estimates uncertainty (coverage at 1σ > 73%) "
                  f"— Tier-1C 'ambiguous' may be overly lenient")
        else:
            print(f"  → std is well-calibrated (coverage at 1σ within ±5% of 68%)")
        print(f"  Scale factor for 68% coverage: {cal['scale_for_68pct']:.3f} "
              f"(values <1.0 mean current std is too large)")
        cal_scale = cal["scale_for_68pct"]
    else:
        cal_scale = 1.0
        print(f"\n[MC-dropout std calibration: no std column or insufficient data]")

    # 4b. Bootstrap CI on within-DOI ρ (statistical significance)
    boot_df = bootstrap_within_doi_ci(df_sp, args.target,
                                       n_boot=1000, seed=42)
    print(f"\n[Bootstrap 95% CI on within-DOI ρ_pred (n_groups={len(boot_df)})]")
    if len(boot_df) > 0:
        # Sign-confidence: how strongly each group agrees with its lit sign
        for _, row in boot_df.iterrows():
            lit_sign = np.sign(row.rho_lit) if np.isfinite(row.rho_lit) else 0
            ci_lo, ci_hi = row.rho_lo, row.rho_hi
            if lit_sign > 0:
                # CI must be entirely > 0 to confidently match lit
                if ci_lo > 0:
                    verdict = "OK (CI > 0)"
                elif ci_hi < 0:
                    verdict = "FLIP (CI < 0)"
                else:
                    verdict = "ambiguous (CI crosses 0)"
            elif lit_sign < 0:
                if ci_hi < 0:
                    verdict = "OK (CI < 0)"
                elif ci_lo > 0:
                    verdict = "FLIP (CI > 0)"
                else:
                    verdict = "ambiguous (CI crosses 0)"
            else:
                verdict = "—"
            boot_df.loc[_, "verdict"] = verdict
        print(boot_df[["doi", "element", "n", "rho_point", "rho_lo", "rho_hi",
                       "sign_stability", "rho_lit", "verdict"]].to_string(index=False))
    else:
        print("  (no eligible groups)")

    # 5. Tier-1C with calibrated intervals
    t1c_df = tier1c_interval_check(df_sp, args.target, std_scale=cal_scale)
    flip_count = int(t1c_df["verdict"].str.contains("FLIP", regex=False).sum()) if len(t1c_df) else 0
    amb_count = int((t1c_df["verdict"] == "ambiguous").sum()) if len(t1c_df) else 0
    ok_count = int((t1c_df["verdict"] == "OK").sum()) if len(t1c_df) else 0
    print(f"\n[Tier-1C interval (baseline={t1c_df.attrs.get('baseline', 0):+.3f}, "
          f"std_scale={cal_scale:.2f} {'[CALIBRATED]' if cal else '[NO CAL]'})]")
    print(f"  OK: {ok_count}, ambiguous: {amb_count}, FLIP: {flip_count}")
    if len(t1c_df) > 0:
        print(t1c_df[["element", "valence", "class_", "n_total", "n_labeled",
                      "pred_mean", "pred_std", "d_vs_baseline",
                      "expected", "verdict"]].to_string(index=False))

    # ── Save markdown ──
    if not args.no_write:
        out_path = bundle / f"sputter_physics_eval_{args.target}_{args.scope}.md"
        md = []
        md.append(f"# Sputter Physics Evaluation — {bundle.name}\n")
        md.append(f"`target={args.target}, scope={args.scope}, "
                  f"Platt slope={slope:+.3f}, intercept={intercept:+.3f}`\n\n")
        md.append("## 1. Overall + R² Decomposition\n\n")
        md.append(f"| Metric | Value | Interpretation |\n|---|---:|---|\n")
        md.append(f"| n_labeled | {decomp.get('n', 0)} | sputter labeled rows |\n")
        md.append(f"| Overall R² | {decomp['overall_r2']:+.3f} | aggregate fit on Platt-calibrated preds |\n")
        md.append(f"| Overall r | {decomp['overall_r']:+.3f} | Pearson on full sputter |\n")
        md.append(f"| Overall MAE | {decomp['overall_mae']:.3f} | log-units |\n")
        md.append(f"| **Between-element R²** | {decomp['between_r2']:+.3f} | cross-element ordering (acceptor vs donor) |\n")
        md.append(f"| **Within-element R²** | {decomp['within_r2']:+.3f} | concentration response within each element — *primary physics test* |\n\n")
        md.append("## 2. Within-DOI per-element slope test (cleanest physics test)\n\n")
        md.append(f"Same paper, same atmosphere/temperature, only concentration varies. "
                  f"Spearman ρ of predicted log V_O vs log(concentration) should match "
                  f"literature sign (LIT_RHO).\n\n")
        md.append(f"**Summary**: {wd_summary['n_groups']} eligible groups, "
                  f"OK={wd_summary['ok']}, FLIP={wd_summary['flip']}\n\n")
        md.append(_df_to_md(wd_df))
        md.append("\n## 3. Per-element parity (raw vs IQR-1.5 filtered)\n\n")
        md.append("Single-point outliers can flip per-element correlation. "
                  "Reports both raw and outlier-filtered Pearson r.\n\n")
        md.append(_df_to_md(rp_df))
        md.append("\n## 4. Tier-1C with ensemble-std interval\n\n")
        md.append(f"Baseline = {t1c_df.attrs.get('baseline_label', 'unknown')} "
                  f"= {t1c_df.attrs.get('baseline', 0):+.3f}\n\n")
        md.append(f"**Summary**: OK={ok_count}, ambiguous={amb_count}, FLIP={flip_count}\n\n")
        md.append(_df_to_md(t1c_df))
        out_path.write_text("\n".join(md))
        print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
