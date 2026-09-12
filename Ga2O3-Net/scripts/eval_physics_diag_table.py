"""Tabular physics-learning diagnostics for a Phase X bundle.

Produces a Markdown file `results/<bundle>/physics_diag.md` with per-element
tables for the Tier-1/Tier-2 acceptance rubric.

Supports both VC (vacancy_concentration) and PDR (photo_dark_ratio) targets.

Direction conventions:
- VC (V_O concentration):
    * Donors (Sn/Si/Ti/Ge, v=4 + super-donors v=5) → push V_O UP → "above" undoped
    * Acceptors (Mg/Zn/Cu, v=2) → push V_O DOWN → "below" undoped
- PDR (photo/dark current ratio):
    * V_O are deep traps that suppress PDR via SRH recombination,
      so MORE V_O → LOWER PDR. Therefore the direction is FLIPPED:
    * Donors → more V_O traps → LOWER PDR → "below" undoped
    * Acceptors → fewer V_O → trap reduction + dark-current suppression → HIGHER PDR → "above" undoped

Usage:
    python scripts/eval_physics_diag_table.py <bundle_dir> [<full_csv>] \
        [--target {vacancy_concentration|photo_dark_ratio}] [--scope {all|sputter}]
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

PROJ = Path(__file__).resolve().parents[1]

ELEM_VALENCE = {
    "Mg": 2, "Zn": 2, "Cu": 2,                # acceptors
    "Al": 3, "Fe": 3, "B": 3, "V": 3,         # isovalent / mixed-valence
    "Er": 3, "Eu": 3,
    "Si": 4, "Sn": 4, "Ti": 4, "Ge": 4,       # donors (v=4)
    "Sb": 5, "Ta": 5, "Bi": 5,                # super-donors (v=5)
    "W":  6,                                  # super-donor (v=6)
    "F": -1,                                  # anion
}
ACCEPTOR_ELEMENTS = {"Mg", "Zn", "Cu"}
DONOR_ELEMENTS = {"Sn", "Si", "Ti", "Ge"}
# V52 (2026-05-06): Sb/Bi moved from super_donor to isovalent.
# Bi³⁺ (6s² lone pair) and Sb³⁺ (5s² lone pair) prefer isovalent +3
# substitution on Ga³⁺ site; their effect on V_O is dominated by lone-pair
# VBM upshift, not donor-style E_F pinning. Reference: J. Phys. Chem. C
# 2025, 10.1021/acs.jpcc.5c02687.
SUPER_DONOR_ELEMENTS = {"Ta", "W"}     # ← V52: was {"Sb", "Ta", "Bi", "W"}
ISOVALENT_ELEMENTS = {"Al", "Fe", "B", "V", "Er", "Eu", "Sb", "Bi"}  # ← V52: added Sb/Bi

# Literature anchor slopes for ρ(log V_O vs log conc):
# V52: Sb/Bi removed. As isovalent dopants with active lone pairs, they
# do NOT have a literature-canonical donor (+) or acceptor (-) sign for
# their effect on V_O. Setting expected sign 0 (≈) is more honest than
# forcing them into donor monotonicity.
LIT_RHO = {"Mg": -0.95, "Zn": -0.90, "Cu": -0.85,
           "Sn": +0.95, "Si": +0.95, "Ti": +0.80, "Ge": +0.85,
           "Ta": +0.95}    # ← V52: removed Sb (+0.95), Bi (+0.85)


def _expected_class(elem: str) -> str:
    """Return 'acceptor' / 'donor' / 'super_donor' / 'isovalent' / 'anion' /
    'unknown' for a dopant element."""
    if elem in ACCEPTOR_ELEMENTS:
        return "acceptor"
    if elem in DONOR_ELEMENTS:
        return "donor"
    if elem in SUPER_DONOR_ELEMENTS:
        return "super_donor"
    if elem in ISOVALENT_ELEMENTS:
        return "isovalent"
    if elem == "F":
        return "anion"
    if elem == "undoped":
        return "undoped"
    return "unknown"


def _expected_side(elem: str, target: str = "vacancy_concentration") -> str:
    """Expected side of mean-pred relative to undoped baseline given valence
    class AND target. 'below', 'above', '≈', or '?'.

    For VC: donors → above, acceptors → below (direct V_O scaling).
    For PDR: donors → below, acceptors → above (V_O traps suppress PDR
    via SRH recombination — direction is FLIPPED relative to VC).
    """
    cls = _expected_class(elem)
    flip_pdr = (target == "photo_dark_ratio")
    if cls == "acceptor":
        return "above" if flip_pdr else "below"
    if cls in ("donor", "super_donor"):
        return "below" if flip_pdr else "above"
    if cls in ("isovalent", "anion"):
        return "≈"
    return "?"


def _target_lit_rho(target: str) -> dict[str, float]:
    """Literature anchor slopes for ρ(log target vs log conc) per element.

    For VC: standard donor/acceptor sign.
    For PDR: flipped sign (V_O traps suppress photo-response → donor → ρ_PDR < 0).
    """
    # V52 (2026-05-06): Sb/Bi removed from both VC and PDR rho tables.
    # As isovalent lone-pair dopants they have no canonical donor/acceptor
    # sign; treating them as ≈ in eval is more honest.
    if target == "photo_dark_ratio":
        return {"Mg": +0.85, "Zn": +0.80, "Cu": +0.75,
                "Sn": -0.85, "Si": -0.80, "Ti": -0.70, "Ge": -0.75,
                "Ta": -0.85}    # ← V52: removed Sb, Bi
    # VC default
    return {"Mg": -0.95, "Zn": -0.90, "Cu": -0.85,
            "Sn": +0.95, "Si": +0.95, "Ti": +0.80, "Ge": +0.85,
            "Ta": +0.95}    # ← V52: removed Sb, Bi


def _spearman_safe(x, y):
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def _atm_group(atm) -> str:
    if pd.isna(atm):
        return "(missing)"
    s = str(atm).lower()
    if "plasma" in s or "o2_plasma" in s or "o₂_plasma" in s:
        return "O2_plasma"
    if "ar:o" in s or "ar/o" in s or "ar_o2" in s or "ar+o" in s:
        return "Ar+O2_mix"
    if s.startswith("o2") or s == "oxygen":
        return "O2"
    if s.startswith("ar") or s == "argon":
        return "Ar"
    if "air" in s:
        return "air"
    if "n2o" in s or "n₂o" in s:
        return "N2O"
    if s.startswith("n2") or "nitrogen" in s:
        return "N2"
    if "vac" in s:
        return "vacuum"
    return s[:12]


def _resolve_csv_from_bundle(bundle_dir: Path) -> Path | None:
    """If bundle has fusion_head128.yaml, read its experimental_csv path."""
    fy = bundle_dir / "fusion_head128.yaml"
    if not fy.exists():
        return None
    try:
        import yaml as _yaml
        cfg = _yaml.safe_load(fy.read_text()) or {}
        rel = (cfg.get("paths") or {}).get("experimental_csv")
        if rel is None:
            return None
        return PROJ / rel
    except Exception:
        return None


def load_oof_with_meta(bundle_dir: Path, full_csv: str | None,
                       target: str = "vacancy_concentration") -> pd.DataFrame:
    pred_col = f"{target}_pred"
    oof = pd.read_csv(bundle_dir / "oof_predictions.csv")
    if pred_col not in oof.columns:
        raise ValueError(f"OOF file lacks {pred_col!r}; columns are {list(oof.columns)}")

    if full_csv is None:
        resolved = _resolve_csv_from_bundle(bundle_dir)
        if resolved is not None and resolved.exists():
            full_csv = str(resolved)

    if full_csv is None:
        sub = oof.dropna(subset=[pred_col]).copy()
        sub["is_sputter"] = True
        if "atmosphere" not in sub.columns:
            sub["atmosphere"] = ""
        if "concentration_at%" not in sub.columns:
            sub["concentration_at%"] = np.nan
        return sub

    src = pd.read_csv(full_csv)
    if "usable_flag" in src.columns:
        src = src[src["usable_flag"] != "exotic_skip"].reset_index(drop=True)
    src = src[~src["element"].isin(["N", "H", "Nb", "In"])].reset_index(drop=True)
    src["sample_idx"] = np.arange(len(src), dtype=int)

    keep = [c for c in
            ["sample_idx", "method", "atmosphere", "concentration_at%",
             "doi", "element"]
            if c in src.columns]
    merged = oof.merge(src[keep], on="sample_idx", how="left",
                       suffixes=("", "_src"))
    if "element" in merged.columns and "element_src" in merged.columns:
        merged["element"] = merged["element"].fillna(merged["element_src"])
    if "atmosphere" not in merged.columns and "atmosphere_src" in merged.columns:
        merged = merged.rename(columns={"atmosphere_src": "atmosphere"})
    merged["is_sputter"] = (
        merged["method"].fillna("").str.lower().str.contains("sputter")
    )
    full = merged.dropna(subset=[pred_col]).copy()
    return full


def fit_sputter_platt(df: pd.DataFrame, target: str = "vacancy_concentration") -> tuple[float, float]:
    pred_col = f"{target}_pred"
    true_col = f"{target}_true"
    sub = df[df.is_sputter].dropna(subset=[pred_col, true_col])
    if len(sub) < 2:
        return 1.0, 0.0
    y_raw  = sub[pred_col].to_numpy()
    y_true = sub[true_col].to_numpy()
    slope, intercept = np.polyfit(y_raw, y_true, 1)
    return float(slope), float(intercept)


def tier1_acceptor_donor(df_sp: pd.DataFrame, target: str = "vacancy_concentration") -> pd.DataFrame:
    """Per-element ρ(log target vs log conc), measured vs predicted (Platt)."""
    pred_col = f"{target}_pred_platt"
    true_col = f"{target}_true"
    rows = []
    work = df_sp.dropna(subset=["dopant_label", "concentration_at%", true_col]).copy()
    work["c"] = pd.to_numeric(work["concentration_at%"], errors="coerce")
    work = work[work.c > 0].copy()
    work["logc"] = np.log10(work.c)

    lit_rho = _target_lit_rho(target)
    flip_pdr = (target == "photo_dark_ratio")

    for elem in sorted(work.dopant_label.unique(),
                       key=lambda e: -len(work[work.dopant_label == e])):
        sub = work[work.dopant_label == elem]
        if len(sub) < 3:
            continue
        rho_true = _spearman_safe(sub.logc.to_numpy(),
                                  sub[true_col].to_numpy())
        rho_pred = _spearman_safe(sub.logc.to_numpy(),
                                  sub[pred_col].to_numpy())
        v = ELEM_VALENCE.get(elem, 99)
        # Expected sign by valence × target. For PDR sign is FLIPPED:
        #   acceptor → ρ > 0 (more conc → fewer V_O → higher PDR)
        #   donor    → ρ < 0
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
        lit = lit_rho.get(elem, np.nan)
        rows.append(dict(
            element=elem, valence=v, n=len(sub),
            rho_true=rho_true, rho_pred=rho_pred,
            rho_lit=lit, sign_match=sign_ok,
        ))
    return pd.DataFrame(rows).sort_values(["valence", "element"]).reset_index(drop=True)


def tier1_element_baseline(df_sp: pd.DataFrame, target: str = "vacancy_concentration") -> pd.DataFrame:
    pred_col = f"{target}_pred_platt"
    true_col = f"{target}_true"
    rows = []
    work = df_sp.dropna(subset=[true_col]).copy()
    for elem in sorted(work.dopant_label.unique(),
                       key=lambda e: -len(work[work.dopant_label == e])):
        sub = work[work.dopant_label == elem]
        if len(sub) < 2:
            continue
        rows.append(dict(
            element=elem,
            valence=ELEM_VALENCE.get(elem, 99),
            n=len(sub),
            mean_true=float(sub[true_col].mean()),
            mean_pred=float(sub[pred_col].mean()),
            mean_bias=float((sub[pred_col] - sub[true_col]).mean()),
        ))
    return pd.DataFrame(rows).sort_values(["valence", "element"]).reset_index(drop=True)


def tier1_atmo_x_element(df_sp: pd.DataFrame, target: str = "vacancy_concentration") -> pd.DataFrame:
    pred_col = f"{target}_pred_platt"
    true_col = f"{target}_true"
    work = df_sp.dropna(subset=["atmosphere", "dopant_label", true_col]).copy()
    work["atm_g"] = work.atmosphere.apply(_atm_group)
    cells = []
    for atm in sorted(work.atm_g.unique()):
        for elem in sorted(work.dopant_label.unique(),
                           key=lambda e: -len(work[work.dopant_label == e])):
            cell = work[(work.atm_g == atm) & (work.dopant_label == elem)]
            if len(cell) < 1:
                continue
            cells.append(dict(
                atmosphere=atm, element=elem, n=len(cell),
                bias=float((cell[pred_col] - cell[true_col]).mean()),
            ))
    return pd.DataFrame(cells)


def tier2_within_element_slope(df_sp: pd.DataFrame, target: str = "vacancy_concentration") -> pd.DataFrame:
    """Per-element pred-vs-truth fit (Pearson r + Spearman ρ + MAE)."""
    pred_col = f"{target}_pred_platt"
    true_col = f"{target}_true"
    work = df_sp.dropna(subset=[true_col]).copy()
    rows = []
    for elem in sorted(work.dopant_label.unique(),
                       key=lambda e: -len(work[work.dopant_label == e])):
        sub = work[work.dopant_label == elem]
        if len(sub) < 3:
            continue
        y_t = sub[true_col].to_numpy()
        y_p = sub[pred_col].to_numpy()
        if np.std(y_t) < 1e-9:
            continue
        r = float(stats.pearsonr(y_t, y_p).statistic)
        rho = _spearman_safe(y_t, y_p)
        rows.append(dict(
            element=elem, valence=ELEM_VALENCE.get(elem, 99),
            n=len(sub),
            pearson_r=r, spearman_rho=rho,
            mae=float(np.mean(np.abs(y_p - y_t))),
        ))
    return pd.DataFrame(rows).sort_values(["valence", "element"]).reset_index(drop=True)


def tier1b_rare_element_direction(df_eval: pd.DataFrame, target: str = "vacancy_concentration") -> pd.DataFrame:
    """Tier-1B (NEW 2026-05-02 user requirement): for EVERY element with
    valid labels (regardless of N), check if the model puts that element
    on the correct SIDE of the population mean given its valence class
    AND the target's direction convention (V_O direct, PDR flipped).
    """
    pred_col = f"{target}_pred_platt"
    true_col = f"{target}_true"
    valid = df_eval.dropna(subset=[pred_col, true_col]).copy()
    if len(valid) < 1:
        return pd.DataFrame()
    mean_pred_global = float(valid[pred_col].mean())
    mean_true_global = float(valid[true_col].mean())

    rows = []
    for elem in sorted(valid.dopant_label.unique(),
                       key=lambda e: -len(valid[valid.dopant_label == e])):
        sub = valid[valid.dopant_label == elem]
        if len(sub) < 1:
            continue
        v = ELEM_VALENCE.get(elem, 99)
        cls = _expected_class(elem)
        expected = _expected_side(elem, target)

        mean_pred = float(sub[pred_col].mean())
        mean_true = float(sub[true_col].mean())
        side_pred = mean_pred - mean_pred_global
        side_true = mean_true - mean_true_global

        # Did pred direction match TRUE direction?
        if abs(side_true) < 0.20:
            true_dir_match = "≈"
        elif np.sign(side_pred) == np.sign(side_true):
            true_dir_match = "OK"
        else:
            true_dir_match = "**FLIP**"

        # Did pred direction match VALENCE expectation?
        if expected == "below":
            val_match = "OK" if side_pred < 0 else "**FLIP**"
        elif expected == "above":
            val_match = "OK" if side_pred > 0 else "**FLIP**"
        elif expected == "≈":
            val_match = "OK" if abs(side_pred) < 0.5 else "**OFFSET**"
        else:
            val_match = "—"

        rows.append(dict(
            element=elem, valence=v, class_=cls, n=len(sub),
            mean_true=mean_true, mean_pred=mean_pred,
            d_true=side_true, d_pred=side_pred,
            true_dir=true_dir_match,
            valence_dir=val_match,
        ))
    return pd.DataFrame(rows).sort_values(["valence", "element"]).reset_index(drop=True)


def tier1c_unsupervised_element_direction(df_full: pd.DataFrame, target: str = "vacancy_concentration") -> pd.DataFrame:
    """Tier-1C (NEW 2026-05-02 user requirement): for EVERY element that has
    predictions — even those with NO valid labels — check direction
    relative to the **undoped Ga₂O₃ baseline**, given valence class.

    Expected direction per class is now driven by EMPIRICAL labeled data:
    for each valence class (acceptor / donor / super_donor / isovalent),
    compute the mean true target value across labeled elements in that
    class and compare to undoped true mean. If the empirical sign matches
    a valence-theory prior, we use that; if data disagrees, data wins.

    This avoids enforcing textbook-only direction priors that may not hold
    in this specific dataset (e.g., PDR's donor-suppression-via-V_O-trap
    rule does NOT empirically hold for Sn-Ga2O3 photodetectors).
    """
    pred_col = f"{target}_pred_platt"
    true_col = f"{target}_true"
    work = df_full.dropna(subset=[pred_col]).copy()
    if len(work) < 1:
        return pd.DataFrame()
    # Use labeled-overall mean (pred) as reference. This is robust when undoped
    # rows are unlabeled and aligns with the labeled-overall true mean used
    # for empirical class direction below.
    labeled = work.dropna(subset=[true_col])
    if len(labeled) >= 2:
        baseline_pred = float(labeled[pred_col].mean())
        baseline_label = f"labeled-overall mean (N={len(labeled)})"
    else:
        # Fallback: undoped pred mean (legacy)
        undoped = work[work.dopant_label == "undoped"]
        if len(undoped) >= 1:
            baseline_pred = float(undoped[pred_col].mean())
            baseline_label = f"undoped (N={len(undoped)})"
        else:
            baseline_pred = float(work[pred_col].median())
            baseline_label = "global median"

    # Empirical class-direction map (from labeled rows only).
    # Guard: require ≥ 5 labeled rows AND ≥ 2 distinct elements per class
    # before trusting empirical direction. With less data, fall back to
    # valence-theory for that class (which protects against single-element
    # classes like "Fe is the only isovalent" dictating direction for B/V/Eu).
    class_emp_dir: dict[str, int] = {}
    if len(labeled) >= 5:
        ref_true = float(labeled[true_col].mean())
        for cls in ("acceptor", "donor", "super_donor", "isovalent"):
            elems = [e for e in labeled.dopant_label.unique()
                     if _expected_class(e) == cls]
            if len(elems) < 2:
                continue   # not enough distinct elements to generalize
            sub = labeled[labeled.dopant_label.isin(elems)]
            if len(sub) < 5:
                continue   # not enough rows to trust class direction
            mean_true = float(sub[true_col].mean())
            d = mean_true - ref_true
            if abs(d) < 0.20:
                class_emp_dir[cls] = 0
            elif d > 0:
                class_emp_dir[cls] = +1
            else:
                class_emp_dir[cls] = -1

    rows = []
    for elem in sorted(work.dopant_label.unique(),
                       key=lambda e: -len(work[work.dopant_label == e])):
        sub = work[work.dopant_label == elem]
        if len(sub) < 1:
            continue
        if elem == "undoped":
            continue
        v = ELEM_VALENCE.get(elem, 99)
        cls = _expected_class(elem)
        n_total = len(sub)
        n_labeled = sub[true_col].notna().sum()

        # Expected side: prefer empirical class direction; fall back to
        # valence-theory prior when class has no labeled exemplars.
        emp = class_emp_dir.get(cls)
        if emp is not None:
            if emp > 0:
                expected = "above"
            elif emp < 0:
                expected = "below"
            else:
                expected = "≈"
            expected_src = "empirical"
        else:
            expected = _expected_side(elem, target)
            expected_src = "valence_theory"

        mean_pred = float(sub[pred_col].mean())
        side_pred = mean_pred - baseline_pred

        if expected == "below":
            val_match = "OK" if side_pred < 0 else "**FLIP**"
        elif expected == "above":
            val_match = "OK" if side_pred > 0 else "**FLIP**"
        elif expected == "≈":
            val_match = "OK" if abs(side_pred) < 0.5 else "**OFFSET**"
        else:
            val_match = "—"

        rows.append(dict(
            element=elem, valence=v, class_=cls,
            n_total=int(n_total), n_labeled=int(n_labeled),
            mean_pred=mean_pred, d_vs_undoped=side_pred,
            expected=expected, expected_src=expected_src,
            verdict=val_match,
        ))
    df_out = pd.DataFrame(rows).sort_values(["valence", "element"]).reset_index(drop=True)
    df_out.attrs["baseline_label"] = baseline_label
    df_out.attrs["baseline_value"] = baseline_pred
    df_out.attrs["class_emp_dir"]  = class_emp_dir
    return df_out


def tier2_cross_element_ordering(df_sp: pd.DataFrame, target: str = "vacancy_concentration") -> pd.DataFrame:
    """Cross-element rank: predicted vs true means ordered."""
    pred_col = f"{target}_pred_platt"
    true_col = f"{target}_true"
    work = df_sp.dropna(subset=["concentration_at%", true_col]).copy()
    work["c"] = pd.to_numeric(work["concentration_at%"], errors="coerce")
    work = work.dropna(subset=["c"])
    work = work[work.c > 0]
    rows = []
    for elem in sorted(work.dopant_label.unique(),
                       key=lambda e: -len(work[work.dopant_label == e])):
        sub = work[work.dopant_label == elem]
        if len(sub) < 2:
            continue
        rows.append(dict(
            element=elem,
            valence=ELEM_VALENCE.get(elem, 99),
            n=len(sub),
            mean_true=float(sub[true_col].mean()),
            mean_pred=float(sub[pred_col].mean()),
        ))
    df = pd.DataFrame(rows).sort_values("mean_pred").reset_index(drop=True)
    df["pred_rank"] = df.index + 1
    df = df.sort_values("mean_true").reset_index(drop=True)
    df["true_rank"] = df.index + 1
    df = df.sort_values(["valence", "element"]).reset_index(drop=True)
    return df


def df_to_md(df: pd.DataFrame, fmt: dict | None = None) -> str:
    if df.empty:
        return "_(no data)_\n"
    fmt = fmt or {}
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in fmt:
                cells.append(fmt[c](v))
            elif isinstance(v, float):
                if np.isnan(v):
                    cells.append("(N/A)")
                else:
                    cells.append(f"{v:+.3f}" if abs(v) < 100 else f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(bundle_dir: Path, full_csv: str | None, label: str | None,
         scope: str = "all", target: str = "vacancy_concentration"):
    """Run physics-learning diagnostic tables.

    target = 'vacancy_concentration' (V_O log) or 'photo_dark_ratio' (PDR log).
             Direction conventions flip per target.
    scope  = 'all' (every element with N≥3 in OOF) or 'sputter' (legacy).
    """
    label = label or bundle_dir.name
    pred_col = f"{target}_pred"
    true_col = f"{target}_true"
    pred_platt_col = f"{target}_pred_platt"

    df_full = load_oof_with_meta(bundle_dir, full_csv, target=target)

    slope, intercept = fit_sputter_platt(df_full, target=target)
    df_full[pred_platt_col] = df_full[pred_col] * slope + intercept

    if scope == "sputter":
        df_full = df_full[df_full.is_sputter].copy()

    df_eval = df_full.dropna(subset=[true_col]).copy()

    y_t = df_eval[true_col].to_numpy()
    y_p = df_eval[pred_platt_col].to_numpy()
    if len(y_t) >= 2:
        r = float(stats.pearsonr(y_t, y_p).statistic)
        ss_res = float(np.sum((y_t - y_p) ** 2))
        ss_tot = float(np.sum((y_t - y_t.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mae = float(np.mean(np.abs(y_t - y_p)))
    else:
        r = r2 = mae = float("nan")

    target_short = "VC" if target == "vacancy_concentration" else (
        "PDR" if target == "photo_dark_ratio" else target)
    out_path = bundle_dir / f"physics_diag_{target_short.lower()}_{scope}.md"
    md = []
    md.append(f"# Physics-learning diagnostics — {label} ({target_short}, scope={scope})\n")
    md.append(f"OOF rows total = {len(df_full)}  ·  sputter subset = {df_full.is_sputter.sum()}  ·  "
              f"labeled (with {target_short} true) = {len(df_eval)}\n")
    md.append(f"Platt fit (on sputter labeled rows): slope = {slope:+.4f}, intercept = {intercept:+.4f}\n")
    md.append(f"Overall (Platt, labeled, scope={scope}): R² = {r2:+.3f}, r = {r:+.3f}, MAE = {mae:.3f}\n\n")

    if target == "vacancy_concentration":
        sign_note = "Acceptors (v=2) expected ρ < 0; donors (v=4) expected ρ > 0."
        baseline_note = ("Acceptor → predicted below undoped V_O baseline; "
                         "donor / super-donor → above; isovalent → ≈.")
    else:  # PDR
        sign_note = ("Acceptors (v=2) expected ρ > 0 (compensation reduces dark current → higher PDR); "
                     "donors (v=4+) expected ρ < 0 (more V_O traps suppress photo-response).")
        baseline_note = ("Acceptor → predicted above undoped PDR baseline; "
                         "donor / super-donor → below (V_O traps suppress PDR); "
                         "isovalent → ≈.")

    md.append("## Tier-1A — Per-element slope ρ (N≥3 only)\n")
    md.append(f"Spearman ρ between log₁₀({target_short}) and log₁₀(concentration). "
              f"{sign_note}\n\n")
    t1 = tier1_acceptor_donor(df_eval, target=target)
    md.append(df_to_md(t1))

    md.append("\n## Tier-1B — Direction check on EVERY labeled element\n")
    md.append(f"For every element with at least one valid {target_short} label "
              f"(rare elements included), check if the model's mean prediction is "
              f"on the correct SIDE of the LABELED population mean given valence "
              f"class. {baseline_note}\n\n")
    t1b = tier1b_rare_element_direction(df_eval, target=target)
    md.append(df_to_md(t1b))

    md.append("\n## Tier-1C — Direction on UNSUPERVISED elements\n")
    t1c = tier1c_unsupervised_element_direction(df_full, target=target)
    md.append(f"Reference baseline = mean predicted {target_short} on "
              f"**{t1c.attrs.get('baseline_label', 'unknown')}** "
              f"= {t1c.attrs.get('baseline_value', float('nan')):+.3f}.\n")
    md.append(f"For every element that has predictions but NO valid {target_short} "
              f"label, check if its predicted mean is on the correct SIDE of the "
              f"undoped baseline. User's PRIMARY acceptance criterion: predict "
              f"CORRECT DIRECTION for unseen elements. {baseline_note}\n\n")
    md.append(df_to_md(t1c))

    md.append("\n## Tier-1 #2 — Element baseline (predicted vs measured mean)\n")
    md.append("Mean prediction must NOT be pulled toward cross-method baselines. "
              "|bias| < 0.5 healthy. Labeled elements only.\n\n")
    t2 = tier1_element_baseline(df_eval, target=target)
    md.append(df_to_md(t2))

    md.append("\n## Tier-1 #5 — Atmosphere × element bias\n")
    md.append("Mean signed residual (predicted − measured). |bias| > 0.5 = warning, "
              "> 1.0 = atmosphere ordinal violated for that element.\n\n")
    t3 = tier1_atmo_x_element(df_eval, target=target)
    md.append(df_to_md(t3))

    md.append("\n## Tier-2 #6 — Within-element panorama\n")
    md.append("Per-element pred-vs-truth fit (Pearson r + Spearman ρ + MAE).\n\n")
    t4 = tier2_within_element_slope(df_eval, target=target)
    md.append(df_to_md(t4))

    md.append("\n## Tier-2 #7 — Cross-element ordering\n")
    if target == "vacancy_concentration":
        rank_note = "Donor elements (v=4) should have LARGER predicted V_O than acceptor (v=2)."
    else:
        rank_note = ("Acceptor elements (v=2) should have LARGER predicted PDR than "
                     "donor (v=4+) — V_O traps suppress photo-response.")
    md.append(f"Per-element predicted mean vs true mean. {rank_note} "
              "Includes every labeled element with N≥2.\n\n")
    t5 = tier2_cross_element_ordering(df_eval, target=target)
    md.append(df_to_md(t5))

    out_path.write_text("\n".join(md))
    print(f"[OK] wrote {out_path}")
    print()
    print(f"Tier-1A (slope, N≥3, scope={scope}):")
    print(t1.to_string(index=False))
    print()
    print(f"Tier-1B (labeled-element direction):")
    print(t1b.to_string(index=False))
    print()
    print(f"Tier-1C (UNSUPERVISED element direction):")
    print(t1c.to_string(index=False))
    return dict(t1=t1, t1b=t1b, t1c=t1c, t2=t2, t3=t3, t4=t4, t5=t5,
                r2=r2, r=r, mae=mae,
                slope=slope, intercept=intercept,
                n_total=len(df_full), n_labeled=len(df_eval))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle_dir")
    p.add_argument("full_csv", nargs="?", default=None)
    p.add_argument("--scope", default="all", choices=["all", "sputter"],
                   help="Evaluation scope. 'all' (default — every element regardless of N) "
                        "or 'sputter' (sputter-only).")
    p.add_argument("--target", default="vacancy_concentration",
                   choices=["vacancy_concentration", "photo_dark_ratio", "both"],
                   help="Which target to evaluate. 'both' runs VC then PDR.")
    args = p.parse_args()

    bundle = Path(args.bundle_dir)
    if not bundle.is_absolute():
        bundle = PROJ / bundle

    targets = ["vacancy_concentration", "photo_dark_ratio"] if args.target == "both" else [args.target]
    for t in targets:
        try:
            main(bundle, args.full_csv, label=bundle.name, scope=args.scope, target=t)
        except ValueError as e:
            print(f"[SKIP] {t}: {e}")
            continue
