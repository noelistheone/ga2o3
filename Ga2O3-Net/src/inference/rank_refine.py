"""Post-hoc RankRefine — Bradley-Terry pairwise refinement of regression OOF.

Reference: Wijaya et al., "Post Hoc Regression Refinement via Pairwise
Rankings", NeurIPS 2025 (arXiv:2508.16495).

Idea: for each OOF sample, build pairwise sign comparisons against labeled
neighbours using physics-derived rules (Tier-1A: within-element acceptor /
donor slope; cross-class hierarchy when within-element refs are scarce).
Solve 1-D Bradley-Terry to get ŷ^rank in true-target units. Fuse with the
regression estimate via inverse-variance weighting:

    ŷ_fused = (ŷ_reg/σ_reg² + ŷ_rank/σ_rank²) / (1/σ_reg² + 1/σ_rank²)

Operates strictly at the OUTPUT level — no retraining, no latent
modification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- Element class table (mirrors scripts/eval_physics_diag_table.py) ---
ACCEPTOR = {"Mg", "Zn", "Cu"}
DONOR = {"Sn", "Si", "Ti", "Ge"}
SUPER_DONOR = {"Sb", "Ta", "Bi", "W"}
ISOVALENT = {"Al", "Fe", "B", "V", "Er", "Eu"}

# Literature-derived ρ(log target vs log conc), per target.
LIT_RHO_VC = {"Mg": -0.95, "Zn": -0.90, "Cu": -0.85,
              "Sn": +0.95, "Si": +0.95, "Ti": +0.80, "Ge": +0.85,
              "Sb": +0.95, "Ta": +0.95, "Bi": +0.85}
LIT_RHO_PDR = {"Mg": +0.85, "Zn": +0.80, "Cu": +0.75,
               "Sn": -0.85, "Si": -0.80, "Ti": -0.70, "Ge": -0.75,
               "Sb": -0.85, "Ta": -0.85, "Bi": -0.75}

# Class hierarchy on the *target* axis (V_O direction). For PDR this is
# flipped: more V_O → fewer carriers → lower PDR.
# Higher integer ⇒ higher V_O ⇒ (for PDR) lower target.
CLASS_ORDER_VC = {"acceptor": 0, "isovalent": 1, "undoped": 1,
                  "donor": 2, "super_donor": 3}


def _expected_class(elem: str) -> str:
    if elem in ACCEPTOR:
        return "acceptor"
    if elem in DONOR:
        return "donor"
    if elem in SUPER_DONOR:
        return "super_donor"
    if elem in ISOVALENT:
        return "isovalent"
    if elem == "undoped":
        return "undoped"
    return "unknown"


def _atm_group(atm) -> str:
    if pd.isna(atm):
        return "(missing)"
    s = str(atm).lower()
    if "plasma" in s:
        return "O2_plasma"
    if "ar:o" in s or "ar/o" in s or "ar_o2" in s or "ar+o" in s:
        return "Ar+O2_mix"
    if s.startswith("o2") or s == "oxygen":
        return "O2"
    if s.startswith("ar") or s == "argon":
        return "Ar"
    if "air" in s:
        return "air"
    return s[:12]


def _bradley_terry_1d(refs: np.ndarray, signs: np.ndarray, weights: np.ndarray,
                      y_init: float, max_iter: int = 60, tol: float = 1e-5
                      ) -> tuple[float, float, int]:
    """Solve  argmin Σ w_i · -log σ(s_i · (y - y_i))  via Newton's method.

    s_i ∈ {-1, +1}: sign of expected (y_query - y_i).
    Hessian is positive (pure logistic loss) so the optimization is convex
    and 1-D Newton converges in <10 iterations.

    Returns (ŷ, σ_rank, n_used).
    σ_rank = 1/√Hessian at optimum.
    """
    n = len(refs)
    if n == 0:
        return float(y_init), float("inf"), 0
    y = float(y_init)
    for _ in range(max_iter):
        z = signs * (y - refs)                # [n]
        sig = 1.0 / (1.0 + np.exp(-z))        # σ(z)
        sig_neg = 1.0 - sig
        grad = -np.sum(weights * signs * sig_neg)
        hess = np.sum(weights * sig * sig_neg)
        if hess < 1e-9:
            break
        step = grad / hess
        y -= step
        if abs(step) < tol:
            break
    z = signs * (y - refs)
    sig = 1.0 / (1.0 + np.exp(-z))
    sig_neg = 1.0 - sig
    hess = float(np.sum(weights * sig * sig_neg))
    sigma = 1.0 / np.sqrt(max(hess, 1e-9))
    return y, sigma, n


def _build_pairs_for_query(query: dict, refs_df: pd.DataFrame, target: str,
                            min_dlogc: float = 0.10,
                            cross_class_weight: float = 0.30,
                            max_within_pairs: int = 30,
                            max_cross_pairs: int = 12,
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build (refs, signs, weights) for a single query row.

    Rule 1 (high-confidence) — same element + valid concentrations + |Δlogc|≥δ.
        Sign from LIT_RHO[elem] · sign(logc_q − logc_r). Weight 1.0.

    Rule 2 (medium-confidence) — only when Rule 1 yields <3 pairs and query
        element has a known class (acceptor/donor/super_donor). Pair against
        labeled refs from a *different* class. Sign from CLASS_ORDER_VC
        comparison (flipped for PDR). Weight `cross_class_weight`.
    """
    elem_q = query["dopant_label"]
    cls_q = _expected_class(elem_q)
    is_pdr = (target == "photo_dark_ratio")
    lit_rho = LIT_RHO_PDR if is_pdr else LIT_RHO_VC

    true_col = f"{target}_true"
    refs_with_y = refs_df.dropna(subset=[true_col]).copy()
    refs_with_y = refs_with_y[refs_with_y["sample_idx"] != query["sample_idx"]]

    info = dict(n_within=0, n_cross=0, atm_filtered=False)
    pair_refs, pair_signs, pair_w = [], [], []

    # --- Rule 1: within-element concentration ---
    rho = lit_rho.get(elem_q)
    c_q = query.get("concentration_at%")
    if rho is not None and c_q is not None and np.isfinite(c_q) and c_q > 0:
        same_elem = refs_with_y[refs_with_y.dopant_label == elem_q].copy()
        same_elem["c_r"] = pd.to_numeric(same_elem["concentration_at%"],
                                         errors="coerce")
        same_elem = same_elem[(same_elem.c_r > 0) & np.isfinite(same_elem.c_r)]
        if len(same_elem) > 0:
            same_elem["dlogc"] = np.log10(c_q) - np.log10(same_elem["c_r"])
            usable = same_elem[np.abs(same_elem.dlogc) >= min_dlogc]
            # Prefer same atmosphere group
            atm_q = _atm_group(query.get("atmosphere"))
            usable = usable.copy()
            usable["atm_g"] = usable["atmosphere"].apply(_atm_group)
            same_atm = usable[usable.atm_g == atm_q]
            chosen = same_atm if len(same_atm) >= 3 else usable
            if len(chosen) > max_within_pairs:
                chosen = chosen.sample(max_within_pairs, random_state=0)
            for _, r in chosen.iterrows():
                sign = float(np.sign(rho)) * float(np.sign(r.dlogc))
                if sign == 0:
                    continue
                pair_refs.append(float(r[true_col]))
                pair_signs.append(sign)
                pair_w.append(1.0)
            info["n_within"] = len(pair_refs)
            info["atm_filtered"] = (len(same_atm) >= 3)

    # --- Rule 2: cross-class hierarchy (only if within is sparse) ---
    if len(pair_refs) < 3 and cls_q in {"acceptor", "donor", "super_donor"}:
        order_q = CLASS_ORDER_VC[cls_q]
        # Flip class direction sign for PDR (more V_O → lower PDR)
        target_sign = -1.0 if is_pdr else +1.0
        cross = refs_with_y.copy()
        cross["cls"] = cross.dopant_label.apply(_expected_class)
        cross = cross[cross.cls.isin(CLASS_ORDER_VC.keys())]
        cross = cross[cross.cls != cls_q]
        cross = cross[cross.dopant_label != elem_q]
        cross = cross[cross.cls.isin({"acceptor", "donor", "super_donor"})]
        if len(cross) > 0:
            cross["order_r"] = cross.cls.map(CLASS_ORDER_VC)
            cross["sign_v"] = np.sign(order_q - cross.order_r)
            cross = cross[cross.sign_v != 0]
            # Sample at most max_cross_pairs distinct elements, balanced by class
            sampled = []
            for cls_r, grp in cross.groupby("cls"):
                k = max(1, max_cross_pairs // max(1, cross.cls.nunique()))
                sampled.append(grp.sample(min(k, len(grp)), random_state=0))
            sel = pd.concat(sampled) if sampled else cross.iloc[:0]
            for _, r in sel.iterrows():
                sign = target_sign * float(r.sign_v)
                if sign == 0:
                    continue
                pair_refs.append(float(r[true_col]))
                pair_signs.append(sign)
                pair_w.append(float(cross_class_weight))
            info["n_cross"] = len(pair_refs) - info["n_within"]

    return (np.asarray(pair_refs, dtype=float),
            np.asarray(pair_signs, dtype=float),
            np.asarray(pair_w, dtype=float),
            info)


def refine_oof(df: pd.DataFrame, target: str = "vacancy_concentration",
               platt_slope: float = 1.0, platt_intercept: float = 0.0,
               min_dlogc: float = 0.10, cross_class_weight: float = 0.30,
               sigma_reg_floor: float = 0.20,
               sigma_reg_scale: float = 1.0,
               sputter_only_refs: bool = True,
               unlabeled_only: bool = False,
               max_shift: float = 1.5,
               ) -> pd.DataFrame:
    """Apply RankRefine to every row in `df`.

    `df` must contain columns: sample_idx, dopant_label, atmosphere,
    concentration_at%, is_sputter, {target}_pred, {target}_pred_platt,
    {target}_std, {target}_true.

    Returns df with three new columns:
      {target}_pred_rank        — BT estimate ŷ^rank (NaN if no pairs)
      {target}_pred_refined     — inverse-variance fused
      {target}_alpha            — fusion weight on rank (0=unchanged, 1=pure rank)
    """
    pred_platt_col = f"{target}_pred_platt"
    std_col = f"{target}_std"
    true_col = f"{target}_true"

    out = df.copy().reset_index(drop=True)
    n = len(out)
    rank_col = f"{target}_pred_rank"
    refined_col = f"{target}_pred_refined"
    alpha_col = f"{target}_alpha"

    out[rank_col] = np.nan
    out[refined_col] = out[pred_platt_col].astype(float)
    out[alpha_col] = 0.0

    # σ_reg = MC-dropout std × |Platt slope| × scale, lower-bounded
    sigma_reg_arr = (
        out[std_col].astype(float).fillna(out[std_col].median()).to_numpy()
        * abs(platt_slope) * sigma_reg_scale
    ).clip(min=sigma_reg_floor)

    # Reference pool: labeled rows. By default sputter-only to keep refs
    # in-domain (sputter is the V_O scope of interest).
    ref_pool = out.dropna(subset=[true_col]).copy()
    if sputter_only_refs:
        ref_pool = ref_pool[ref_pool.is_sputter]

    diag = []
    for i in range(n):
        q = out.iloc[i]
        elem = q["dopant_label"]
        if elem == "undoped" or _expected_class(elem) == "unknown":
            continue
        # Optional: skip rows that have a true label (to avoid disturbing
        # parity on labeled rows where regression was fit to truth)
        if unlabeled_only and pd.notna(q[true_col]):
            continue
        refs, signs, ws, info = _build_pairs_for_query(
            q.to_dict(), ref_pool, target=target,
            min_dlogc=min_dlogc, cross_class_weight=cross_class_weight,
        )
        if len(refs) == 0:
            continue
        y_init = float(q[pred_platt_col])
        y_rank, sigma_rank, _ = _bradley_terry_1d(refs, signs, ws, y_init=y_init)
        if not np.isfinite(y_rank):
            continue
        # Cap absolute shift to limit damage from outlier truth points
        if abs(y_rank - y_init) > max_shift:
            y_rank = y_init + np.sign(y_rank - y_init) * max_shift
        sigma_reg = float(sigma_reg_arr[i])
        # Inverse-variance fusion
        inv_r2 = 1.0 / (sigma_reg ** 2)
        inv_k2 = 1.0 / max(sigma_rank ** 2, 1e-9)
        y_fused = (y_init * inv_r2 + y_rank * inv_k2) / (inv_r2 + inv_k2)
        alpha = inv_k2 / (inv_r2 + inv_k2)
        out.at[i, rank_col] = y_rank
        out.at[i, refined_col] = y_fused
        out.at[i, alpha_col] = float(alpha)
        diag.append(dict(
            sample_idx=int(q["sample_idx"]),
            element=elem,
            n_within=info["n_within"],
            n_cross=info["n_cross"],
            sigma_reg=sigma_reg,
            sigma_rank=sigma_rank,
            y_reg=y_init,
            y_rank=y_rank,
            y_fused=y_fused,
            alpha=alpha,
        ))
    diag_df = pd.DataFrame(diag)
    return out, diag_df
