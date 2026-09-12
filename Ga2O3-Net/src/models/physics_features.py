"""
Phase 24A — first-principles physics features for the multimodal model.

Computes 8 interpretable physics scalars per (dopant_spec, process) pair:

    [0] E_photon_eV        photon energy of probe light (1240/λ)
    [1] E_g_eV             estimated optical bandgap of doped β-Ga₂O₃
    [2] E_photon_minus_Eg  above-bandgap excess (>0 → photoresponse possible)
    [3] kT_eV              thermal energy 8.617e-5·(T_C+273.15)
    [4] pO2_proxy          O₂ chemical-potential proxy (atmo + plasma boost)
    [5] radius_mismatch    (r_dop − r_Ga³⁺) / r_Ga³⁺  (Shannon CN=6 ionic radii)
    [6] valence_diff       dopant_valence − 3   (donor↑ vs acceptor↓ vs Ga)
    [7] electroneg_diff    χ_dopant − χ_Ga      (Pauling)

These quantities are *interpretable physics constants* — they are wired into
both the model input (auxiliary process channel) and the output envelope
(see `physics_envelope.py`).

Anion dopants (N, F, H) are referenced to O²⁻ rather than Ga³⁺ because they
substitute on the anion site; their valence/radius mismatch is reported as
the difference from the anion host.
"""
from __future__ import annotations

from typing import Iterable

import torch

from src.data.dopant_spec import DopantSpec, parse_spec

# ── Host references ──────────────────────────────────────────────────────────
GA_VALENCE = 3.0           # Ga³⁺
GA_RADIUS = 0.62           # Shannon CN=6, Å
GA_X = 1.81                # Pauling electronegativity

O_VALENCE = -2.0
O_RADIUS = 1.40            # Shannon CN=6
O_X = 3.44

BASE_E_G_EV = 4.85         # β-Ga₂O₃ optical bandgap (Onuma 2015, Tippins 1965)

# ── Per-element property table ───────────────────────────────────────────────
# valence: most-likely substitutional charge state in Ga₂O₃ host
# radius:  Shannon ionic radius for that valence at CN=6 (Å)
# X:       Pauling electronegativity
# eg_shift: empirical ΔE_g per unit dopant fraction (eV); positive=widening
# is_anion: substitutes on O site (rather than Ga site)
_ELEM_PROPS: dict[str, dict] = {
    # cations (Ga site)
    "Mg": dict(valence=2, radius=0.72,  X=1.31, eg_shift=-0.05, is_anion=False),
    "Zn": dict(valence=2, radius=0.74,  X=1.65, eg_shift=-0.03, is_anion=False),
    "Cu": dict(valence=2, radius=0.73,  X=1.90, eg_shift=-0.05, is_anion=False),
    "Sn": dict(valence=4, radius=0.69,  X=1.96, eg_shift=+0.02, is_anion=False),
    "Si": dict(valence=4, radius=0.40,  X=1.90, eg_shift=+0.05, is_anion=False),
    "Ti": dict(valence=4, radius=0.605, X=1.54, eg_shift= 0.00, is_anion=False),
    "Ge": dict(valence=4, radius=0.53,  X=2.01, eg_shift=+0.05, is_anion=False),
    "Fe": dict(valence=3, radius=0.645, X=1.83, eg_shift=-0.10, is_anion=False),
    "Cr": dict(valence=3, radius=0.615, X=1.66, eg_shift=-0.05, is_anion=False),
    "Mn": dict(valence=3, radius=0.645, X=1.55, eg_shift=-0.05, is_anion=False),
    "Al": dict(valence=3, radius=0.535, X=1.61, eg_shift=+0.50, is_anion=False),  # Al₂O₃ ~8.8 eV
    "In": dict(valence=3, radius=0.80,  X=1.78, eg_shift=-0.20, is_anion=False),  # In₂O₃ ~3.7 eV
    "B":  dict(valence=3, radius=0.27,  X=2.04, eg_shift=+0.30, is_anion=False),
    "Sb": dict(valence=5, radius=0.60,  X=2.05, eg_shift= 0.00, is_anion=False),
    "V":  dict(valence=5, radius=0.54,  X=1.63, eg_shift=-0.05, is_anion=False),
    "Ta": dict(valence=5, radius=0.64,  X=1.50, eg_shift= 0.00, is_anion=False),
    "Nb": dict(valence=5, radius=0.64,  X=1.60, eg_shift= 0.00, is_anion=False),
    "W":  dict(valence=6, radius=0.60,  X=2.36, eg_shift= 0.00, is_anion=False),
    "Mo": dict(valence=6, radius=0.59,  X=2.16, eg_shift= 0.00, is_anion=False),
    "Eu": dict(valence=3, radius=0.947, X=1.20, eg_shift=-0.05, is_anion=False),
    "Er": dict(valence=3, radius=0.890, X=1.24, eg_shift=-0.05, is_anion=False),
    "Y":  dict(valence=3, radius=0.900, X=1.22, eg_shift= 0.00, is_anion=False),
    "La": dict(valence=3, radius=1.030, X=1.10, eg_shift= 0.00, is_anion=False),
    "Nd": dict(valence=3, radius=0.983, X=1.14, eg_shift= 0.00, is_anion=False),
    "Gd": dict(valence=3, radius=0.938, X=1.20, eg_shift= 0.00, is_anion=False),
    "Tb": dict(valence=3, radius=0.923, X=1.10, eg_shift= 0.00, is_anion=False),
    "Dy": dict(valence=3, radius=0.912, X=1.22, eg_shift= 0.00, is_anion=False),
    "Yb": dict(valence=3, radius=0.868, X=1.10, eg_shift= 0.00, is_anion=False),
    "Sr": dict(valence=2, radius=1.18,  X=0.95, eg_shift= 0.00, is_anion=False),
    "Ba": dict(valence=2, radius=1.35,  X=0.89, eg_shift= 0.00, is_anion=False),
    "K":  dict(valence=1, radius=1.38,  X=0.82, eg_shift=-0.10, is_anion=False),
    "Na": dict(valence=1, radius=1.02,  X=0.93, eg_shift=-0.10, is_anion=False),
    "Zr": dict(valence=4, radius=0.72,  X=1.33, eg_shift= 0.00, is_anion=False),
    "Hf": dict(valence=4, radius=0.71,  X=1.30, eg_shift= 0.00, is_anion=False),
    "Co": dict(valence=2, radius=0.745, X=1.88, eg_shift=-0.05, is_anion=False),
    "Ni": dict(valence=2, radius=0.69,  X=1.91, eg_shift=-0.05, is_anion=False),
    "Bi": dict(valence=5, radius=0.76,  X=2.02, eg_shift= 0.00, is_anion=False),

    # anions (O site, or interstitial for H)
    "N":  dict(valence=-3, radius=1.46, X=3.04, eg_shift=-1.00, is_anion=True),  # large N→VB push
    "F":  dict(valence=-1, radius=1.33, X=3.98, eg_shift=+0.10, is_anion=True),
    "H":  dict(valence=+1, radius=0.20, X=2.20, eg_shift=-0.10, is_anion=True),  # interstitial donor
}

PHYSICS_FEATURE_NAMES: list[str] = [
    "E_photon_eV",
    "E_g_eV",
    "E_photon_minus_Eg",
    "kT_eV",
    "pO2_proxy",
    "radius_mismatch",
    "valence_diff",
    "electroneg_diff",
]
PHYSICS_FEATURE_DIM = len(PHYSICS_FEATURE_NAMES)


def _aggregate_dopant_props(spec_str: str) -> dict[str, float]:
    """
    Concentration-weighted mean over a (possibly multi-component) dopant spec.

    For undoped or unknown elements, returns Ga-host defaults so the
    feature differences become 0.
    """
    try:
        sp = DopantSpec.parse(spec_str)
    except Exception:
        return dict(valence=GA_VALENCE, radius=GA_RADIUS, X=GA_X,
                    eg_shift=0.0, total_conc=0.0)

    if sp.is_undoped or not sp.components:
        return dict(valence=GA_VALENCE, radius=GA_RADIUS, X=GA_X,
                    eg_shift=0.0, total_conc=0.0)

    total_conc = 0.0
    v_acc = r_acc = x_acc = eg_acc = 0.0
    cation_acc_conc = 0.0  # for radius/valence comparison vs Ga
    anion_radius_acc = 0.0
    anion_valence_acc = 0.0
    anion_x_acc = 0.0
    anion_conc = 0.0

    for c in sp.components:
        props = _ELEM_PROPS.get(c.cation)
        if props is None:
            # Unknown element → assume Ga-like (no contribution to mismatch)
            props = dict(valence=GA_VALENCE, radius=GA_RADIUS, X=GA_X,
                         eg_shift=0.0, is_anion=False)
        conc = float(c.conc)
        total_conc += conc
        eg_acc += props["eg_shift"] * conc

        if props["is_anion"]:
            anion_radius_acc  += props["radius"]  * conc
            anion_valence_acc += props["valence"] * conc
            anion_x_acc       += props["X"]       * conc
            anion_conc        += conc
        else:
            v_acc += props["valence"] * conc
            r_acc += props["radius"]  * conc
            x_acc += props["X"]       * conc
            cation_acc_conc += conc

    # Decide reference site: if anion-dominated, compare against O; else against Ga
    if anion_conc > cation_acc_conc and anion_conc > 0:
        valence = anion_valence_acc / anion_conc
        radius  = anion_radius_acc  / anion_conc
        X       = anion_x_acc       / anion_conc
        # Repurpose host: caller will subtract O reference
        return dict(valence=valence, radius=radius, X=X, eg_shift=eg_acc,
                    total_conc=total_conc, host="O")

    if cation_acc_conc > 0:
        valence = v_acc / cation_acc_conc
        radius  = r_acc / cation_acc_conc
        X       = x_acc / cation_acc_conc
    else:
        valence, radius, X = GA_VALENCE, GA_RADIUS, GA_X

    return dict(valence=valence, radius=radius, X=X, eg_shift=eg_acc,
                total_conc=total_conc, host="Ga")


def compute_physics_features(
    dopant_specs: Iterable[str],
    process: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the 8-dim physics feature tensor.

    Args:
        dopant_specs: list[str] of length B (DopantSpec strings).
        process:      Tensor [B, 18] in build_process_tensor layout.

    Returns:
        Tensor [B, 8] of dtype matching `process`.
    """
    if process.dim() != 2 or process.shape[1] < 11:
        raise ValueError(f"process must be [B, ≥11], got {tuple(process.shape)}")

    device = process.device
    dtype = process.dtype
    B = process.shape[0]

    T_C = process[:, 0]
    o2 = process[:, 2]
    is_plasma = process[:, 3]
    wl_nm = process[:, 10].clamp_min(50.0)

    E_photon = 1240.0 / wl_nm
    kT = 8.617e-5 * (T_C + 273.15)
    # pO2 proxy: O₂ fraction with plasma boost (plasma O has higher μ_O)
    pO2_proxy = o2 * (1.0 + 0.5 * is_plasma)

    valences = torch.empty(B, device=device, dtype=dtype)
    radii    = torch.empty(B, device=device, dtype=dtype)
    Xs       = torch.empty(B, device=device, dtype=dtype)
    eg_shift = torch.empty(B, device=device, dtype=dtype)
    radius_ref  = torch.empty(B, device=device, dtype=dtype)
    valence_ref = torch.empty(B, device=device, dtype=dtype)
    x_ref       = torch.empty(B, device=device, dtype=dtype)

    for i, s in enumerate(dopant_specs):
        p = _aggregate_dopant_props(s)
        valences[i] = p["valence"]
        radii[i]    = p["radius"]
        Xs[i]       = p["X"]
        eg_shift[i] = p["eg_shift"]
        if p.get("host") == "O":
            radius_ref[i]  = O_RADIUS
            valence_ref[i] = O_VALENCE
            x_ref[i]       = O_X
        else:
            radius_ref[i]  = GA_RADIUS
            valence_ref[i] = GA_VALENCE
            x_ref[i]       = GA_X

    E_g = BASE_E_G_EV + eg_shift
    radius_mismatch = (radii - radius_ref) / radius_ref
    valence_diff = valences - valence_ref
    electroneg_diff = Xs - x_ref

    feats = torch.stack(
        [
            E_photon,
            E_g,
            E_photon - E_g,
            kT,
            pO2_proxy,
            radius_mismatch,
            valence_diff,
            electroneg_diff,
        ],
        dim=-1,
    )
    return feats.to(dtype=dtype)


# ── V_O formation-energy empirical proxy (Phase 30B / D) ─────────────────────
# Coarse anchor for BrouwerHeadVC's learned E_f^VO. NOT DFT — a Pauling-rule
# scaling that captures the dominant donor-vs-acceptor sign of the dopant's
# effect on the Fermi level and hence on the +2 V_O formation energy.
#
# Physics: under O-poor sputter conditions V_O is dominantly V_O^{2+}, whose
# formation energy carries a +2·(E_F − E_VBM) term (Walle 2014). Acceptor
# dopants pin E_F lower → lower E_f^VO → more vacancies; donors pin E_F
# higher → higher E_f^VO → fewer vacancies.
#
#     E_f^VO(eV) ≈ E_BASELINE + α · valence_diff_eff
#
# where valence_diff_eff = (v̄_dopant − 3) · saturation(conc) and the
# saturation factor caps the linear effect at high doping (Fermi pinning by
# the conduction-band DOS).
VO_E_F_BASELINE_EV = 2.50      # β-Ga₂O₃ undoped V_O^{2+} formation, O-poor (lit. ~2-3 eV)
VO_ALPHA_PER_CHARGE = 0.40     # Δ E_f per unit valence_diff at saturation
VO_CONC_SAT_K = 100.0          # log10(1 + K·conc) — saturates ~1% doping


def compute_vo_formation_proxy(dopant_specs: Iterable[str]) -> torch.Tensor:
    """
    Empirical V_O^{2+} formation-energy proxy in β-Ga₂O₃ [eV] per sample.

    Args:
        dopant_specs: list[str] of length B (DopantSpec strings).

    Returns:
        Tensor [B] of float32 — eV, intended as a soft physics anchor for
        BrouwerHeadVC.E_f. Undoped samples → ``VO_E_F_BASELINE_EV``.
    """
    out = []
    for s in dopant_specs:
        p = _aggregate_dopant_props(s)
        if p.get("host") == "O":
            # Anion dopants modify E_g and lattice but not the cation-side
            # Fermi-level argument. Default to baseline.
            out.append(VO_E_F_BASELINE_EV)
            continue
        v = float(p["valence"])
        conc = float(p.get("total_conc", 0.0))
        # log saturation: 0 at conc=0, ~log10(2)≈0.30 at 1%, ~log10(11)≈1.04 at 10%
        sat = torch.log10(torch.tensor(1.0 + VO_CONC_SAT_K * max(conc, 0.0))).item()
        valence_diff_eff = (v - GA_VALENCE) * sat
        E_f = VO_E_F_BASELINE_EV + VO_ALPHA_PER_CHARGE * valence_diff_eff
        out.append(E_f)
    return torch.tensor(out, dtype=torch.float32)
