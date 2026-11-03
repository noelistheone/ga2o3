"""μ_O(T, p_O2) — Reuter–Scheffler with NIST-JANAF Shomate O2 EOS (Ga2O3-Sandbox).

PROVENANCE: ported from Ga2O3-Net src/data/mu_o_table.py (V58, 2026-05). Physics unchanged.
Reference anchor: data/reference/chemical_potentials.json (QE HSE06 80-atom references,
O-rich limit), copied read-only from Ga2O3-Net dft/qe_hse06/results/.

    μ_O(T, p_O2) = μ_O^ref + ½ [ Δμ_O2(T) + k_B T ln(p_O2 / 1 atm) ]
    Δμ_O2(T) = [H°(T) − H°(0)] − T·S°(T)   (NIST Shomate, eV per O2, ref 0 K)
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

KB_EV = 8.617333262e-5
_H298_MINUS_H0_O2_KJMOL = 8.683
_KJMOL_PER_EV = 96.485332

_SHOMATE_O2 = {
    "low": dict(A=31.32234, B=-20.23531, C=57.86644, D=-36.50624,
                E=-0.007374, F=-8.903471, G=246.7945, H=0.0),
    "high": dict(A=30.03235, B=8.772972, C=-3.988133, D=0.788313,
                 E=-0.741599, F=-11.32468, G=236.1663, H=0.0),
}

_REF_JSON = Path(__file__).resolve().parents[2] / "data" / "reference" / "chemical_potentials.json"


def _shomate_delta_mu_o2(T: torch.Tensor) -> torch.Tensor:
    """Δμ_O2(T) in eV per O2, referenced to 0 K. T in K (float64 tensor, any shape)."""
    t = T / 1000.0

    def _branch(c):
        A, B, C, D, E, F, G = (c[k] for k in "ABCDEFG")
        H_minus_H298 = (A * t + B * t**2 / 2 + C * t**3 / 3 + D * t**4 / 4 - E / t + F)  # kJ/mol
        S = (A * torch.log(t) + B * t + C * t**2 / 2 + D * t**3 / 3 - E / (2 * t**2) + G)  # J/mol/K
        return H_minus_H298, S

    Hl, Sl = _branch(_SHOMATE_O2["low"])
    Hh, Sh = _branch(_SHOMATE_O2["high"])
    use_high = T >= 700.0
    H_minus_H298 = torch.where(use_high, Hh, Hl)
    S = torch.where(use_high, Sh, Sl)
    H_minus_H0 = H_minus_H298 + _H298_MINUS_H0_O2_KJMOL
    delta_mu_kjmol = H_minus_H0 - T * S / 1000.0
    return delta_mu_kjmol / _KJMOL_PER_EV


def load_reference(ref_json: str | Path = _REF_JSON) -> dict:
    """QE HSE06 chemical-potential limits (Ga_rich / O_rich / mid), absolute eV scale."""
    return json.loads(Path(ref_json).read_text())


def mu_o(T_K, p_O2_atm, mu_o_ref_eV: float | None = None, device: str | None = None) -> torch.Tensor:
    """μ_O(T, p_O2) in eV, anchored to the O-rich HSE06 reference by default. Broadcasts."""
    if mu_o_ref_eV is None:
        mu_o_ref_eV = float(load_reference()["limits"]["O_rich"]["mu_O"])
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    T = torch.as_tensor(T_K, dtype=torch.float64, device=device)
    p = torch.as_tensor(p_O2_atm, dtype=torch.float64, device=device)
    return mu_o_ref_eV + 0.5 * (_shomate_delta_mu_o2(T) + KB_EV * T * torch.log(p))


def delta_mu_o(T_K, p_O2_atm, device: str | None = None) -> torch.Tensor:
    """Δμ_O(T, p_O2) relative to the O-rich (T=0, 1 atm) reference — the quantity that shifts
    V_O formation energies: E_f(T,p) = E_f(O-rich) − Δμ_O. Always ≤ 0 for T>0 or p<1 atm."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    T = torch.as_tensor(T_K, dtype=torch.float64, device=device)
    p = torch.as_tensor(p_O2_atm, dtype=torch.float64, device=device)
    return 0.5 * (_shomate_delta_mu_o2(T) + KB_EV * T * torch.log(p))
