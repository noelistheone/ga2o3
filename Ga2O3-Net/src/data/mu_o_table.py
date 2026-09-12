"""V58 — Reuter-Scheffler oxygen chemical potential μ_O(T, p_O2) table.

Design ref: phase58_v58_dft_llm_hybrid_design.md §4.2 + §7.8.

    μ_O(T, p_O2) = μ_O^ref + ½ [ Δμ_O2(T) + k_B T ln(p_O2 / p°) ]

where
  - μ_O^ref is the T=0 K standard (O-rich-limit) oxygen reference taken from the
    *same* HSE06 reference calculation that produced the 87 ΔE_f values
    (`dft/qe_hse06/results/chemical_potentials.json` → limits.O_rich.mu_O).
    Anchoring to that file is what makes the V58 μ_O table commensurable with the
    stored ΔE_f (H3 traceability — see design-doc §7 "⚠ 实现决策" callout).
  - Δμ_O2(T) = [H°(T) − H°(0)] − T·S°(T)  (eV per O2, ref 0 K) is computed from the
    NIST-JANAF / Shomate equation of state for O2 gas (rigorous; the doc's
    illustrative "800°C → −1.42 eV" is superseded by the exact Shomate value).
  - p° = 1 atm.

GPU-over-CPU (memory rule feedback_gpu_over_cpu, 2026-05-27): the 1000×100 grid is
built as a single broadcast op on CUDA when available, then moved to host once.
The Shomate math is bit-identical on CPU/GPU (float64), so accuracy is unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

# Boltzmann constant in eV/K
KB_EV = 8.617333262e-5

# O2(g) standard enthalpy difference H°(298.15 K) − H°(0 K), kJ/mol (NIST-JANAF).
_H298_MINUS_H0_O2_KJMOL = 8.683

# NIST Shomate coefficients for O2 gas (two temperature ranges).
# Cp° = A + B t + C t^2 + D t^3 + E/t^2          (J/mol/K),  t = T/1000
# H°(T) − H°(298.15) = A t + B t^2/2 + C t^3/3 + D t^4/4 − E/t + F − H   (kJ/mol)
# S°(T) = A ln t + B t + C t^2/2 + D t^3/3 − E/(2 t^2) + G               (J/mol/K)
_SHOMATE_O2 = {
    # 100 – 700 K
    "low": dict(A=31.32234, B=-20.23531, C=57.86644, D=-36.50624,
                E=-0.007374, F=-8.903471, G=246.7945, H=0.0, tmin=100.0, tmax=700.0),
    # 700 – 2000 K
    "high": dict(A=30.03235, B=8.772972, C=-3.988133, D=0.788313,
                 E=-0.741599, F=-11.32468, G=236.1663, H=0.0, tmin=700.0, tmax=2000.0),
}

_KJMOL_PER_EV = 96.485332  # 1 eV/atom = 96.485 kJ/mol


def _shomate_delta_mu_o2(T: torch.Tensor) -> torch.Tensor:
    """Δμ_O2(T) = [H°(T)−H°(0)] − T·S°(T) in eV per O2, referenced to 0 K.

    T: tensor of temperatures in K (any shape, float64). Returns same shape.
    Vectorized; picks the low/high Shomate set per element via torch.where.
    """
    t = T / 1000.0

    def _branch(c):
        A, B, C, D, E, F, G = (c[k] for k in "ABCDEFG")
        H_minus_H298 = (A * t + B * t**2 / 2 + C * t**3 / 3 + D * t**4 / 4
                        - E / t + F)  # kJ/mol  (H term is 0 for O2)
        S = (A * torch.log(t) + B * t + C * t**2 / 2 + D * t**3 / 3
             - E / (2 * t**2) + G)  # J/mol/K
        return H_minus_H298, S

    Hl, Sl = _branch(_SHOMATE_O2["low"])
    Hh, Sh = _branch(_SHOMATE_O2["high"])
    use_high = T >= 700.0
    H_minus_H298 = torch.where(use_high, Hh, Hl)  # kJ/mol
    S = torch.where(use_high, Sh, Sl)             # J/mol/K

    H_minus_H0 = H_minus_H298 + _H298_MINUS_H0_O2_KJMOL  # kJ/mol
    # Δμ_O2(T) = (H°(T)−H°(0)) − T S°(T) ; convert kJ/mol → eV
    delta_mu_kjmol = H_minus_H0 - T * S / 1000.0
    return delta_mu_kjmol / _KJMOL_PER_EV  # eV per O2


def mu_o(T_K: torch.Tensor, p_O2_atm: torch.Tensor, mu_o_ref_eV: float,
         device: str | None = None) -> torch.Tensor:
    """μ_O(T, p_O2) in eV (absolute scale anchored to mu_o_ref_eV).

    Broadcasts T_K against p_O2_atm. Computed on `device` (cuda if available).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    T = torch.as_tensor(T_K, dtype=torch.float64, device=device)
    p = torch.as_tensor(p_O2_atm, dtype=torch.float64, device=device)
    delta_mu_o2 = _shomate_delta_mu_o2(T)                       # eV per O2
    kT_lnp = KB_EV * T * torch.log(p)                           # eV per O2
    return mu_o_ref_eV + 0.5 * (delta_mu_o2 + kT_lnp)           # eV per O atom


def build_table(
    chem_pot_json: str | Path = "dft/qe_hse06/results/chemical_potentials.json",
    out_path: str | Path = "data/processed/mu_O_table_v58.npz",
    n_T: int = 1000,
    n_p: int = 100,
    T_min: float = 300.0,
    T_max: float = 1500.0,
    p_min: float = 1e-6,
    p_max: float = 1.0,
) -> dict:
    """Build the 1000×100 μ_O(T, p_O2) lookup table and save as NPZ.

    Returns a small dict of metadata (also embedded in the NPZ).
    """
    chem = json.loads(Path(chem_pot_json).read_text())
    mu_o_ref = float(chem["limits"]["O_rich"]["mu_O"])  # T=0 O-rich standard reference

    device = "cuda" if torch.cuda.is_available() else "cpu"
    T_grid = torch.linspace(T_min, T_max, n_T, dtype=torch.float64, device=device)
    logp_grid = torch.linspace(np.log10(p_min), np.log10(p_max), n_p,
                               dtype=torch.float64, device=device)
    p_grid = 10.0 ** logp_grid

    # Single broadcast op on GPU: [n_T, 1] vs [1, n_p] -> [n_T, n_p]
    mu = mu_o(T_grid[:, None], p_grid[None, :], mu_o_ref, device=device)  # [n_T, n_p]

    T_np = T_grid.cpu().numpy()
    logp_np = logp_grid.cpu().numpy()
    mu_np = mu.cpu().numpy()

    meta = dict(
        mu_o_ref_O_rich_eV=mu_o_ref,
        mu_o_Ga_rich_eV=float(chem["limits"]["Ga_rich"]["mu_O"]),
        mu_o_mid_eV=float(chem["limits"]["mid"]["mu_O"]),
        source="Reuter-Scheffler + NIST-JANAF Shomate O2; ref=chemical_potentials.json O_rich",
        functional="HSE06-consistent (mu_O_ref from 80-atom HSE06 references)",
        T_min=T_min, T_max=T_max, p_min=p_min, p_max=p_max,
        n_T=n_T, n_p=n_p, device_built=device,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        T_grid_K=T_np,
        log10_p_O2_atm=logp_np,
        mu_O_eV=mu_np,
        meta_json=json.dumps(meta),
    )
    return meta


def lookup(npz_path: str | Path, T_K: float, p_O2_atm: float) -> float:
    """Bilinear lookup of μ_O at (T_K, p_O2_atm) from a saved table."""
    d = np.load(npz_path, allow_pickle=False)
    T_grid = d["T_grid_K"]
    logp_grid = d["log10_p_O2_atm"]
    mu = d["mu_O_eV"]
    lp = np.log10(max(p_O2_atm, 10.0 ** logp_grid[0]))
    lp = min(lp, logp_grid[-1])
    Tc = float(np.clip(T_K, T_grid[0], T_grid[-1]))
    i = np.clip(np.searchsorted(T_grid, Tc) - 1, 0, len(T_grid) - 2)
    j = np.clip(np.searchsorted(logp_grid, lp) - 1, 0, len(logp_grid) - 2)
    ti = (Tc - T_grid[i]) / (T_grid[i + 1] - T_grid[i])
    pj = (lp - logp_grid[j]) / (logp_grid[j + 1] - logp_grid[j])
    m = (mu[i, j] * (1 - ti) * (1 - pj) + mu[i + 1, j] * ti * (1 - pj)
         + mu[i, j + 1] * (1 - ti) * pj + mu[i + 1, j + 1] * ti * pj)
    return float(m)


if __name__ == "__main__":
    import sys
    proj = Path(__file__).resolve().parents[2]
    import os
    os.chdir(proj)
    meta = build_table()
    print("Built mu_O_table_v58.npz")
    print(json.dumps(meta, indent=2))
    # Sanity prints at representative sputter conditions
    npz = "data/processed/mu_O_table_v58.npz"
    for T_C, atm, p in [(700, "O2 1atm", 1.0), (700, "Ar:O2=4:1", 0.2),
                        (700, "Ar (~1e-4)", 1e-4), (900, "O2", 1.0), (500, "Ar", 1e-4)]:
        print(f"  T={T_C}C ({T_C+273.15:.0f}K) {atm:14s} p_O2={p:.0e}  "
              f"mu_O = {lookup(npz, T_C + 273.15, p):.4f} eV")
