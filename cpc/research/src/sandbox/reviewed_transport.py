"""Explicit transport interface for source-audited predictions.

The legacy nine-output simulator remains available for reproducing historical analyses.
This interface accepts incorporated volume density only, reports the 300 K Hall observable,
and never changes to an anneal-state value when the quenched calculation is ill-conditioned.
"""
from __future__ import annotations
import math
from . import full_forward as ff


def predict(dopant: str, incorporated_cm3: float, *, freeze_temperature_K: float,
            oxygen_pressure_atm: float, background_donor_cm3: float,
            grain_boundary_barrier_eV: float = 0.0, grain_boundaries: bool = False,
            db=None) -> dict:
    try:
        incorporated_cm3=float(incorporated_cm3)
        freeze_temperature_K=float(freeze_temperature_K)
        oxygen_pressure_atm=float(oxygen_pressure_atm)
        background_donor_cm3=float(background_donor_cm3)
    except (TypeError,ValueError) as exc:
        raise ValueError("Explicit finite density, temperature and pressure values are required") from exc
    if not math.isfinite(incorporated_cm3) or incorporated_cm3 < 0:
        raise ValueError("A finite incorporated dopant density in cm^-3 is required")
    if incorporated_cm3 > ff.N_CAT_SITE:
        raise ValueError("Incorporated dopant density exceeds the cation site density")
    if not math.isfinite(freeze_temperature_K) or freeze_temperature_K <= 0:
        raise ValueError("freeze_temperature_K must be positive")
    if not math.isfinite(oxygen_pressure_atm) or oxygen_pressure_atm <= 0:
        raise ValueError("oxygen_pressure_atm must be positive")
    if not math.isfinite(background_donor_cm3) or background_donor_cm3 < 0:
        raise ValueError("background_donor_cm3 must be finite and nonnegative")
    r=ff.forward(dopant,incorporated_cm3/ff.N_CAT_SITE,T_anneal=freeze_temperature_K,
                 pO2=oxygen_pressure_atm,Nd_bg=background_donor_cm3,film=grain_boundaries,
                 E_B_eV=grain_boundary_barrier_eV,db=db,undoped_cache={})
    reasons=[]
    for field in ['compensation_abstain','site_blocking_limited','above_reported_incorporation_ceiling']:
        if r[field]: reasons.append(field)
    if r['hall_p_cm3'] >= r['free_electron_cm3']:
        reasons.append('single_electron_Hall_approximation_not_applicable')
    if r['hall_density_cm3'] < ff.N_DET_CM3:
        reasons.append('below_legacy_Hall_comparability_threshold')
    comparable = not reasons
    return dict(dopant=dopant,incorporated_input_cm3=incorporated_cm3,
        measurement_temperature_K=300.,prediction_stage='quenched',
        free_electron_cm3=r['free_electron_cm3'],hall_factor=r['hall_factor_rH'],
        raw_hall_density_cm3=r['hall_density_cm3'],
        comparable_hall_density_cm3=r['hall_density_cm3'] if comparable else None,
        hall_mobility_cm2Vs=r['hall_mu_cm2Vs'],drift_mobility_cm2Vs=r['mu_drift_cm2Vs'],
        conductivity_S_cm=r['sigma_S_cm'],comparison_refusal_reasons=reasons,
        capability='Physical scenario calculation; numerical comparability is not experimental validation',
        hall_factor_note='Heuristic interpolation; evaluate r_H=1 sensitivity when convention is unknown',
        uncertainty_note='Abstention thresholds are sensitivity diagnostics, not calibrated confidence intervals')
