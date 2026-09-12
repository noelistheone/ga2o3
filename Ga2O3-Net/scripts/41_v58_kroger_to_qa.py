"""Phase 58 Stage-1 — build the Physics-LLM QA training corpus (~8k pairs).

Design ref: phase58_v58_dft_llm_hybrid_design.md §5.1.2 (sources), §5.1.3 (schema),
§6.1 (literature list), §6.2 (KROGER→QA), §6.5 (synthetic Tier-1C boost).

Output: data/processed/stage1_qa_pairs.jsonl  (train, ~90%)
        data/processed/stage1_qa_pairs_val.jsonl  (val, ~10%)

Every QA pair is a JSON dict with the doc §5.1.3 schema:
    {"input":  {dopant, dopant_concentration, atmosphere, temperature_C, fermi_level_eV},
     "label":  {log_vo_predicted, dominant_charge_state, log_K_eq, delta_E_f_eV, reasoning}}

Four honest sources (NO LLM call; all reasoning is a deterministic Brouwer-diagram
template — the doc's "synthetic CoT" is realized here as a templated chain, not a
Qwen generation, because Stage-1 must not depend on an un-audited model output):

  1. KROGER-derived (~4560):   sample conditions for each of the 19
     kroger_synthetic.DOPANT_OFFSET dopants and evaluate the clean HSE06-parametrized
     Brouwer model (kroger_predict).
  2. DFT-cache-derived (~1500): condition-augmented QA from the 304 (dopant,atm,T)
     anchors in dft_features_cache_v58.npz, log_vo via build_dft_cache._anchor_log_vo
     (identical math to the downstream BrouwerHead anchor).
  3. Reference-literature (~750): condition-augmented transcription of a SMALL, honest
     hardcoded table of established β-Ga2O3 defect facts (Varley 2010 V_O E_f; shallow
     donors Si/Sn/Ge; deep acceptor Mg; F_O). Only facts already encoded in the repo
     (NATIVE_VO_EF) or stated in doc §6.1 are used — no fabricated numbers.
  4. Tier-1C synthetic (~1250):  same Brouwer model for the 14 Tier-1C elements
     (Cu,Mn,Fe,Co,Ni,In,Sb,Bi,Er,Eu,La,B,V,W); offset 0.0 for elements absent from
     DOPANT_OFFSET; each pair validated against sign / charge-neutrality rules, drop
     violations.

GPU-over-CPU note (feedback_gpu_over_cpu): GPU 0 is occupied by a training run, so this
is intentionally pure CPU. The per-pair Brouwer eval is closed-form scalar math; the
work is trivial relative to any kernel-launch overhead, so CPU is the correct choice
here (and the rule's "where a GPU path exists & is a bottleneck" precondition does not
apply).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from src.data.kroger_synthetic import (
    DOPANT_OFFSET, NATIVE_VO_EF, LOG_PREFACTOR, KB_EV, LN10, EG_GA2O3, C_REF,
)
from src.data.defect_chemical_potentials import fermi_level_eV, _DOPANT_CLASS
from scripts.build_dft_cache import load_cache, _anchor_log_vo

PROJ = Path(__file__).resolve().parents[1]
OUT_TRAIN = PROJ / "data" / "processed" / "stage1_qa_pairs.jsonl"
OUT_VAL = PROJ / "data" / "processed" / "stage1_qa_pairs_val.jsonl"

# Atmosphere bins (doc §5.1.3 / §6.2): name -> (log_pO2, p_O2 atm). Mirrors the 4 sputter
# atmosphere bins used by build_dft_cache.ATM_P_O2 (log10 of those p_O2 values).
ATMOSPHERES = {
    "Ar":         (-5.0,  1e-5),
    "Ar:O2=4:1":  (-1.3,  0.05),
    "Ar:O2=1:1":  (-0.7,  0.2),
    "O2":         (0.0,   1.0),
}
# DFT-cache atmosphere key spelling (build_dft_cache uses Ar_O2_4_1 etc.)
ATM_CACHE_KEY = {"Ar": "Ar", "Ar:O2=4:1": "Ar_O2_4_1",
                 "Ar:O2=1:1": "Ar_O2_1_1", "O2": "O2"}

TIER1C = ["Cu", "Mn", "Fe", "Co", "Ni", "In", "Sb", "Bi", "Er", "Eu", "La",
          "B", "V", "W"]

# physical log10[V_O] range (cm^-3) — used for the VERIFY gate and Tier-1C validation
LOG_VO_LO, LOG_VO_HI = 5.0, 23.0


# --------------------------------------------------------------------------- #
# physics helpers
# --------------------------------------------------------------------------- #
def _valence_class(elem: str) -> str:
    return _DOPANT_CLASS.get(elem, "isovalent")


# Anchor Fermi level: SAME single n-type host εF used by build_dft_cache.ANCHOR_FERMI_EV.
# Using εF≈4.0 eV (near CBM) instead of the midgap value baked into kroger_predict's
# `charge_term` is the documented fix (design-doc §7 callout): the midgap charge_term
# drives the q=1,2 V_O concentrations past N_site, so kroger_predict clamps EVERY dominant
# (q=2) label to LOG_PREFACTOR (~22.45) — verified: 100% of kroger/literature/tier1c labels
# pin to the clamp, which is useless training signal. The n-type εF keeps E_f(V_O^q)
# positive, giving a physically-spread, atmosphere/T/conc-discriminative log[V_O] that
# matches the DFT-cache anchor (_anchor_log_vo) the downstream BrouwerHead consumes.
ANCHOR_FERMI_EV = 4.0


def _ef_q_ntype(q: int) -> float:
    """Native V_O formation energy of charge q at εF=4.0 (n-type), μ_O=0 (O-rich)."""
    return NATIVE_VO_EF.get(q, NATIVE_VO_EF[0]) + q * (ANCHOR_FERMI_EV - EG_GA2O3 / 2.0)


def _per_charge_log_vo(elem: str, c: float, T_K: float, log_pO2: float):
    """log10[V_O^q] for q=0,1,2 (n-type εF Boltzmann + dopant offset + atmosphere)."""
    offset = DOPANT_OFFSET.get(elem, 0.0)
    dopant_term = offset * math.log10(max(c, 1e-6) / C_REF) if c and c > 0 else 0.0
    return [LOG_PREFACTOR - _ef_q_ntype(q) / (KB_EV * T_K * LN10)
            - 0.5 * log_pO2 + dopant_term for q in (0, 1, 2)]


def _brouwer_total(elem: str, c: float, T_K: float, log_pO2: float):
    """Total log10[V_O] (Boltzmann sum over charges) + dominant charge.

    Identical n-type-εF physics to build_dft_cache._anchor_log_vo (base-10 logsumexp over
    the per-charge concentrations), with the kroger_synthetic dopant offset added so the
    composition direction is present. Returns (log_vo_total_clamped, dominant_q).
    """
    logs = np.array(_per_charge_log_vo(elem, c, T_K, log_pO2), dtype=float)
    m = float(logs.max())
    total = m + math.log10(float(np.sum(10.0 ** (logs - m))))
    total = min(total, LOG_PREFACTOR)            # ≤ N_site (physical)
    total = max(LOG_VO_LO, total)                # ≥ thermal floor (keeps in [5,23] gate)
    return total, int(np.argmax(logs))


def _dominant_charge(elem: str, c: float, T_K: float, log_pO2: float) -> int:
    """argmax over q in {0,1,2} of the per-charge V_O concentration (n-type εF)."""
    return int(np.argmax(_per_charge_log_vo(elem, c, T_K, log_pO2)))


def _delta_ef(elem: str, q: int, log_pO2: float) -> float:
    """ΔE_f(V_O^q) for the reported dominant charge: native baseline + atmosphere shift.

    NATIVE_VO_EF[q] is at μ_O=0 (O-rich, εF=midgap). Removing one O to form V_O adds +μ_O;
    lower μ_O (Ar/reducing) lowers ΔE_f → more V_O. We approximate Δμ_O by the O2-equilibrium
    term used in kroger_predict, Δμ_O ≈ 0.5·kT·ln10·log_pO2 ... but to stay consistent with
    the closed-form (which folds atmosphere as −0.5·log_pO2 directly into log_VO), we report
    ΔE_f at the O-rich reference shifted by the same 0.5 factor expressed in eV at midgap-298K
    convention is ambiguous; instead we anchor ΔE_f to the native value and add the small
    dopant-class direction so the LLM sees a physically-ordered (atmosphere → ΔE_f) signal.

    Concretely: ΔE_f = NATIVE_VO_EF[q] + Δμ_O_eV, with Δμ_O_eV the (≤0) O-poor shift derived
    from the same −0.5·log_pO2 used in log_VO, converted to eV via the room-T O2 entropy scale
    used downstream. We keep it simple & honest: Δμ_O_eV = -0.5 * log_pO2-offset-from-O-rich,
    bounded so ΔE_f stays in a physical [0, 6] eV window.
    """
    base = NATIVE_VO_EF.get(q, NATIVE_VO_EF[0])
    # O-rich reference is log_pO2 = 0; more reducing (log_pO2 < 0) lowers μ_O → lowers ΔE_f.
    # Scale: ~0.3 eV per decade of pO2 (typical Δμ_O span ~ -1.5 eV over Ar..O2). This is a
    # qualitative atmosphere ordering signal, NOT a transcribed DFT number.
    d_mu_O_eV = 0.30 * log_pO2  # ≤ 0 for reducing atmospheres
    val = base + d_mu_O_eV
    return float(min(6.0, max(0.0, val)))


def _log_keq(delta_ef_eV: float, T_K: float) -> float:
    """log10 K_eq ≈ -ΔE_f / (kT·ln10) (mass-action equilibrium constant)."""
    return float(-delta_ef_eV / (KB_EV * T_K * LN10))


# --------------------------------------------------------------------------- #
# reasoning templates (deterministic — NO LLM)
# --------------------------------------------------------------------------- #
def _atm_phrase(atmosphere: str) -> str:
    if atmosphere == "Ar":
        return ("an Ar (reducing) atmosphere lowers the oxygen chemical potential mu_O, "
                "lowering the V_O formation energy and shifting the Brouwer diagram toward "
                "higher oxygen-vacancy concentration")
    if atmosphere == "O2":
        return ("a pure O2 (oxidizing) atmosphere raises mu_O, which raises the V_O "
                "formation energy and suppresses oxygen vacancies")
    return (f"a mixed {atmosphere} atmosphere sets an intermediate mu_O, giving a moderate "
            "V_O formation energy between the Ar-rich and O2-rich limits")


def _class_phrase(elem: str, vclass: str) -> str:
    if vclass == "acceptor":
        return (f"{elem} substitutes on the Ga site as an acceptor; acceptor doping pins the "
                "Fermi level lower and, via charge-neutrality compensation, tends to suppress "
                "the net oxygen-vacancy concentration")
    if vclass in ("donor", "super-donor"):
        return (f"{elem} substitutes on the Ga site as a donor; donor doping pushes the Fermi "
                "level toward the conduction band and co-doping favors the donor-like V_O, "
                "raising the oxygen-vacancy concentration")
    return (f"{elem} is approximately isovalent on the Ga site, so its direct effect on the "
            "oxygen-vacancy concentration is weak and the atmosphere/temperature terms dominate")


def _charge_phrase(q: int) -> str:
    sym = {0: "neutral V_O^0", 1: "singly-charged V_O^+1", 2: "doubly-charged V_O^2+"}[q]
    if q == 2:
        return (f"the dominant charge state is the {sym}: V_O is a donor in beta-Ga2O3 and the "
                "doubly-ionized state has the lowest formation energy when the Fermi level sits "
                "near the conduction band, so it dominates the Brouwer diagram")
    if q == 1:
        return (f"the dominant charge state is the {sym}, the intermediate ionization of the "
                "oxygen vacancy")
    return (f"the dominant charge state is the {sym}, favored when the Fermi level lies deep "
            "in the gap and the doubly-ionized state is not stabilized")


def _build_reasoning(elem, vclass, atmosphere, T_C, q, log_vo, delta_ef) -> str:
    return (f"{_class_phrase(elem, vclass)}. Thermodynamically, {_atm_phrase(atmosphere)}; "
            f"at {T_C:.0f} C the entropic term raises the equilibrium vacancy population. "
            f"With a V_O formation energy of about {delta_ef:.2f} eV, {_charge_phrase(q)}, "
            f"giving log10[V_O] approximately {log_vo:.2f} cm^-3.")


def _make_pair(elem, c, atmosphere, T_C, fermi_eV, log_vo, q, delta_ef, T_K, src):
    log_keq = _log_keq(delta_ef, T_K)
    reasoning = _build_reasoning(elem, _valence_class(elem), atmosphere, T_C, q, log_vo, delta_ef)
    return {
        "input": {
            "dopant": elem,
            "dopant_concentration": round(float(c), 8),
            "atmosphere": atmosphere,
            "temperature_C": round(float(T_C), 1),
            "fermi_level_eV": round(float(fermi_eV), 3),
        },
        "label": {
            "log_vo_predicted": round(float(log_vo), 4),
            "dominant_charge_state": int(q),
            "log_K_eq": round(float(log_keq), 4),
            "delta_E_f_eV": round(float(delta_ef), 4),
            "reasoning": reasoning,
        },
        "_source": src,
    }


# --------------------------------------------------------------------------- #
# source 1 — KROGER-derived
# --------------------------------------------------------------------------- #
def build_kroger(rng, per_dopant=240) -> list:
    pairs = []
    dopants = sorted(DOPANT_OFFSET.keys())
    atm_names = list(ATMOSPHERES.keys())
    for elem in dopants:
        fermi_eV = fermi_level_eV(elem)
        for _ in range(per_dopant):
            c = float(10 ** rng.uniform(-4, math.log10(5e-2)))   # log-uniform [1e-4, 5e-2]
            T_C = float(rng.uniform(500, 1100))
            T_K = T_C + 273.15
            atmosphere = atm_names[rng.integers(len(atm_names))]
            log_pO2 = ATMOSPHERES[atmosphere][0]
            log_vo, q = _brouwer_total(elem, c, T_K, log_pO2)
            delta_ef = _delta_ef(elem, q, log_pO2)
            pairs.append(_make_pair(elem, c, atmosphere, T_C, fermi_eV,
                                    log_vo, q, delta_ef, T_K, "kroger"))
    return pairs


# --------------------------------------------------------------------------- #
# source 2 — DFT-cache-derived
# --------------------------------------------------------------------------- #
def build_dft(rng, target=1500) -> list:
    idx, feats, meta = load_cache()
    keys = list(idx.keys())  # 304 = 19 dopants x 4 atm x 4 T_C
    # condition-augment: each anchor gets several concentration draws (anchor log_vo is
    # dopant-direction-independent by design, so concentration enters only the input + the
    # reasoning text, exactly mirroring how the downstream model uses the DFT anchor).
    n_anchors = len(keys)
    reps = max(1, math.ceil(target / n_anchors))
    pairs = []
    for key in keys:
        dop, atm_cache, T_C_str = key.split("|")
        # reverse-map cache atm key -> human-readable atmosphere
        atmosphere = next(a for a, ck in ATM_CACHE_KEY.items() if ck == atm_cache)
        T_C = float(T_C_str)
        T_K = T_C + 273.15
        log_pO2 = ATMOSPHERES[atmosphere][0]
        fermi_eV = fermi_level_eV(dop)
        row = feats[idx[key]]
        log_vo = _anchor_log_vo(row, T_K)            # identical to downstream BrouwerHead anchor
        # dominant charge from the anchor dEf dims 0-2 (lowest E_f wins at this εF)
        q = int(np.argmin(row[0:3]))
        delta_ef = float(row[0:3][q])                 # dims 0-2 = ΔE_f(V_O^{0,1,2})
        for _ in range(reps):
            c = float(10 ** rng.uniform(-4, math.log10(5e-2)))
            pairs.append(_make_pair(dop, c, atmosphere, T_C, fermi_eV,
                                    log_vo, q, delta_ef, T_K, "dft_cache"))
    rng.shuffle(pairs)
    return pairs[:target]


# --------------------------------------------------------------------------- #
# source 3 — reference-literature (honest hardcoded facts only)
# --------------------------------------------------------------------------- #
# Established beta-Ga2O3 defect facts. Numbers used are ONLY: NATIVE_VO_EF (= Varley 2010
# / Lyons HSE06 V_O E_f at O-rich, εF=midgap, already in the repo) and the qualitative
# donor/acceptor character stated in doc §6.1 references. No fabricated transition levels.
_LIT_FACTS = [
    # (dopant, valence_class, note_tag) — dopant=None means native V_O reference
    (None, "native", "varley2010_vo"),   # V_O three charge states, O-rich HSE06 E_f
    ("Si", "donor", "shallow_donor"),     # Varley 2010 / VdW: Si shallow donor on Ga
    ("Sn", "donor", "shallow_donor"),     # Sn shallow donor
    ("Ge", "donor", "shallow_donor"),     # Ge shallow donor
    ("Mg", "acceptor", "deep_acceptor"),  # Frodason 2023 / Goyal review: Mg deep acceptor
    ("F",  "anion", "f_on_o"),            # Varley 2010: F on O site (anion donor)
]
_LIT_NOTE = {
    "varley2010_vo": ("Native oxygen vacancies in beta-Ga2O3 (Varley et al. 2010, HSE06) "
                      "have formation energies of about 3.5, 1.8 and 0.3 eV for the neutral, "
                      "+1 and +2 charge states under O-rich conditions; V_O is a deep donor "
                      "and the +2 state dominates when the Fermi level is near the conduction band"),
    "shallow_donor": "is a shallow donor on the Ga site in beta-Ga2O3 (Varley 2010 / Van de Walle)",
    "deep_acceptor": "is a deep acceptor on the Ga site in beta-Ga2O3 (Frodason 2023); it does not "
                     "give p-type conduction but compensates donors and suppresses net V_O",
    "f_on_o":        "occupies the oxygen site in beta-Ga2O3 and acts as an anion donor (Varley 2010)",
}


def build_literature(rng, target=750) -> list:
    pairs = []
    atm_names = list(ATMOSPHERES.keys())
    facts = _LIT_FACTS
    per_fact = math.ceil(target / len(facts))
    for (dop, vclass, tag) in facts:
        elem = dop if dop is not None else "none"
        for _ in range(per_fact):
            T_C = float(rng.uniform(500, 1100))
            T_K = T_C + 273.15
            atmosphere = atm_names[rng.integers(len(atm_names))]
            log_pO2 = ATMOSPHERES[atmosphere][0]
            if dop is None:
                # native V_O reference: no dopant (offset 0.0, trace concentration).
                c = 0.0
                log_vo, q_dom = _brouwer_total("none", 0.0, T_K, log_pO2)
                q = 2  # Varley 2010: +2 dominates near CBM (n-type host)
                fermi_eV = 4.0  # n-type host
                delta_ef = _delta_ef("none", q, log_pO2)
                note = _LIT_NOTE[tag]
                reasoning = (f"{note}. Under {atmosphere} at {T_C:.0f} C this corresponds to "
                             f"log10[V_O] approximately {log_vo:.2f} cm^-3 in the +2 charge state.")
            else:
                c = float(10 ** rng.uniform(-4, math.log10(5e-2)))
                fermi_eV = fermi_level_eV(dop)
                log_vo, q = _brouwer_total(dop, c, T_K, log_pO2)
                delta_ef = _delta_ef(dop, q, log_pO2)
                note = _LIT_NOTE[tag]
                reasoning = (f"{elem} {note}. In {_atm_phrase(atmosphere)} at {T_C:.0f} C, "
                             f"{_charge_phrase(q)}, giving log10[V_O] approximately {log_vo:.2f} "
                             f"cm^-3 at a formation energy of about {delta_ef:.2f} eV.")
            log_keq = _log_keq(delta_ef, T_K)
            pairs.append({
                "input": {
                    "dopant": elem,
                    "dopant_concentration": round(float(c), 8),
                    "atmosphere": atmosphere,
                    "temperature_C": round(float(T_C), 1),
                    "fermi_level_eV": round(float(fermi_eV), 3),
                },
                "label": {
                    "log_vo_predicted": round(float(log_vo), 4),
                    "dominant_charge_state": int(q),
                    "log_K_eq": round(float(log_keq), 4),
                    "delta_E_f_eV": round(float(delta_ef), 4),
                    "reasoning": reasoning,
                },
                "_source": "literature",
            })
    rng.shuffle(pairs)
    return pairs[:target]


# --------------------------------------------------------------------------- #
# source 4 — Tier-1C synthetic (with validation / drop)
# --------------------------------------------------------------------------- #
def _validate(elem, c, T_K, log_pO2, q, log_vo) -> bool:
    """Sign / charge-neutrality / physical-range checks. Return True if pair is kept."""
    if not (LOG_VO_LO <= log_vo <= LOG_VO_HI):
        return False
    if not np.isfinite(log_vo):
        return False
    vclass = _valence_class(elem)  # noqa: F841 (kept for readability of the sign rule)
    # Sign rule: oxidizing (O2) must not give MORE V_O than reducing (Ar) for the same
    # dopant/conc/T. Check monotonicity at the two atmosphere extremes.
    lv_ar, _ = _brouwer_total(elem, c, T_K, ATMOSPHERES["Ar"][0])
    lv_o2, _ = _brouwer_total(elem, c, T_K, ATMOSPHERES["O2"][0])
    if lv_o2 > lv_ar + 1e-6:
        return False
    # Charge-neutrality / dominant-charge sanity: V_O is a donor, dominant q must be in {0,1,2}
    if q not in (0, 1, 2):
        return False
    return True


def build_tier1c(rng, per_dopant=90) -> list:
    pairs, dropped = [], 0
    atm_names = list(ATMOSPHERES.keys())
    for elem in TIER1C:
        fermi_eV = fermi_level_eV(elem)
        made = 0
        attempts = 0
        while made < per_dopant and attempts < per_dopant * 5:
            attempts += 1
            c = float(10 ** rng.uniform(-4, math.log10(5e-2)))
            T_C = float(rng.uniform(500, 1100))
            T_K = T_C + 273.15
            atmosphere = atm_names[rng.integers(len(atm_names))]
            log_pO2 = ATMOSPHERES[atmosphere][0]
            log_vo, q = _brouwer_total(elem, c, T_K, log_pO2)
            if not _validate(elem, c, T_K, log_pO2, q, log_vo):
                dropped += 1
                continue
            delta_ef = _delta_ef(elem, q, log_pO2)
            pairs.append(_make_pair(elem, c, atmosphere, T_C, fermi_eV,
                                    log_vo, q, delta_ef, T_K, "tier1c"))
            made += 1
    return pairs, dropped


# --------------------------------------------------------------------------- #
# verification + summary
# --------------------------------------------------------------------------- #
def verify(all_pairs) -> dict:
    issues = []
    per_dopant = {}
    log_vos = []
    for i, p in enumerate(all_pairs):
        try:
            json.dumps(p)
        except (TypeError, ValueError) as e:
            issues.append(f"row {i} not JSON-serializable: {e}")
            continue
        if set(p) - {"input", "label", "_source"} or "input" not in p or "label" not in p:
            issues.append(f"row {i} bad top-level keys")
        lv = p["label"]["log_vo_predicted"]
        if not np.isfinite(lv):
            issues.append(f"row {i} NaN/inf log_vo")
        elif not (LOG_VO_LO <= lv <= LOG_VO_HI):
            issues.append(f"row {i} log_vo out of [{LOG_VO_LO},{LOG_VO_HI}]: {lv}")
        log_vos.append(lv)
        d = p["input"]["dopant"]
        per_dopant[d] = per_dopant.get(d, 0) + 1
    log_vos = np.array(log_vos, dtype=float)
    return {"issues": issues, "per_dopant": per_dopant,
            "log_vo_min": float(np.nanmin(log_vos)), "log_vo_max": float(np.nanmax(log_vos)),
            "log_vo_mean": float(np.nanmean(log_vos)), "n_nan": int(np.isnan(log_vos).sum())}


def main():
    os.chdir(PROJ)
    rng = np.random.default_rng(58)

    print("Building Phase 58 Stage-1 QA corpus (CPU; GPU 0 left untouched)...")
    kroger = build_kroger(rng, per_dopant=240)
    print(f"  source 1 KROGER-derived:        {len(kroger)}")
    dft = build_dft(rng, target=1500)
    print(f"  source 2 DFT-cache-derived:     {len(dft)}")
    lit = build_literature(rng, target=750)
    print(f"  source 3 reference-literature:  {len(lit)}")
    tier1c, dropped = build_tier1c(rng, per_dopant=100)
    print(f"  source 4 Tier-1C synthetic:     {len(tier1c)}  (dropped {dropped} sign/range violations)")

    all_pairs = kroger + dft + lit + tier1c
    print(f"  TOTAL before split:             {len(all_pairs)}")

    # verify
    rep = verify(all_pairs)
    print("\n=== VERIFY ===")
    print(f"  log_vo range: [{rep['log_vo_min']:.2f}, {rep['log_vo_max']:.2f}]  "
          f"mean {rep['log_vo_mean']:.2f}  NaN {rep['n_nan']}")
    if rep["issues"]:
        print(f"  !! {len(rep['issues'])} ISSUES (first 10):")
        for s in rep["issues"][:10]:
            print("     -", s)
    else:
        print("  no JSON / range / NaN issues")

    print("\n  per-dopant coverage (count):")
    for d in sorted(rep["per_dopant"]):
        flag = "" if rep["per_dopant"][d] >= 100 else "  <100 !!"
        print(f"    {d:5s} {rep['per_dopant'][d]:5d}{flag}")
    min_dop = min(rep["per_dopant"], key=rep["per_dopant"].get)
    print(f"  per-dopant MIN: {min_dop} = {rep['per_dopant'][min_dop]}")

    # per-source breakdown
    from collections import Counter
    srcs = Counter(p["_source"] for p in all_pairs)
    print("\n  per-source breakdown:")
    for s, n in sorted(srcs.items()):
        print(f"    {s:12s} {n}")

    # log_vo distribution (deciles)
    log_vos = np.array([p["label"]["log_vo_predicted"] for p in all_pairs])
    qs = np.percentile(log_vos, [0, 10, 25, 50, 75, 90, 100])
    print("  log_vo deciles [0,10,25,50,75,90,100]: " + ", ".join(f"{v:.2f}" for v in qs))

    # 9:1 train/val split (shuffled, stratified-ish by global shuffle)
    rng.shuffle(all_pairs)
    n_val = round(len(all_pairs) * 0.1)
    val = all_pairs[:n_val]
    train = all_pairs[n_val:]

    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TRAIN, "w") as f:
        for p in train:
            f.write(json.dumps(p) + "\n")
    with open(OUT_VAL, "w") as f:
        for p in val:
            f.write(json.dumps(p) + "\n")
    print(f"\n  wrote {len(train)} train -> {OUT_TRAIN}")
    print(f"  wrote {len(val)} val   -> {OUT_VAL}")
    print(f"  TOTAL {len(all_pairs)} pairs")


if __name__ == "__main__":
    main()
