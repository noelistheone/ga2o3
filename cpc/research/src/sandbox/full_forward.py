"""full_forward — the integrated (dopant, process) → nine-property forward simulator.

Wires Tier-1 (charge-neutrality) + transport + optical + device layers into one call that emits
all nine corpus properties for a doped β-Ga2O3 film, each tagged with an honest fidelity grade.
Integration alone does not establish novelty or predictive accuracy. Current evidence and
withdrawn historical claims are recorded under paper/npj/revision_20261015/.

Pipeline: solve equilibrium at T_anneal(process) with fixed [dopant] → quench to 300 K →
n, p, [V_O], ionized densities → μ(N_I), σ=e·n·μ → Burstein-Moss Eg/dEg → device FOM directions.
The undoped reference is solved once at matched process for the within-paper dEg differential.
"""
from __future__ import annotations

import numpy as np

from . import engine, transport, optical, device, thermo
from .kroger_db import DefectDB, N_SITE, load


# --------------------------------------------------------------------------------------
# Cation (Ga) site density — THE single definition used to turn conc_frac into cm^-3.
#
# Derivation.  kroger_db.N_SITE = 1.91e22 cm^-3 is the density of ONE inequivalent lattice
# site type, not of all Ga sites: β-Ga2O3 has 2 inequivalent Ga sites and 3 inequivalent O
# sites, and KROGER's own comment ("all 9 KROGER site types identical") is precisely the
# statement that every one of them carries the same 1.91e22.  Therefore
#     N_cat = 2 * N_SITE = 3.82e22 cm^-3      (Ga sites)
#     N_an  = 3 * N_SITE = 5.73e22 cm^-3      (O sites)
# Independent check against this repo's own perfect DFT cell dft/qe_cc/gate2_perfect.in
# (32 Ga + 48 O in V = 845.87 A^3):
#     N_Ga = 32/845.87 * 1e24 = 3.7831e22 cm^-3   (2*N_SITE agrees to 1.0%; the residual is
#                                                  the PBE volume overestimate)
#     N_O  = 48/845.87 * 1e24 = 5.6746e22 cm^-3
#     N_Ga/2 = N_O/3 = 1.8915e22 cm^-3           <-- reproduces N_SITE, confirming the
#                                                    per-site-type reading
# 2*N_SITE is adopted (rather than the DFT 3.7831e22) because it is the value the defect
# database's own site prefactors are built from, so loading and defect densities stay on one
# consistent site convention.
#
# BUG FIXED 2026-07-28: the previous code used N_SITE * 2/3 = 1.2733e22, i.e. it applied the
# Ga:O stoichiometric ratio 2/3 to a quantity that was already a per-site-type density.  That
# understated N_cat by a factor 2*N_SITE / (N_SITE*2/3) = 3.0 exactly (2.971 vs the DFT value),
# so every historical call with conc_frac=x actually loaded x/2.971 cation fraction.  See
# results/accuracy_baseline/stage1_ncat_migration.md.
# --------------------------------------------------------------------------------------
N_CAT_SITE = 2.0 * N_SITE          # Ga cation site density [cm^-3] = 3.82e22
N_AN_SITE = 3.0 * N_SITE           # O anion site density  [cm^-3] = 5.73e22

# ONE definition string, emitted by forward() and by codoping.forward_codoped alike.
VO_DEFINITION = ("headline V_O = total oxygen deficiency Sum(-dm_O)*N_cs over all "
                 "O-deficient charge states; V_O_native_* = isolated native V_O only")


# ======================================================================================
# STAGE-3 DESIGN 1 — COMPENSATION-MARGIN ABSTENTION
# pre-registration: results/accuracy_baseline/stage3_prereg_abstention.md
#
# n is the residual of two nearly equal charge sums.  With
#     P = Sum_{q>0} q*N_cs + p + Nd_bg          (all positive charge)
#     M = Sum_{q<0} |q|*N_cs + Na_bg            (all negative charge EXCEPT free electrons)
# quench neutrality is P - M - n = 0, so  m := (P-M)/max(P,M) = n/max(P,M) EXACTLY, and 1/m is
# exactly the condition number of the subtraction that produces n: a relative error eps on either
# side gives Delta n / n = eps/m, and the SIGN of the residual flips once eps > |m|.
#
# THRESHOLD (derived, never swept against a gate — stage3_threshold_derivation.json):
# m*(T_a) is the MEASURED displacement, in charge-budget units, that a +/-sigma = 0.5 eV shift of
# the whole acceptor block (all 346 of 873 charge states with q<0, a RELATIVE acceptor-vs-donor
# shift because only that difference enters the balance) causes in the re-solved residual.
# sigma = 0.500 eV is the stated 1-sigma of the KROGER HSE enthalpies the engine consumes
# (docs/absolute_accuracy_path.md L44, docs/thesis_finding.md L29, docs/feasibility_research.md
# L28).  The textbook frozen-E_F bound m*_frozen = 1 - exp(-sigma/kT_a) = 0.9955 (1073 K) is
# honest but useless; the solver re-solves E_F, which damps the response by ~3 orders at 1073 K
# and ~1 order at 1950 K.  That damping was MEASURED, not assumed.
# ======================================================================================
SIGMA_DHO_EV = 0.500                 # headline 1-sigma on the KROGER HSE formation enthalpies
# ROUND 13a — THE SHIPPED sigma IS 10-40x TOO PERMISSIVE, and this is recorded rather than
# silently fixed. sigma = 0.500 eV below was PRE-REGISTERED from docs (absolute_accuracy_path.md
# L44, thesis_finding.md L29, feasibility_research.md L28). Round 12 then MEASURED the real
# inter-setup systematic on Ga-containing formation energies at ~1.3 eV, by three independent
# routes (docs/RESEARCH_LOG.md sec.12e): the Ga elemental-reference disagreement 1.329 eV; our HSE
# V_O q0 3.674 eV vs KROGER's 4.47/4.99/5.20 (0.80-1.53 eV, i.e. NOT COMMENSURATE); and the
# identity A-B = -(dHf^DFT - dHf^exp)/2 per Ga atom with dHf^HSE -8.005 vs dHf^exp -11.170 eV/f.u.
# Re-deriving the threshold at the measured sigma
# (results/accuracy_baseline/round13_threshold_measured_sigma.json) gives:
#     T_a:      1073      1350      1500      1950   K
#     sigma=0.5 9.39e-4  6.61e-2   1.33e-1   2.85e-1     <- shipped, M_STAR_TABLE below
#     sigma=1.3 3.62e-2  3.72e-1   6.71e-1   6.41e-1     <- MEASURED
# At the measured sigma even Cr (margin 0.168) abstains at 1350 K and no doped point reaches
# m* at 1500 K, i.e. the quenched n is not identified for ANY doped condition >= 1350 K. That is
# self-consistent with Round 9: the conditioned rule then always reports the anneal stage in the
# doped regime, which is the operation that took max|err| from 58.594 to 12.172 dex.
# sigma=0.5 is NOT replaced here, because substituting it would hide the discrepancy. Callers that
# want the measured-uncertainty behaviour pass
#     m_star(T_anneal, table=M_STAR_TABLE_MEASURED_SIGMA_1P3)
# (m_star's second argument is the TABLE, not a sigma).
M_STAR_TABLE_MEASURED_SIGMA_1P3 = {1073.0: 3.623e-2, 1350.0: 3.717e-1,
                                   1500.0: 6.712e-1, 1950.0: 6.405e-1}

M_STAR_TABLE = {                     # T_anneal [K] -> m*   (sigma = 0.500 eV, median estimator)
    1073.0: 3.681e-3,
    1350.0: 7.994e-2,
    1500.0: 1.756e-1,
    1950.0: 2.847e-1,
}
# Worst charge-closure residue measured across the KROGER-19 (Mg, 2.0e-10 of the budget;
# stage3_margin_sensitivity.json identity_check).  Below this the digits of n are brentq noise,
# independent of any energetics question.  Rounded DOWN a decade -> always-on numerics flag.
MARGIN_NUMERICS_FLOOR = 1e-9

IDENTIFIABILITY_BAR_NOTE = ("1-sigma energetics identifiability only; NOT calibrated coverage "
                            "(measured 2/10 at 1073 K)")

# ======================================================================================
# STAGE-3 DESIGN 3 — ACCEPTOR-FLOOR HONESTY (representability + detectability)
# pre-registration: results/accuracy_baseline/stage3_prereg_acceptor_pdr.md, PART A
#
# N_REP: a standard van-der-Pauw Hall coupon 5 mm x 5 mm x 300 nm has V = 7.5e-6 cm^3, so ONE
#        carrier in the whole coupon is 1.333e5 cm^-3.  Rounded DOWN to the decade (rounding down
#        makes the flag fire strictly less often) -> 1e5 cm^-3.
# N_DET: the smallest carrier density any paper in the transport corpus reports is 5.0e12 cm^-3
#        (PECVD Zn, mu 0.13).  Rounded to the decade -> 1e13; independently this is the stated
#        Hall metrology bound for this material.
# Neither threshold is swept and neither is allowed to CLIP anything: hall_n_cm3 / hall_p_cm3 /
# sigma_S_cm are always the raw solver values.  The flags are reported alongside.
# ======================================================================================
N_REP_CM3 = 1e5                      # representability floor (A1)
N_DET_CM3 = 1e13                     # detection / comparability floor (A2)
COUPON_VOLUME_CM3 = 7.5e-6           # 5 mm x 5 mm x 300 nm
UNPHYSICAL_REASON = (f"max(n,p) < {N_REP_CM3:.0e} cm^-3 = less than one carrier in a standard "
                     f"van-der-Pauw coupon (5 mm x 5 mm x 300 nm = {COUPON_VOLUME_CM3:.1e} cm^3); "
                     "the number is an arithmetic output, not a statement about a sample")

# ======================================================================================
# STAGE-3 DESIGN 4 — SOURCED PER-DOPANT INCORPORATION CEILING (Sn only)
# pre-registration: results/accuracy_baseline/stage3_prereg_sn.md
#
# The chain must not silently answer a question no experiment can pose.  The highest MEASURED
# incorporated [Sn] anywhere in the corpus is a SIMS saturation at 2e20 cm^-3 = 0.524 at%; the
# battery's 1 at% test point is ~2x above it.  Requests above a sourced ceiling are CLAMPED to
# the ceiling and flagged.  Every other element has no entry => no clamp, no flag, bit-identical
# behaviour.  This is deliberately the LOOSEST defensible bound (largest measured incorporation),
# so the clamp can only ever be too permissive; the KROGER paper's own equilibrium SnO2-phase
# solubility is <4.5e18 cm^-3 below ~1850 K, three orders lower.
#
# NOTE ON LOCATION: stage3_prereg_sn.md §3 asks for this table as
# data/reference/dopant_incorporation_ceiling.json.  data/** is write-protected for this task, so
# the table ships here as a module constant with the identical provenance fields.  Nothing about
# the numbers or their sourcing changed.
# ======================================================================================
INCORPORATION_CEILING = {
    "Sn": {
        "C_max_cm3": 2.0e20,
        "doi": "10.1063/5.0027870",
        "method": "SIMS incorporation saturation",
        "quote": ("doping concentrations were measured by SIMS ... the doping concentration "
                  "saturated at about 2x10^20 cm^-3 at higher Sn cell temperatures"),
        "reference": ("Mauze, Zhang, Itoh, Ahmadi, Speck, Appl. Phys. Lett. 117, 222102 (2020)"),
        "local_copy": "Ga2O3-Net/extra-paper/v56_markdown/3f2633329692e3d4.md:138-141",
    },
}


def m_star(T_anneal: float, table: dict = None) -> tuple:
    """m*(T_anneal) for sigma = 0.5 eV: linear interpolation in log10 m* vs T_a between the four
    tabulated anneal temperatures, held FLAT outside [1073, 1950] K.

    Returns (m_star, extrapolated).  Extrapolation is refused, not guessed: outside the calibrated
    range the nearest endpoint is returned and `extrapolated` is True so the caller can see it.
    """
    tab = table or M_STAR_TABLE
    Ts = sorted(tab)
    T = float(T_anneal)
    if T <= Ts[0]:
        return float(tab[Ts[0]]), bool(T < Ts[0])
    if T >= Ts[-1]:
        return float(tab[Ts[-1]]), bool(T > Ts[-1])
    for a, b in zip(Ts[:-1], Ts[1:]):
        if a <= T <= b:
            f = (T - a) / (b - a)
            lg = (1.0 - f) * np.log10(tab[a]) + f * np.log10(tab[b])
            return float(10.0 ** lg), False
    return float(tab[Ts[-1]]), True                      # unreachable


def compensation_margin(db, q, Nd_bg: float = 0.0, Na_bg: float = 0.0):
    """(m, P, M) from an already-solved quench EqResult.  Two dot products, ZERO extra solves.

    P = Sum_{q>0} q*N_cs + p + Nd_bg ;  M = Sum_{q<0} |q|*N_cs + Na_bg ;  m = (P-M)/max(P,M).
    """
    Q = db.charge
    P = float((np.clip(Q, 0, None) * q.N_cs).sum()) + float(q.p) + float(Nd_bg)
    M = float((np.clip(-Q, 0, None) * q.N_cs).sum()) + float(Na_bg)
    denom = max(P, M)
    m = (P - M) / denom if denom > 0.0 else 0.0
    return float(m), P, M


def compensation_block(db, q, T_anneal: float, Nd_bg: float = 0.0, Na_bg: float = 0.0,
                       n_anneal: float = None) -> dict:
    """The full NON-LOSSY abstention block.  Suppresses nothing; adds fields only."""
    m, P, M = compensation_margin(db, q, Nd_bg=Nd_bg, Na_bg=Na_bg)
    ms, extrap = m_star(T_anneal)
    am = abs(m)
    abstain = bool(am < ms)
    numerics_floor = bool(am < MARGIN_NUMERICS_FLOOR)
    # identifiability bar: how far a 1-sigma energetics error can move log10 n.  When the
    # displacement exceeds the margin the residual can be driven through zero, so the LOWER edge
    # is unbounded.  This is NOT a confidence interval (measured coverage 2/10 at 1073 K).
    if am > 0.0:
        half = float(np.log10(1.0 + ms / am))
    else:
        half = float("inf")
    bar = [float("-inf"), half] if abstain else [-half, half]
    if abstain:
        reason = (f"ABSTAIN: |margin| {am:.4e} < m* {ms:.4e} (T_a={T_anneal:g} K, "
                  f"sigma_dHo={SIGMA_DHO_EV:g} eV). n = {float(q.n):.4e} is the residual of "
                  f"P = {P:.4e} and M = {M:.4e} cm^-3; a 1-sigma energetics error displaces that "
                  f"residual by more than the residual itself, so the ABSOLUTE VALUE of n is not "
                  f"identified. Still claimed: the SIGN of the doping response vs the matched "
                  f"undoped reference, the anneal-stage carrier density, and the direction of the "
                  f"pO2 / concentration derivatives.")
    else:
        reason = (f"identified: |margin| {am:.4e} >= m* {ms:.4e} (T_a={T_anneal:g} K, "
                  f"sigma_dHo={SIGMA_DHO_EV:g} eV)")
    if extrap:
        reason += (f" [EXTRAPOLATED: m* held flat outside the calibrated range "
                   f"{min(M_STAR_TABLE):g}-{max(M_STAR_TABLE):g} K]")
    return {
        "compensation_margin": m,
        "compensation_margin_terms": {
            "P_cm3": P, "M_cm3": M, "denominator": "max(P,M)",
            "m_star": ms, "m_star_extrapolated": bool(extrap),
            "sigma_dHo_eV": SIGMA_DHO_EV, "T_anneal_K": float(T_anneal),
            # closure audit: quench neutrality says P - M - n = 0 exactly.  What is left is the
            # solver's own root residue (plus the self-trapped-hole terms, which the
            # pre-registered P/M definition deliberately omits).
            "closure_residual_cm3": float(P - M - float(q.n)),
            "closure_residual_rel": float((P - M - float(q.n)) / max(P, M, 1.0)),
        },
        "compensation_abstain": abstain,
        "compensation_abstain_reason": reason,
        "identifiability_bar_dex": bar,
        "identifiability_bar_note": IDENTIFIABILITY_BAR_NOTE,
        "compensation_numerics_floor": numerics_floor,
        "n_anneal_cm3": float(n_anneal) if n_anneal is not None else None,
    }


def recommended_carrier_block(n_quench: float, n_anneal: float, abstain: bool) -> dict:
    """ROUND 9 — conditioned, NON-LOSSY recommendation of WHICH carrier density to compare against
    a measurement.  Adds fields only; hall_n_cm3 stays the raw quenched solver value.

    This is a MEASURED rule, not an assertion.  Basis, at the frozen T1 process point
    (T_a = 1073 K, pO2 1e-5, 1 at%, Nd_bg 1e17) against the corpus median per dopant over the 10
    KROGER dopants that have transport rows (results/mathwork/anneal_vs_quench.json,
    scripts/probe_math_anneal_vs_quench.py):

      * pooled mean |log10 error|   18.713 -> 3.701 dex   (anneal beats quench by 15.013)
      * pooled median |log10 error|  2.094 -> 1.784 dex
      * Ta (the one testable compensated element)  13.63 -> 0.54 dex
      * WORST single-dopant degradation           +0.000 dex

    The worst degradation is essentially zero because for every well-conditioned dopant the two
    stages agree to far below any measurement noise.  MEASURED relative difference |n_a/n_q - 1| at
    1073 K / pO2 1e-5 / 1 at%: Si 2.6e-5, Sn 4.6e-6, Ge/Hf/Zr 3.4e-6, Cr 2.1e-4, undoped 3.1e-7 —
    i.e. <= 1e-4 dex, about 3000x below the 0.3 dex corpus noise.  (They are NOT bit-identical; an
    earlier version of this note said so and was wrong.)  Contrast Ta: 1.2e13 relative, +13.09 dex.
    Because the well-conditioned cases coincide, switching on the abstention flag introduces no
    practical discontinuity, which is why this can be a hard rule rather than a blend.

    NOTE ON CONSERVATISM.  The abstention flag is deliberately conservative: at T_a = 1073 K,
    m* = 3.681e-3 while the well-conditioned donors sit at margins 1e-4 to 7e-4, so 9 of 10 corpus
    dopants abstain and the rule reduces to "always report the anneal stage" at that temperature
    (Stage 3 measured specificity 0.20).  That over-flagging is HARMLESS HERE precisely because the
    over-flagged cases are the ones where the two stages coincide to 1e-4 dex.  It would not be
    harmless for a rule that suppressed output, which is why this block suppresses nothing.

    HONEST COST, recorded here so it travels with the code: the cross-dopant RANKING gets worse,
    Spearman rho 0.350 -> 0.092.  Both are statistically insignificant (p 0.322 and 0.800; the
    frozen gate's own reference is rho 0.255), so the trade is 13 dex of absolute accuracy on the
    one testable compensated element against a ranking that was never demonstrated.  It is a trade,
    not a free win.  The +15 dex mean improvement is driven ENTIRELY by the four collapsed dopants
    (Ta, H, Mg, Zn); Mg and Zn remain wrong either way (8.93 and 12.17 dex) — the anneal stage makes
    them non-absurd, not correct.  The acceptor problem is untouched by this rule.

    WHY the anneal stage is the well-conditioned one (Rounds 8-10 of docs/RESEARCH_LOG.md): the
    quenched n is the residual of two ~1e19-1e20 charge sums, so in the compensated regime it is not
    a well-posed prediction target at all — a correct 0.08 eV symmetry-derived degeneracy correction
    moves it 12.5 dex, 50-digit arithmetic moves it 0.000 dex (so it is a parameter-sensitivity, not
    an arithmetic, limit), the corpus cannot determine the parameters, and the HSE VBM needed to
    pin them carries 3.7 eV of uncertainty on dHo(V_Ga^-3).  The anneal-stage n is not a difference
    of large numbers and is therefore stable.
    """
    stage = "anneal" if abstain else "quench"
    val = float(n_anneal) if abstain else float(n_quench)
    return {
        "recommended_n_cm3": val,
        "recommended_n_stage": stage,
        "recommended_n_grade": (
            "compensated regime: quenched n is NOT identified; the anneal-stage n is the "
            "comparable quantity. Measured basis: pooled mean |err| 18.713 -> 3.701 dex, Ta "
            "13.63 -> 0.54 dex, worst degradation 0.000 dex (results/mathwork/"
            "anneal_vs_quench.json). Ranking skill is NOT improved (rho 0.350 -> 0.092, both "
            "insignificant). Acceptors (Mg/Zn) remain wrong by 9-12 dex either way."
            if abstain else
            "well-conditioned regime: the quenched n is identified and equals the anneal-stage n "
            "to machine precision, so this rule is a no-op here."
        ),
        # Does the recommendation actually differ from the raw quenched value by an amount that
        # could matter?  Tolerance 1e-3 dex, chosen as 300x below the 0.3 dex corpus noise rather
        # than as a float-equality test (the two stages are NOT bit-identical; see the docstring).
        "recommended_n_agrees_with_quench": bool(
            n_quench > 0 and n_anneal > 0
            and abs(np.log10(n_anneal) - np.log10(n_quench)) < 1e-3),
        "recommended_vs_quench_dex": (
            float(np.log10(max(n_anneal, 1e-300)) - np.log10(max(n_quench, 1e-300)))),
        "recommended_n_basis": "results/mathwork/anneal_vs_quench.json (Round 9); rule switches on "
                               "compensation_abstain, which is derived from sigma_dHo, not fitted.",
    }


def stage_consistent_block(rec_n: float, n_quench: float, mu_drift: float, n_und_rec: float,
                           quench_vals: dict) -> dict:
    """ROUND 15 — make the DERIVED properties consistent with whichever carrier density is reported.

    Round 9 made the carrier density conditional but left the other eight properties on the quench
    stage. Measured consequence (results/mathwork/round15_stage_consistency.json,
    scripts/round15_stage_consistency.py): for the two elements that abstain, sigma computed from
    the quenched n is wrong by **16.32 dex (W) and 14.46 dex (Nb)** against the reported carrier
    density — W would be printed as n = 2.565e18 cm^-3 (a conductor) beside sigma = 4.887e-16 S/cm
    (an insulator). A self-contradictory datasheet is exactly the class of error that made v1 wrong.

    The fix follows the CONDITIONING, not convenience, and "recompute everything at the anneal
    temperature" is explicitly NOT it — mu and sigma are ROOM-TEMPERATURE observables and a mobility
    at 1350 K is not what a Hall measurement returns. Three groups:

      (a) FUNCTIONS OF n  ->  must follow n:  sigma = e*n*mu_drift, Eg (Burstein-Moss), dEg.
      (b) FUNCTIONS OF SUMS  ->  already well conditioned, UNCHANGED:  mu(N_I), V_O.
          N_I = sum |q| N_cs is a SUM, not a difference; the quench freezes the anneal per-defect
          totals, so N_I at 300 K is determined even where n is not. This is the same structural
          fact the rank analysis found when mobility added ZERO identifiable rank.
      (c) FUNCTIONS OF E_F  ->  inherit the ill-conditioning:  dark activation, PDR, tau.
          These are NOT recomputed. They stay on the quench stage and are labelled, because E_F in
          the compensated regime is exactly as unidentified as n is.

    NON-LOSSY as always: every quench-stage value is returned unchanged by the caller; this block
    only ADDS the stage-consistent companions and the labels.
    """
    from . import optical as _optical
    changed = bool(rec_n > 0 and n_quench > 0
                   and abs(np.log10(rec_n) - np.log10(n_quench)) >= 1e-3)
    sigma_rec = float(transport.conductivity(rec_n, mu_drift))
    eg_rec = float(_optical.optical_eg(rec_n))
    deg_rec = _optical.dEg(rec_n, max(float(n_und_rec), 1.0))
    return {
        "stage_consistent": {
            "sigma_S_cm": sigma_rec,
            "Eg_optical_eV": eg_rec,
            "dEg_eV": deg_rec["dEg"],
            "hall_mu_cm2Vs_unchanged": float(mu_drift),
            "differs_from_quench": changed,
            "sigma_shift_dex": (float(np.log10(max(sigma_rec, 1e-300))
                                      - np.log10(max(quench_vals.get("sigma_S_cm", 1e-300), 1e-300)))),
        },
        "stage_consistent_note": (
            "sigma / Eg / dEg here are recomputed from recommended_n_cm3 so they are consistent "
            "with the carrier density actually being reported; mu and V_O are functions of SUMS "
            "over the frozen defect populations and are unchanged; dark_activation_eV, PDR_score "
            "and tau_score are functions of E_F, are NOT recomputed, and inherit exactly the same "
            "ill-conditioning as the quenched n. Do not mix a stage-consistent sigma with a "
            "quench-stage PDR in the same claim."),
        "EF_dependent_stay_quench": ["dark_activation_eV", "PDR_score", "tau_score"],
    }


def carrier_regime_block(n: float, p: float, sigma: float) -> dict:
    """A1/A2 honesty contract.  RAW n/p/sigma are returned unchanged by the caller; this block
    only says which of them, if either, corresponds to something an instrument could see."""
    mx = max(float(n), float(p))
    dominant = "n" if float(n) >= float(p) else "p"
    representable = bool(mx >= N_REP_CM3)
    if mx < N_DET_CM3:
        regime = "below_detection"
    else:
        regime = "electron_measurable" if dominant == "n" else "hole_measurable"
    return {
        "dominant_carrier": dominant,
        "carrier_regime": regime,
        "n_detection_floor_cm3": N_DET_CM3,
        "carrier_representable": representable,
        "unphysical_carrier_density": bool(not representable),
        "unphysical_carrier_density_reason": (UNPHYSICAL_REASON if not representable else None),
        "sigma_regime": regime,
        "carrier_regime_note": ("below_detection means the prediction is LEFT-CENSORED: the "
                                "correct comparison operator against a measurement is '<', not "
                                "'='. hall_n_cm3 / hall_p_cm3 / sigma_S_cm are always the raw "
                                "solver values and are never clipped."),
    }


def apply_incorporation_ceiling(fixed: dict) -> tuple:
    """Clamp requested loadings to their SOURCED measured incorporation ceiling (Sn only).

    Returns (clamped_fixed, info) where info carries the flag + the requested/ceiling numbers +
    the source DOI.  Elements with no sourced ceiling are untouched and unflagged.
    """
    out = dict(fixed)
    requested, ceilings, dois, fired = {}, {}, {}, {}
    for el, want in fixed.items():
        requested[el] = float(want)
        entry = INCORPORATION_CEILING.get(el)
        if entry is None:
            ceilings[el] = None
            dois[el] = None
            fired[el] = False
            continue
        cmax = float(entry["C_max_cm3"])
        ceilings[el] = cmax
        dois[el] = entry["doi"]
        if float(want) > cmax:
            out[el] = cmax
            fired[el] = True
        else:
            fired[el] = False
    info = {"above_reported_incorporation_ceiling": bool(any(fired.values())),
            "requested_dopant_cm3": requested, "ceiling_cm3": ceilings,
            "ceiling_source_doi": dois, "ceiling_fired_per_element": fired}
    return out, info


def dominant_compensating_family(db, r, top_k: int = 3) -> dict:
    """Report-only diagnostic (stage3_prereg_sn.md §3 rank 2): which charge states actually carry
    the compensating NEGATIVE charge, and how much of it is native vs dopant-bearing."""
    Q = db.charge
    w = np.clip(-Q, 0, None) * r.N_cs                    # |q|*N per negative state
    tot = float(w.sum())
    if tot <= 0:
        return {"total_negative_charge_cm3": 0.0, "top_negative_charge_states": []}
    order = np.argsort(-w)[:int(top_k)]
    ga, ox = db.col("Ga"), db.col("O")
    host = {ga, ox}
    top = []
    for i in order:
        dm = db.dm[i]
        parts = [f"{db.elements[j]}{int(dm[j]):+d}" for j in range(len(db.elements)) if dm[j] != 0]
        carries_dopant = any(dm[j] != 0 for j in range(len(db.elements)) if j not in host)
        top.append({"charge_state_index": int(i), "defect_id": int(db.cs_ID[i]),
                    "composition": " ".join(parts) or "(none)", "q": int(Q[i]),
                    "N_cm3": float(r.N_cs[i]), "charge_carried_cm3": float(w[i]),
                    "frac_of_negative_charge": float(w[i] / tot),
                    "dopant_bearing": bool(carries_dopant)})
    dopant_cols = [j for j in range(len(db.elements)) if j not in host]
    has_dop = (np.abs(db.dm[:, dopant_cols]).sum(axis=1) > 0) if dopant_cols else np.zeros(db.nCS,
                                                                                           bool)
    return {"total_negative_charge_cm3": tot,
            "frac_negative_charge_dopant_bearing": float(w[has_dop].sum() / tot),
            "top_negative_charge_states": top}


HISTORICAL_FIDELITY_PRE_REVIEW_20261015 = {
    # ROUND 21 REGRADE. The previous grade — "quantitative(bulk/epi ~2x); order-of-mag(sputtered
    # films)" — claimed 0.30 dex on bulk/epi and had never been measured against a floor. It is
    # now measured: results/acc_r21/certification_v2.json, 244 deduplicated rows over 88 source
    # documents, leave-one-document-out. Pooled, the model scores 1.049 dex against a CONSTANT
    # PREDICTOR's 1.034 — it loses to predicting the same number for every sample, so the pooled
    # carrier density is not a usable quantity at any stated accuracy.
    #
    # Two conditional modes DO clear the constant floor and a permutation null, and both are
    # single-document fragile (removing one paper puts them back under the floor), so they are
    # reported and not certified:
    #     solver does not abstain                     0.738 dex, 23% of rows, perm p 0.015
    #     that AND the paper reports a doping level   0.458 dex, 13% of rows, perm p 0.005
    # The shipped certification (results/hybrid/certification.json, 1.10 dex) predates the
    # Round 9-20 engine changes, groups 45 separate papers into one pseudo-lab, contains 69
    # duplicated rows, and compares against a floor computed on a different row set.
    #
    # ROUND 22 adds the REGIME boundary, which is sharper than the accuracy number. Within a
    # single paper -- where the lab confound cancels exactly and the prediction has no free
    # parameters -- measured log10 n rises with log10 c at an exponent of 0.94, and this model
    # predicts 0.26. "One dopant atom, one carrier" scores 0.134 dex on those contrasts against
    # the forward pass's 0.603. The cause is a redundant two-channel self-compensation pin
    # (Si-2V_Ga complexes, then isolated V_Ga at q=-3); removing either alone changes nothing,
    # removing both plus a 1.0 eV V_Ga shift reaches 0.136 dex AND breaks Gate-1 anchor A2.
    # Reproducing the measured activation needs an effective freeze-in T ~800 K above these
    # films' real growth temperature. All three say the same thing: an as-grown epitaxial film's
    # vacancy population is not thermodynamic, so this model class does not apply to it.
    # results/acc_r22/{within_doc,dose_exponent,complex_penalty,vga_penalty,gate1_under_shift}.json
    "hall_n": "NOT USABLE AS A NUMBER pooled (1.05 dex, loses to a constant predictor's 1.03). "
              "Conditional and single-document fragile: 0.74 dex where the solver does not "
              "abstain (23% of rows), 0.46 dex where it also has a reported doping level (13%). "
              "DOSE RESPONSE IS OUT OF REGIME for as-grown epitaxial films: predicted exponent "
              "0.26 against a measured 0.94, and 'n proportional to c' beats this model 4.5x on "
              "within-paper contrasts. Near-equilibrium material only. "
              "See results/acc_r21/certification_v2.json and results/acc_r22/.",
    "V_O": "trend/ordinal-proxy only (XPS O1s metrology disputed). Headline V_O = TOTAL "
           "oxygen deficiency (native V_O + all O-deficient dopant/native complexes), which "
           "is what XPS O1s lattice/deficient area-ratios actually integrate; the isolated "
           "native-vacancy count is kept alongside as V_O_native_cm3_*.",
    # ROUND 21 REGRADE. The old grade split on single-crystal vs films, which the data does not
    # support: the split that matters is POLYCRYSTALLINE DEPOSITS (sputter/PLD/evaporation/ALD)
    # against everything else, and homoepitaxial film mobility is predicted about as well as
    # bulk. Measured on 224 deduplicated rows over 77 documents, leave-one-document-out
    # (results/acc_r21/certification_v2.json):
    #     zero-anchor, one global offset            0.236 dex  — LOSES to a constant's 0.222
    #     zero-anchor, shrunk per-element offset    0.199 dex  — beats it; perm p 0.0025,
    #                                               random-group p 0.005, 0/77 drop-one failures
    #     given a MEASURED n (Hall or C-V)          0.246 dex  — floor 0.328, perm p 0.0025,
    #                                               0/40 drop-one failures, and this is AFTER
    #                                               removing all 8 documents containing a
    #                                               Caughey-Thomas fit anchor (without that
    #                                               removal it reads 0.166 dex, so ~40% of the
    #                                               apparent skill was in-sample)
    # These two are the only certified quantitative modes anywhere in the forward pass.
    "hall_mu": "cross-lab ~1.6x (0.199 dex) with the shrunk per-element offset; ~1.8x (0.246 "
               "dex) given a measured n. Polycrystalline deposits are ~1 dex and excluded from "
               "both. See results/acc_r21/certification_v2.json.",
    # ROUND 21. sigma = q*n*mu and its error decomposes with an n-share of 0.876 (Round 18), so
    # it inherits hall_n's grade rather than carrying one of its own. The old "~2-4x for bulk
    # Sn/Si" named a subset the corpus does not contain (Round 17 found no bulk Sn row).
    "sigma": "inherits hall_n (n-share of the sigma error is 0.876); no independent grade. The "
             "previous '~2-4x for bulk Sn/Si' named a subset absent from the corpus.",
    "Eg": "quantitative for degenerate-donor BM(~0.1 eV); trend otherwise",
    # ROUND 16 DOWNGRADE. The previous grade was "sign/ranking (headline); magnitude for
    # degenerate-donor subset" and it had NEVER been tested: corpus_trend_battery scores T1 (n
    # ranking), T2 (n sign) and T3 (PDR sign) and has no dEg test at all. The "headline" wording
    # borrowed credibility from the SIBLING ML project's dEg rho = 0.617, which is a supervised
    # ridge FITTED on this same optical corpus — a different model on a different task.
    # First test ever run: results/mathwork/round16_dEg_confront.json, 39 deduplicated
    # within-paper contrasts, 0.05 eV Tauc deadband, 20000 permutations.
    #
    # The decisive argument is DETERMINISTIC, not statistical, and it is the one to quote (the
    # permutation null was built at the contrast level while the chain's prediction is a
    # per-dopant constant, so that null is anti-conservative and the effective N is ~16, not 39):
    #     the chain predicts a POSITIVE dEg for 27-33 of the 39 contrasts while only 12 are
    #     positive, so NO rearrangement of its own predictions can exceed agreement 0.436-0.564
    #     against a majority-class baseline of 0.692. It cannot beat "always predict narrowing".
    # Measured agreement: 0.410 (quench) / 0.487 (stage-consistent), i.e. below a coin flip at
    # three of the four pre-registered cells.
    #
    # STRUCTURAL, not a tuning error. dEg is a pure function of (n_doped, n_undoped) — verified
    # bit-identically over 910 points — and carries band filling only: the disorder channel that
    # optical.py itself documents (the largest single term in the module) is omitted at BOTH call
    # sites and is identically zero in every dEg ever reported. Fitting m_bm/k_bgr cannot rescue
    # it: an oracle fitted ON THE TEST SET over 650 grid points in a box wider than physics allows
    # still fails. More data cannot rescue it either: at a true agreement of 0.80 only N = 10
    # contrasts are needed for 80% power, so the test is already ~4x over-powered for a real effect.
    #
    # IN-SCOPE SUBSET IS UNRESOLVED, NOT NEGATIVE — this is the reviewers' correction and it is
    # kept: the degenerate shallow-donor regime is the only one optical.py claims, and only 0-2 of
    # its 18 contrasts reach the Mott density, so that subset is UNTESTED rather than failed.
    #
    # TO EARN THE OLD GRADE BACK: a pre-registered within-paper set (N >= 39, content-deduplicated,
    # 0.05 eV deadband, one contrast per (paper, dopant)) on which sign agreement EXCEEDS that set's
    # own majority baseline AND beats a DOPANT-BLOCK permutation null at p < 0.05 after correcting
    # for the number of cells, at a NAMED stage. On today's corpus that is >= 28 of 39 vs M = 0.692.
    # ROUND 17 CORRECTION to the string below: "first test ever run" was FALSE and is removed.
    # The Round-17 grade audit surfaced a previously-uncited confrontation, results/tier2/
    # deg_validation.json (48/100 against a 0.52 baseline), which reaches the same conclusion by an
    # independent route. Round 16 was the first PRE-REGISTERED test with a floor, not the first test.
    # The audit rates this grade SUPPORTED AS WRITTEN on those two independent confrontations.
    "dEg": ("NOT VALIDATED — no demonstrated within-paper sign skill. Two independent "
            "confrontations agree: Round 16 (pre-registered, with a floor) and the earlier "
            "results/tier2/deg_validation.json (48/100 vs a 0.52 baseline). "
            "Agreement 0.41 quench / 0.49 stage-consistent on 39 within-paper contrasts "
            "against a 0.69 constant-'narrows' baseline, and the chain's own sign distribution "
            "(27-33 positive vs 12 measured positive) bounds it below that baseline by "
            "construction. Band-filling only: the disorder/narrowing channel is omitted, so gap "
            "NARROWING — 69% of the measured contrasts — is not expressible. The degenerate "
            "shallow-donor subset optical.py actually claims is UNRESOLVED, not failed (0-2 of 18 "
            "contrasts reach the Mott density). Report with an explicit stage tag; the sign is "
            "stage-dependent."),
    "dark": "trend/ranking only",
    "PDR": "direction/ranking only",
    "tau": "class-ranking only (absolute withheld)",
}

# The historical strings above are retained to explain old output files. They are not the
# current capability statement: input-unit and sample-pairing errors invalidate those grades.
FIDELITY = {
    "hall_n": "Unvalidated for general experimental prediction. Legacy hall_n_cm3 is free "
              "electron density; use hall_density_cm3 for a single-electron Hall comparison. "
              "Round 21/22 corpus accuracy claims are withdrawn pending source-level repair. "
              "See results/review_20261015 and paper/npj/revision_20261015.",
    "V_O": "Calculated total oxygen deficiency; not an XPS peak-area measurement. "
           "Isolated native vacancies are reported separately. No calibrated XPS mapping.",
    "hall_mu": "Empirical Caughey-Thomas transport with optional grain-boundary factor. "
               "A physics-derived ionized-defect input does not make this a first-principles mobility. "
               "Historical paired n/mu corpus results require re-evaluation.",
    "sigma": "Derived from free electron density and drift mobility; not an independent validation target.",
    "Eg": "Band-filling model output, without a sample-specific disorder or optical-transition calculation.",
    "dEg": "Exploratory band-filling difference; no general optical-gap accuracy established.",
    "dark": "Exploratory activation score; not an absolute dark current or a validated device ranking.",
    "PDR": "Exploratory score; supplied in-house summaries lack reproducible methods and contain "
           "arithmetic inconsistencies, so they do not validate photo/dark response.",
    "tau": "Exploratory deep-defect score; not an absolute recombination or photoresponse lifetime."
}


def _ionized_densities(db, q):
    """N_I (total ionized scattering centers) and N_A (ionized acceptors) [cm^-3] from quench.

    N_A is a strict SUBSET of N_I (N_I = Σ|q|N_cs already sums the ionized acceptors with
    weight |q|), so it is reported for diagnostics only — see transport.mobility.
    """
    Q = db.charge
    N_I = float((np.abs(Q) * q.N_cs).sum())
    N_A = float((np.clip(-Q, 0, None) * q.N_cs).sum())
    return max(N_I, 1.0), N_A


def _vo_native(db, r):
    """Density of ISOLATED native oxygen vacancies [cm^-3] (the 9 V_O charge states only).

    Selection: dm[O] == -1 and no other species changed. Verified census: 9 of 873 KROGER
    charge states.
    """
    o = db.col("O")
    is_vo = (db.dm[:, o] == -1) & (np.abs(db.dm).sum(axis=1) == 1)
    return float(r.N_cs[is_vo].sum())


def _vo_total_oxygen_deficiency(db, r):
    """TOTAL oxygen deficiency [cm^-3 of missing O] over ALL charge states.

    Σ_cs (−dm[O])·N_cs for dm[O] < 0, i.e. every O-deficient defect weighted by how many O
    atoms it actually removes from the lattice — isolated V_O, multi-vacancy natives, and
    dopant complexes alike.  Verified census of the KROGER DB (scripts/stage1_vo_census.py,
    results/accuracy_baseline/stage1_vo_census.json): 522 of 873 charge states have dm[O] < 0,
    of which only 9 are isolated native V_O; 318 of the 513 dropped ones carry a dopant.  In
    the Sn+Fe solution the native-only count was 5.94e10 of 6.74e12 cm^-3, i.e. it discarded
    99.1% (113x) of the oxygen-vacancy-bearing density.
    """
    o = db.col("O")
    dmO = db.dm[:, o]
    return float((np.clip(-dmO, 0, None) * r.N_cs).sum())


def forward(dopant, conc_frac, T_anneal=1073.0, pO2=1e-4, EcT_fraction=0.40,
            film=True, E_B_eV=0.02, Nd_bg=1e17, db: DefectDB | None = None, undoped_cache=None):
    """Simulate all nine properties for one (dopant, concentration, process) point.

    dopant: element symbol or 'undoped'. conc_frac: cation fraction (0 for undoped).
    Nd_bg: unintentional shallow-donor background (Si/H contamination); UID Ga2O3 ~1e17.
    Returns a dict with the nine properties + provenance + fidelity grades.
    """
    db = db or load()
    N_cat = N_CAT_SITE                             # cation site density [cm^-3]; see module head
    fixed = {} if dopant == "undoped" or conc_frac <= 0 else {dopant: conc_frac * N_cat}
    # STAGE-3 DESIGN 4: clamp to the SOURCED measured incorporation ceiling (Sn only; every other
    # element has no entry, so `fixed` is returned untouched and the flag is False).
    fixed, ceiling_info = apply_incorporation_ceiling(fixed)

    eq = engine.solve_single(db, T_anneal, pO2=pO2, EcT_fraction=EcT_fraction,
                             fixed_conc=fixed, fd=True, Nd_bg=Nd_bg)
    q = engine.quench(db, eq, T_quench=300.0, EcT_fraction=EcT_fraction, fd=True, Nd_bg=Nd_bg)

    n, p = float(q.n), float(q.p)
    N_I, N_A = _ionized_densities(db, q)
    N_I = N_I + Nd_bg                              # background donors also scatter
    # Oxygen vacancies: report BOTH the isolated-native count and the total oxygen deficiency.
    vo_nat_anneal = _vo_native(db, eq)             # isolated native V_O at process T
    vo_nat_quench = _vo_native(db, q)
    vo_tot_anneal = _vo_total_oxygen_deficiency(db, eq)   # native + all O-deficient complexes
    vo_tot_quench = _vo_total_oxygen_deficiency(db, q)
    # HEADLINE V_O = total oxygen deficiency. Justification against FIDELITY["V_O"]
    # ("trend/ordinal-proxy only, XPS O1s metrology disputed"): the corpus V_O observable is an
    # XPS O1s lattice-vs-deficient peak area ratio, which integrates ALL under-coordinated
    # oxygen environments and cannot resolve an isolated native vacancy from an
    # oxygen-deficient dopant complex. Matching the simulated quantity to what the metrology
    # actually measures is the correct choice for an ordinal proxy; the native-only count is
    # retained unchanged under V_O_native_cm3_* so nothing is lost.
    vo_anneal = vo_tot_anneal
    vo_quench = vo_tot_quench

    mu = transport.mobility(300.0, n_cm3=n, N_I_cm3=N_I, N_A_cm3=N_A, film=film, E_B_eV=E_B_eV)
    sigma = float(transport.conductivity(n, mu["mu_drift"]))

    # undoped reference at matched process (for within-paper dEg)
    if undoped_cache is None and dopant != "undoped":
        und = forward("undoped", 0.0, T_anneal=T_anneal, pO2=pO2, EcT_fraction=EcT_fraction,
                      film=film, E_B_eV=E_B_eV, Nd_bg=Nd_bg, db=db, undoped_cache={})
        n_und = und["hall_n_cm3"]
    elif dopant == "undoped":
        n_und = n
    else:
        n_und = undoped_cache.get("n_und", n)

    eg_opt = float(optical.optical_eg(n))
    deg = optical.dEg(n, max(n_und, 1.0))
    dev = device.device_foms(db, q, sigma, n)

    # ── STAGE-3 honesty blocks (all ADDITIVE; nothing above is suppressed or clipped) ──
    comp = compensation_block(db, q, T_anneal, Nd_bg=Nd_bg, Na_bg=0.0, n_anneal=float(eq.n))
    creg = carrier_regime_block(n, p, sigma)
    # ROUND 9: name WHICH carrier density is the comparable one. Switches on comp's abstention
    # flag, which is itself derived from sigma_dHo — no new parameter. No-op (bit-identical) in the
    # well-conditioned regime, so the rule has no discontinuity. See recommended_carrier_block.
    rec = recommended_carrier_block(n, float(eq.n), comp["compensation_abstain"])
    # ROUND 15: derived properties consistent with whichever carrier density is reported.
    # n_und for the differential must come from the SAME stage as the value it is differenced
    # against, otherwise dEg mixes stages too.
    _n_und_rec = n_und if dopant == "undoped" else (
        undoped_cache.get("n_und_rec", n_und) if undoped_cache else n_und)
    scb = stage_consistent_block(rec["recommended_n_cm3"], n, mu["mu_drift"], _n_und_rec,
                                 {"sigma_S_cm": sigma})

    return {
        "input": {"dopant": dopant, "conc_frac": conc_frac, "T_anneal": T_anneal,
                  "pO2": pO2, "film": film, "E_B_eV": E_B_eV, "EcT_fraction": EcT_fraction,
                  "N_cat_site_cm3": N_cat,
                  # REQUESTED loading (conc_frac * N_cat) ...
                  "dopant_loading_cm3": float(fixed.get(dopant, 0.0)) if fixed else 0.0,
                  # ... and what the solver ACTUALLY dissolved.  These differ whenever the
                  # site-blocking cap binds (measured: 540 of 1919 sweep points), and reporting
                  # only the request advertises a loading that was never reached.
                  "dopant_achieved_cm3": float(eq.meta.get("achieved_cm3", {})
                                               .get(dopant, 0.0)) if fixed else 0.0,
                  "dopant_target_hit_dex": float(eq.meta.get("target_hit_dex", {})
                                                 .get(dopant, 0.0)) if fixed else 0.0},
        "hall_n_cm3": n,  # deprecated legacy name: this is the free electron density
        "free_electron_cm3": n,
        "hall_density_cm3": n / mu["r_hall"],
        "hall_factor_rH": mu["r_hall"],
        "measurement_temperature_K": 300.0,
        "carrier_definition": "single-electron approximation: n_H=n/r_H; legacy hall_n_cm3=n",
        "recommended_output_warning": "Legacy recommended_* fields may refer to the anneal state; "
                                      "do not compare them with a 300 K Hall measurement.",
        "hall_p_cm3": p,
        "V_O_cm3_anneal": vo_anneal,          # == V_O_total_incl_complexes_cm3_anneal
        "V_O_cm3_quench": vo_quench,          # == V_O_total_incl_complexes_cm3_quench
        "V_O_native_cm3_anneal": vo_nat_anneal,
        "V_O_native_cm3_quench": vo_nat_quench,
        "V_O_total_incl_complexes_cm3_anneal": vo_tot_anneal,
        "V_O_total_incl_complexes_cm3_quench": vo_tot_quench,
        "V_O_definition": VO_DEFINITION,
        "hall_mu_cm2Vs": mu["mu_hall"],
        "mu_drift_cm2Vs": mu["mu_drift"],
        "sigma_S_cm": sigma,
        "Eg_optical_eV": eg_opt,
        "dEg_eV": deg["dEg"],
        "dark_activation_eV": dev["dark_activation_eV"],
        "PDR_score": dev["PDR_score"],
        "tau_score": dev["tau_score"],
        "N_I_ionized_cm3": N_I,
        "N_A_ionized_cm3": N_A,
        "EF_300K_below_Ec": float(q.Ec - q.EF),
        # honest name first; `solubility_limited` is the DEPRECATED alias (this is a dilute /
        # site-blocking ceiling, not a thermodynamic solubility boundary — see engine.py head)
        "site_blocking_limited": eq.meta.get("site_blocking_limited", False),
        "solubility_limited": eq.meta.get("solubility_limited", False),
        "mobility_components": mu,
        "dEg_components": deg,
        "device_detail": dev,
        "fidelity": FIDELITY,
        # ── STAGE-3 DESIGN 1 — compensation-margin abstention (non-lossy: hall_n_cm3 above is
        #    the unchanged computed value; these fields only DESCRIBE its conditioning) ──
        **comp,
        **rec,
        **scb,
        # ── STAGE-3 DESIGN 3 — acceptor-floor honesty (representability + detectability) ──
        **creg,
        # ── STAGE-3 DESIGN 4 — sourced incorporation ceiling (Sn only) ──
        "above_reported_incorporation_ceiling": ceiling_info[
            "above_reported_incorporation_ceiling"],
        "requested_dopant_cm3": float(ceiling_info["requested_dopant_cm3"].get(dopant, 0.0)),
        "ceiling_cm3": ceiling_info["ceiling_cm3"].get(dopant),
        "ceiling_source_doi": ceiling_info["ceiling_source_doi"].get(dopant),
        # ── STAGE-3 DESIGN 4 rank-2 — report-only compensation diagnostics, consumed by nothing.
        #    charge_cancellation_margin is the DEFECT-ONLY form quoted in
        #    results/codoping/cliff_mechanism.json (no p, no Nd_bg); it is kept for continuity and
        #    is NOT the quantity the abstention flag tests (that is compensation_margin above).
        "charge_cancellation_margin_defect_only": _defect_only_margin(db, q),
        "dominant_compensating_family": dominant_compensating_family(db, q),
    }


def _defect_only_margin(db, r) -> float:
    """(sum_pos - sum_neg)/max(sum_pos, sum_neg) over DEFECT charge only — the form quoted in
    results/codoping/cliff_mechanism.json.  Report-only; loses the exact n = P - M identity."""
    Q = db.charge
    pos = float((np.clip(Q, 0, None) * r.N_cs).sum())
    neg = float((np.clip(-Q, 0, None) * r.N_cs).sum())
    d = max(pos, neg)
    return float((pos - neg) / d) if d > 0 else 0.0
