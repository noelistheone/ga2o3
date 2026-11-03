"""new_dopant — inject a dopant OUTSIDE KROGER's 19 elements into the defect-equilibrium
solver, using MLIP-relaxed geometry + QE/HSE energetics (Tier B).

The KROGER DB covers 19 elements. To predict a new dopant (Sb, Bi, ...) we compute its
substitutional defect's charge-state energies with our own QE (structure from MACE-MPA-0,
Tier B), convert to the KROGER dHo convention, and EXTEND the DefectDB with a new element
column + the new defect's charge states. Then the ordinary engine.solve_single with
fixed_conc={dopant: level} solves mu_dopant internally and yields the full self-consistent
equilibrium -> the same 9-property forward chain as for the native dopants.

  dHo(D^q) = E_tot(D^q) - E_perfect + q*E_VBM + E_corr(q)      [KROGER convention: E_F=VBM=0, mu=0]

Only dHo(q) is needed for the fixed-concentration prediction (mu_dopant is solved to hit the
target [dopant]); no elemental metal reference is required. The absolute concentration scale
is set by the specified doping level (itself the exp-sensitivity-limited quantity), while the
donor/acceptor character + compensation come from the dHo(q) slopes (mu-independent).
"""
from __future__ import annotations

import numpy as np

from .kroger_db import DefectDB, N_SITE
from .levels import transition_levels


class ExtDefectDB(DefectDB):
    """DefectDB whose col() resolves against its own (extended) element list."""
    def col(self, elem: str) -> int:
        return self.elements.index(elem) if isinstance(elem, str) else elem


def dHo_from_qe(E_tot_q: dict, E_perfect: float, E_VBM: float, makov_payne_q2: float) -> dict:
    """Convert QE total energies per charge state to KROGER dHo (eV).
    E_tot_q: {q: E_tot_eV}. E_corr(q) = Makov-Payne monopole ∝ q^2 (MP2 at q=2)."""
    out = {}
    for q, Et in E_tot_q.items():
        e_corr = makov_payne_q2 * (q / 2.0) ** 2
        out[int(q)] = float(Et - E_perfect + q * E_VBM + e_corr)
    return out


#: which host element a substitutional dopant displaces, per lattice sublattice.
SITE_HOST = {"cation": "Ga", "anion": "O"}


def extend_with_dopant(db: DefectDB, dopant: str, charge_dHo: dict,
                       site_multiplicity: float = 1.0,
                       site: str = "cation") -> ExtDefectDB:
    """Return an extended DB with `dopant` added as a new element column and a single
    substitutional defect whose charge states are `charge_dHo` = {q: dHo_eV}.

    `site` selects the sublattice the dopant substitutes ON (ROUND 20):
      'cation' (default, unchanged behaviour) -> X_Ga: dm[Ga] = -1, dm[X] = +1
      'anion'                                 -> X_O : dm[O]  = -1, dm[X] = +1

    The default is 'cation' and the cation path is BIT-IDENTICAL to the pre-Round-20 code
    (which hard-coded `r[0] = -1`, i.e. column 0 = Ga); verified for all 18 injected dopants
    against the byte-for-byte backup `new_dopant.py.pre_round20` in
    `results/mathwork/round20_anion.json` (gate G1).

    WHY THE ANION SITE NEEDS NOTHING ELSE. The dm bookkeeping is the whole mechanism: the
    engine forms dG = dHo + q*E_F - 3 kB T s_vib sum(dm) - dm.mu, so an anion-site row
    automatically picks up +mu_O (dm[O] = -1) instead of +mu_Ga, i.e. O-rich conditions
    correctly SUPPRESS X_O while they promote X_Ga. The KROGER DB already contains 28 genuine
    isolated X_O charge states (H 5, Si 5, Cr 3, Fe 3, Ir 12 — measured, see gate G0), all with
    exactly this row structure, so the solver has always supported the anion site; only this
    injection helper did not.

    SITE PREFACTOR. `site_multiplicity` scales the site prefactor. The default 1.0 is correct
    for BOTH sublattices and is not an approximation: every SIMPLE single-site substitutional
    row in the KROGER DB carries prefactor = N_SITE = 1.91e22 exactly — all 28 anion rows and
    95 of the 132 cation rows — because KROGER resolves the 2 inequivalent Ga sites and the 3
    inequivalent O sites as SEPARATE defect IDs of multiplicity one (verified, gate G3). The
    remaining 37 cation rows carry N_SITE/2 because they are split/dumbbell configurations
    occupying two Ga sites plus an interstitial; a future split-configuration ladder would need
    site_multiplicity=0.5, and no anion row in the DB is of that kind. A DFT ladder is computed
    for one named site, so multiplicity 1 is the matching convention. Pass 2.0 (cation) or 3.0
    (anion) only to represent a LUMPED sublattice whose sites are assumed iso-energetic; that
    is a modelling choice, not a correction.

    NOT HANDLED HERE, AND IT MATTERS: `full_forward.forward` converts conc_frac with
    N_CAT_SITE = 2*N_SITE = 3.82e22 unconditionally. An anion-site dopant sits on
    N_AN_SITE = 3*N_SITE = 5.73e22. Using the cation density for an anion dopant mis-loads by
    exactly 3/2 = 0.176 dex. full_forward is NOT changed by this function, so any anion-site
    prediction MUST set its own target concentration; see `round20_anion.json:blockers`.
    """
    if site not in SITE_HOST:
        raise ValueError(f"site must be one of {sorted(SITE_HOST)}, got {site!r}")
    host = SITE_HOST[site]
    try:
        host_col = list(db.elements).index(host)
    except ValueError:                                   # pragma: no cover - malformed DB
        raise ValueError(f"host element {host!r} for site={site!r} absent from db.elements")
    nEl = len(db.elements)
    dm = np.hstack([db.dm, np.zeros((db.nCS, 1), dtype=db.dm.dtype)])
    new_id = db.nD + 1
    rows_dm, rows_dHo, rows_q, rows_pref, rows_id, rows_sum = [], [], [], [], [], []
    for q, dHo in sorted(charge_dHo.items()):
        r = np.zeros(nEl + 1, dtype=db.dm.dtype)
        r[host_col] = -1       # remove a host atom (Ga for 'cation', O for 'anion')
        r[nEl] = +1            # add the dopant (new column)
        rows_dm.append(r)
        rows_dHo.append(float(dHo)); rows_q.append(int(q))
        rows_pref.append(N_SITE * site_multiplicity); rows_id.append(new_id)
        rows_sum.append(0.0)   # -1 + 1
    dm = np.vstack([dm, np.array(rows_dm, dtype=db.dm.dtype)])
    ext = ExtDefectDB(
        dHo=np.concatenate([db.dHo, rows_dHo]),
        charge=np.concatenate([db.charge, np.array(rows_q, dtype=db.charge.dtype)]),
        dm=dm,
        cs_ID=np.concatenate([db.cs_ID, np.array(rows_id, dtype=db.cs_ID.dtype)]),
        prefactor=np.concatenate([db.prefactor, rows_pref]),
        sum_dm=np.concatenate([db.sum_dm, rows_sum]),
        elements=list(db.elements) + [dopant],
        nD=db.nD + 1, nCS=db.nCS + len(charge_dHo),
    )
    # PROVENANCE TAG, additive and read by no numeric path (proved in gate G1: the
    # forward-chain output is bit-identical with and without it). An injected dopant without
    # its site is not a claim, the same way a carrier density without its stage is not.
    ext.injected_dopant = dopant
    ext.injected_site = site
    ext.injected_host_element = host
    ext.injected_site_prefactor_cm3 = float(N_SITE * site_multiplicity)
    # ROUND 20 REVIEW BLOCKER — the anion path shipped with NO ungated marker, in the one module
    # that is already in production. Added here so the marker travels with the object rather than
    # living only in a JSON nobody reads. The cation path is unmarked because it IS gated:
    # scripts/gate3_newdopant_corpus.py confronts every injected cation dopant with the corpus.
    if site == "anion":
        ext.UNGATED = True
        ext.ungated_reason = (
            "ANION-SITE INJECTION IS UNGATED. No anion-site prediction has ever been confronted "
            "with a measurement: gate3_newdopant_corpus scores cation-site injected dopants only, "
            "and the corpus contains no N- or F-doped beta-Ga2O3 transport row at all. Two further "
            "limits travel with any anion result. (1) full_forward.forward converts conc_frac with "
            "N_CAT_SITE = 2*N_SITE unconditionally, so an anion dopant is mis-loaded by exactly "
            "3/2 = 0.176 dex unless the caller sets an absolute target itself. (2) An anion "
            "substituent in an n-type oxide is an ACCEPTOR, which places it in the "
            "cancellation-dominated regime where the quenched carrier density is not identifiable "
            "(docs/RESEARCH_LOG.md sec. 8d, 13a) — so the level may be reported, the carrier "
            "density may not.")
        ext.ungated_mis_load_dex = 0.176
    return ext


def classify(charge_dHo: dict, E_VBM_to_CBM: float = 4.9, Eg_calc: float | None = None) -> dict:
    """Donor/acceptor character from the dHo(q) slopes, using the ADJACENT transition.

    ROUND 14 FIX (docs/RESEARCH_LOG.md sec. 13d/14). This function previously classified on
    eps(q_max/0) — a q-AVERAGE of two transitions — which is the thermodynamic level ONLY in the
    negative-U case. sandbox.levels' own module docstring already states the rule it was breaking:
    "Only ADJACENT transitions (q <-> q+-1) are used for trap depths ... never by a (q/0) average."
    The SOLVER was always right: for W its E_F pins at CBM-1.00664 against eps(1/0) = CBM-1.020
    (0.013 eV apart), i.e. on the adjacent transition. Only this reporting path was wrong.

    For a double donor with q = 2, 1, 0 and POSITIVE U (eps(2/1) < eps(1/0), so the +1 state IS
    stable over a finite E_F window), the level that governs n-type behaviour is the HIGHEST
    adjacent transition eps(1/0): under n-type conditions E_F sits near the CBM, and eps(1/0) is
    where the defect gives up its last electron. Measured for W: eps(2/1) = 3.4925 < eps(1/0) =
    3.8797 eV above VBM, U = +0.387 eV, so W is positive-U and eps(2/0) = 3.686 was never the right
    number to report. Only when U < 0 (the +1 state never stable) does the (2/0) average become the
    physical level, and that case is detected and handled here rather than assumed away.

    THE GAP REFERENCE IS A CHOICE, AND IT IS NOW RECORDED AS ONE. `E_VBM_to_CBM` places the level
    below the CBM. Passing the EXPERIMENTAL gap while the eigenvalues come from a functional whose
    own gap is smaller is a VBM-referenced scissor: it assumes the level tracks the valence band.
    For a donor built from cation d/s states hybridising with the conduction band (W 5d6s, Nb 4d)
    the level tracks the CBM instead, and the two choices bracket the truth. With alpha = 0.25
    (own gap 3.9050 eV) and the pipeline's E_VBM_to_CBM = 4.9, the bracket is ~0.995 eV wide, which
    was 83% of the W/Nb discrepancy. Pass `Eg_calc` (the gap of the calculation that produced
    charge_dHo) to get the bracket emitted explicitly; omit it and only the single VBM-referenced
    number is returned, exactly as before.

    NOTE ON WHAT THIS DOES AND DOES NOT FIX: Round 14 showed the delivered W/Nb prediction was
    wrong because it was reported on the QUENCHED stage, which is cancellation-dominated for those
    elements. On the anneal stage the original alpha = 0.25 energetics already land within the
    measurement scatter. So this fix corrects the CLASSIFICATION, not the carrier density; the
    carrier density is fixed by reporting the conditioned quantity (full_forward.
    recommended_carrier_block).
    """
    qs = sorted(charge_dHo)
    # transition_levels() enumerates every pair (needed for the record); the CLASSIFICATION below
    # uses only adjacent ones.
    levels = transition_levels(charge_dHo, E_VBM_to_CBM)
    pos = [q for q in qs if q > 0]
    out = {"transition_levels": levels}

    ladder_complete = all(qs[k + 1] - qs[k] == 1 for k in range(len(qs) - 1))
    neg = [q for q in qs if q < 0]
    if not pos:
        # ROUND 20 FIX. This branch previously returned "isovalent/neutral (no donor states)" for
        # ANY ladder without a positive charge state — including a PURE ACCEPTOR, which is not
        # isovalent and not neutral. That mislabel matters now: N on the O site, the entire
        # motivating case for the anion-site injection added this round, is a pure acceptor, so the
        # new capability's headline use case would have been reported as "no donor states" with no
        # depth at all. Found by an adversarial reviewer of Round 20.
        if not neg:
            out["classification"] = "isovalent/neutral (no charged states)"
            return out
        # Acceptor branch, mirroring the donor logic: under p-type conditions E_F sits near the
        # VBM, so the governing level is the LOWEST adjacent transition — where the defect accepts
        # its first electron — and its depth is measured ABOVE THE VBM, not below the CBM.
        adj_dn = [(qs[i + 1], qs[i]) for i in range(len(qs) - 1) if qs[i] < 0]
        if not adj_dn:
            out["classification"] = "acceptor (single charge state; no adjacent transition)"
            return out
        gq1, gq2 = min(adj_dn, key=lambda ab: levels[f"{ab[0]}/{ab[1]}"]["above_VBM"])
        gkey = f"{gq1}/{gq2}"
        eps = levels[gkey]["above_VBM"]
        U_acc = None
        if len(adj_dn) >= 2:
            srt = sorted(adj_dn, key=lambda ab: ab[1])           # ascending in the lower charge
            U_acc = (levels[f"{srt[-1][0]}/{srt[-1][1]}"]["above_VBM"]
                     - levels[f"{srt[0][0]}/{srt[0][1]}"]["above_VBM"])
        out.update({
            "governing_level": gkey,
            "governing_level_above_VBM": round(eps, 3),
            "governing_level_rule": "lowest ADJACENT transition (acceptor; depth is above the VBM)",
            "effective_U_eV": (None if U_acc is None else round(U_acc, 3)),
            "charge_ladder_complete": bool(ladder_complete),
            # A level BELOW the VBM (eps < 0) is RESONANT, not shallow: the negative charge
            # state is more stable than the neutral one even at E_F = VBM, so the defect is
            # always ionised. Calling that "shallow" would be wrong in the flattering direction.
            "resonant": bool(eps < 0.0),
            "classification": (
                f"RESONANT ACCEPTOR ({gkey} at {abs(eps):.2f} eV BELOW the VBM; always ionised)"
                if eps < 0.0 else
                f"SHALLOW ACCEPTOR ({gkey} at VBM+{eps:.2f})" if eps < 0.3 else
                f"deep acceptor ({gkey} at VBM+{eps:.2f})" if eps < E_VBM_to_CBM / 2 else
                f"deep level/weak acceptor ({gkey} at VBM+{eps:.2f})"),
            "acceptor_caveat": (
                "an acceptor in an n-type oxide sits in the CANCELLATION-DOMINATED regime: the "
                "quenched carrier density there is not identifiable (docs/RESEARCH_LOG.md sec. 8d, "
                "13a). This label describes the LEVEL only. Do not read it as a carrier density, "
                "a conductivity, or a claim that the material becomes p-type."),
        })
        return out

    # --- adjacent transitions among the positive ladder, highest first -------------------------
    # "Adjacent" means adjacent IN THE COMPUTED LADDER, which is not the same as q1 = q2 + 1 when
    # charge states are missing. The 18-element campaign used a q0+q2 recipe for several elements
    # (Mo, Y, Sc, Er, Yb), so their +1 state was NEVER COMPUTED: for those the only available
    # transition is the (2/0) average, the sign of U is UNDETERMINED, and the classification rests
    # on an untested assumption. That is flagged below rather than hidden.
    adj = [(qs[i + 1], qs[i]) for i in range(len(qs) - 1)]           # consecutive in the ladder
    ladder_complete = all(qs[i + 1] - qs[i] == 1 for i in range(len(qs) - 1))
    adj_up = [(a, b) for a, b in adj if a > 0]                        # transitions that ionise
    if not adj_up:
        # ROUND 21 FIX, same family as the Round-20 acceptor mislabel. Reaching here requires
        # pos to be non-empty and adj_up to be empty, and since the last adjacent pair always
        # carries a = max(qs) > 0, that happens if and only if the ladder holds EXACTLY ONE
        # charge state and it is positive. Calling that "isovalent/neutral" is wrong in the
        # most consequential direction available: a defect whose only in-gap charge state is
        # +q has no neutral state anywhere in the gap, i.e. its ionisation level lies ABOVE
        # the CBM. That is a resonant, always-ionised donor — the strongest n-type behaviour
        # the model can express, labelled as its opposite.
        #
        # This is not a corner case. In the KROGER PCCP-2025 database the lowest-formation-
        # energy substitutional ladders of Si, Sn, Ge, Hf and Zr on the Ga site are all exactly
        # {+1} — the canonical shallow donors of this material, every one of them.
        #
        # No depth is emitted, because none is measurable from a one-state ladder: the level is
        # outside the computed window and the data bounds it only from one side. The competing
        # reading — that the other charge states were simply never COMPUTED — cannot be
        # distinguished from inside this function, so it is stated rather than assumed away,
        # and charge_ladder_complete is set False (all() over an empty range would otherwise
        # report a one-state ladder as complete).
        q1 = max(pos)
        out.update({
            "governing_level": None,
            "governing_level_below_CBM": None,
            "charge_ladder_complete": False,
            "resonant": True,
            "single_charge_state": int(q1),
            "classification": (
                f"RESONANT/ALWAYS-IONISED DONOR (q=+{q1} is the only in-gap charge state, so "
                f"the +{q1}/0 level lies above the CBM; depth not measurable from this ladder)"),
            "single_state_caveat": (
                f"only one charge state (q=+{q1}) is present in the input. Read as resonant "
                f"donor if the ladder is the set of states STABLE in the gap (as in the KROGER "
                f"database); read as MISSING DATA if the other charge states were never "
                f"computed. This function cannot tell the two apart from the ladder alone."),
        })
        return out
    # governing level for n-type conditions = the HIGHEST adjacent transition (E_F nearest the CBM)
    gov_q1, gov_q2 = max(adj_up, key=lambda ab: levels[f"{ab[0]}/{ab[1]}"]["above_VBM"])
    gov_key = f"{gov_q1}/{gov_q2}"
    eps_gov = levels[gov_key]["above_VBM"]

    # --- effective-U over the positive ladder: negative U means the intermediate state is never
    #     stable and the q-average IS the physical level ------------------------------------------
    U = None
    if len(adj_up) >= 2:
        srt = sorted(adj_up, key=lambda ab: ab[0])                    # ascending in q1
        lo_key = f"{srt[0][0]}/{srt[0][1]}"                           # e.g. 1/0
        hi_key = f"{srt[-1][0]}/{srt[-1][1]}"                         # e.g. 2/1
        U = levels[lo_key]["above_VBM"] - levels[hi_key]["above_VBM"]
        if U < 0:                                                     # negative-U: use the average
            avg_key = f"{max(pos)}/0" if 0 in charge_dHo else f"{max(pos)}/{min(qs)}"
            if avg_key in levels:
                gov_key, eps_gov = avg_key, levels[avg_key]["above_VBM"]

    def _verdict(bc, key):
        # bc = depth BELOW the CBM. bc < 0 means the level sits ABOVE the CBM: the defect is a
        # RESONANT donor, always ionised. Naming it "shallow" understates it in the flattering
        # direction, which is the error class this project keeps finding, so it is named.
        if bc < 0.0:
            return f"RESONANT DONOR ({key} at {abs(bc):.2f} eV ABOVE the CBM; always ionised)"
        return (f"SHALLOW DONOR ({key} at CBM-{bc:.2f})" if bc < 0.3 else
                f"deep donor ({key} at CBM-{bc:.2f})" if bc < E_VBM_to_CBM / 2 else
                f"deep level/weak donor ({key} at CBM-{bc:.2f})")

    bc_ref = E_VBM_to_CBM - eps_gov
    out.update({
        "governing_level": gov_key,
        "governing_level_above_VBM": round(eps_gov, 3),
        "governing_level_rule": ("highest ADJACENT transition (positive U, intermediate state "
                                 "stable)" if (U is None or U >= 0) else
                                 "q-average (NEGATIVE U, intermediate state never stable)"),
        "effective_U_eV": (None if U is None else round(U, 3)),
        "charge_ladder_complete": bool(ladder_complete),
        "charge_ladder_caveat": (None if ladder_complete else
                                 f"charge states {qs} have gaps: the intermediate state(s) were "
                                 f"never computed, so the sign of U is UNDETERMINED and this "
                                 f"classification assumes the q-average is the physical level. "
                                 f"Computing the missing charge state(s) is required to confirm it."),
        "classification": _verdict(bc_ref, gov_key),
        "gap_reference_eV": E_VBM_to_CBM,
        "gap_reference_note": ("VBM-referenced: the level is placed below the CBM using the gap "
                               "passed in. If that gap is EXPERIMENTAL while charge_dHo came from "
                               "a functional with a smaller gap, this is a scissor CHOICE that "
                               "systematically deepens donors — pass Eg_calc for the bracket."),
    })
    if Eg_calc is not None:
        bc_own = float(Eg_calc) - eps_gov
        lo, hi = sorted((bc_ref, bc_own))
        out.update({
            "Eg_calc_eV": float(Eg_calc),
            "below_CBM_bracket_eV": [round(lo, 3), round(hi, 3)],
            "classification_bracket": [_verdict(lo, gov_key), _verdict(hi, gov_key)],
            "bracket_note": (f"the two scissor conventions bracket the level: {lo:.3f}-{hi:.3f} eV "
                             f"below the CBM (own-gap {Eg_calc:g} eV vs reference "
                             f"{E_VBM_to_CBM:g} eV). Report the bracket, not one end."),
        })
    return out
