"""levels — the ONE definition of a charge-transition level in this package.

The formula lived twice-over implicitly: `new_dopant.classify()` wrote it out for injected
dopants, and `device._deep_trap_densities` needed it but never had it (it summed every negative
charge state as a "deep acceptor", so shallow always-ionised donors/acceptors were counted as
traps — see results/accuracy_baseline/stage3_prereg_acceptor_pdr.md §B0).  Both now call the
functions here, per the Stage-3 pre-registration §B1 instruction "refactor to a shared
sandbox.levels helper, do not duplicate it".

Convention (identical to the KROGER `dHo` convention the DB is stored in): E = 0 at the 0 K VBM,
`dG(q) = dHo(q) + q*E_F - ...` with E_F measured from that same zero.  Therefore

    eps(q1/q2) = (dHo(q1) - dHo(q2)) / (q2 - q1)          [eV above the 0 K VBM],  q1 > q2

is the Fermi level at which charge states q1 and q2 are degenerate — the thermodynamic transition
level.  Only ADJACENT transitions (q <-> q±1) are used for trap depths: the population of state q
is set by its two neighbouring levels, never by a (q/0) average.
"""
from __future__ import annotations


def transition_level(dHo_q1: float, dHo_q2: float, q1: int, q2: int) -> float:
    """eps(q1/q2) above the 0 K VBM, in eV.  Requires q1 != q2 (caller passes q1 > q2)."""
    return (float(dHo_q1) - float(dHo_q2)) / (q2 - q1)


def transition_levels(charge_dHo: dict, E_VBM_to_CBM: float = 4.9) -> dict:
    """All pairwise transition levels of one defect.

    charge_dHo: {q: dHo_eV}.  Returns {"q1/q2": {"above_VBM": eps, "below_CBM": E_gap - eps}}
    with q1 > q2 — the exact structure `new_dopant.classify()` has always returned.
    """
    qs = sorted(charge_dHo)
    out = {}
    for i in range(len(qs)):
        for j in range(i + 1, len(qs)):
            q1, q2 = qs[j], qs[i]                       # q1 > q2
            eps = transition_level(charge_dHo[q1], charge_dHo[q2], q1, q2)
            out[f"{q1}/{q2}"] = {"above_VBM": round(eps, 3),
                                 "below_CBM": round(E_VBM_to_CBM - eps, 3)}
    return out


def trap_depth(q: int, dHo_q: float, dHo_partner: float, Ev: float, Ec: float):
    """Depth of the ADJACENT transition level that governs charge state q, in eV from its band.

    q < 0 -> HOLE trap.  Partner is q+1.  eps_h = eps(q+1/q) = dHo(q) - dHo(q+1);
             depth = eps_h - Ev  (above the VBM at the reporting temperature).
    q > 0 -> ELECTRON trap.  Partner is q-1.  eps_e = eps(q/q-1) = dHo(q-1) - dHo(q);
             depth = Ec - eps_e  (below the CBM at the reporting temperature).
    q == 0 carries no charge and has no depth -> returns (None, None).

    Returns (kind, depth_eV) with kind in {"hole", "electron", None}.
    """
    if q < 0:
        eps = transition_level(dHo_partner, dHo_q, q + 1, q)     # = dHo(q) - dHo(q+1)
        return "hole", eps - float(Ev)
    if q > 0:
        eps = transition_level(dHo_q, dHo_partner, q, q - 1)     # = dHo(q-1) - dHo(q)
        return "electron", float(Ec) - eps
    return None, None
