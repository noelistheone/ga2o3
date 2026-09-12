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


def extend_with_dopant(db: DefectDB, dopant: str, charge_dHo: dict,
                       site_multiplicity: float = 1.0) -> ExtDefectDB:
    """Return an extended DB with `dopant` added as a new element column and a single
    substitutional defect (dopant on Ga: dm[Ga]=-1, dm[dopant]=+1) whose charge states are
    `charge_dHo` = {q: dHo_eV}. site_multiplicity scales the site prefactor (2 Ga sites; use
    1 for a specific site)."""
    nEl = len(db.elements)
    dm = np.hstack([db.dm, np.zeros((db.nCS, 1), dtype=db.dm.dtype)])
    new_id = db.nD + 1
    rows_dm, rows_dHo, rows_q, rows_pref, rows_id, rows_sum = [], [], [], [], [], []
    for q, dHo in sorted(charge_dHo.items()):
        r = np.zeros(nEl + 1, dtype=db.dm.dtype)
        r[0] = -1              # remove a Ga
        r[nEl] = +1            # add the dopant (new column)
        rows_dm.append(r)
        rows_dHo.append(float(dHo)); rows_q.append(int(q))
        rows_pref.append(N_SITE * site_multiplicity); rows_id.append(new_id)
        rows_sum.append(0.0)   # -1 + 1
    dm = np.vstack([dm, np.array(rows_dm, dtype=db.dm.dtype)])
    return ExtDefectDB(
        dHo=np.concatenate([db.dHo, rows_dHo]),
        charge=np.concatenate([db.charge, np.array(rows_q, dtype=db.charge.dtype)]),
        dm=dm,
        cs_ID=np.concatenate([db.cs_ID, np.array(rows_id, dtype=db.cs_ID.dtype)]),
        prefactor=np.concatenate([db.prefactor, rows_pref]),
        sum_dm=np.concatenate([db.sum_dm, rows_sum]),
        elements=list(db.elements) + [dopant],
        nD=db.nD + 1, nCS=db.nCS + len(charge_dHo),
    )


def classify(charge_dHo: dict, E_VBM_to_CBM: float = 4.9) -> dict:
    """Donor/acceptor character from the dHo(q) slopes. The transition level e(q1/q2) above
    VBM = (dHo(q1)-dHo(q2))/(q2-q1). A donor has positive charge states favorable in the lower
    gap; classify by where the highest +q -> 0 transition sits relative to CBM."""
    qs = sorted(charge_dHo)
    levels = {}
    for i in range(len(qs)):
        for j in range(i + 1, len(qs)):
            q1, q2 = qs[j], qs[i]         # q1 > q2
            eps = (charge_dHo[q1] - charge_dHo[q2]) / (q2 - q1)
            levels[f"{q1}/{q2}"] = {"above_VBM": round(eps, 3),
                                    "below_CBM": round(E_VBM_to_CBM - eps, 3)}
    pos = [q for q in qs if q > 0]
    verdict = "isovalent/neutral (no donor states)" if not pos else None
    if pos:
        key = f"{max(pos)}/0" if 0 in charge_dHo else f"{max(pos)}/{min(qs)}"
        bc = levels.get(key, {}).get("below_CBM")
        if bc is not None:
            verdict = (f"SHALLOW DONOR ({key} at CBM-{bc:.2f})" if bc < 0.3 else
                       f"deep donor ({key} at CBM-{bc:.2f})" if bc < E_VBM_to_CBM / 2 else
                       f"deep level/weak donor ({key} at CBM-{bc:.2f})")
    return {"transition_levels": levels, "classification": verdict}
