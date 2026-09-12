"""Clean, portable loader for the KROGER β-Ga2O3 defect database (numeric arrays only).

Numeric formation energies / dm / charges are FACTS from the published PCCP 2025 database
(Arnab et al.), extracted with scipy from Ga2O3_Varley_all_defects_new_092724.mat (873 charge
states / 259 defects / 19 elements — verified). The MCOS name strings are unreadable by scipy,
so the element order is taken from the KROGER script's own comment
(KROGER_Set_Ga2O3_Thermo_Conditions.m line 177 + Ga2O3_stoich.m line 56):

  0=Ga 1=O 2=Si 3=H 4=Fe 5=Sn 6=Cr 7=Ti 8=Ir 9=Mg 10=Ca 11=Zn 12=Co 13=Zr 14=Hf 15=Ta 16=Ge 17=Pt 18=Rh

Formation free energy of a charge state at Fermi level E_F (VBM=0) and host μ vector:
  dG(cs) = cs_dHo + q·E_F − 3 k_B T · s_vib(T) · Σ cs_dm  −  cs_dm · μ_vec
  N_cs   = cs_tot_prefactor · exp(−dG/k_B T)              (dilute Boltzmann)

Provenance note: a from-SI clean-room rebuild is still pending (see data/energetics/); until
then these arrays carry KROGER-database provenance and are used as citable published data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[2]
NPZ = PROJ / "results/phase0/kroger_ga2o3_092724_numeric.npz"

ELEMENTS = ["Ga", "O", "Si", "H", "Fe", "Sn", "Cr", "Ti", "Ir", "Mg",
            "Ca", "Zn", "Co", "Zr", "Hf", "Ta", "Ge", "Pt", "Rh"]
ELEM_COL = {e: i for i, e in enumerate(ELEMENTS)}
N_SITE = 1.91e22  # cm^-3, all 9 KROGER site types identical


@dataclass
class DefectDB:
    dHo: np.ndarray          # [nCS] reference formation energy at E_F=VBM, μ=0 [eV]
    charge: np.ndarray       # [nCS] int
    dm: np.ndarray           # [nCS, nEl] atoms of each element added (removed = negative)
    cs_ID: np.ndarray        # [nCS] parent defect id (1..nD)
    prefactor: np.ndarray    # [nCS] cs_tot_prefactor [cm^-3]
    sum_dm: np.ndarray       # [nCS] Σ over elements (for the vibrational-entropy term)
    elements: list
    nD: int
    nCS: int

    def col(self, elem: str) -> int:
        return ELEM_COL[elem]

    def dopant_cols(self) -> list:
        return [i for i in range(len(self.elements)) if i not in (0, 1)]


def load(npz_path: str | Path = NPZ) -> DefectDB:
    z = np.load(npz_path)
    dHo = z["cs_dHo"].astype(np.float64)
    charge = z["cs_charge"].astype(np.int64)
    dm = z["cs_dm"].astype(np.int64)
    cs_ID = z["cs_ID"].astype(np.int64)
    degen_cfg = z["cs_degen_config"].astype(np.float64)
    degen_elec = z["cs_degen_elec"].astype(np.float64)
    num_each_site = z["cs_num_each_site"].astype(np.float64)      # [nCS, 9]
    nEl = int(z["num_elements"])
    assert nEl == len(ELEMENTS), f"element-count mismatch {nEl} vs {len(ELEMENTS)}"

    # site_prefactor = min_j (N_site_j / num_each_site_j) over used sites; all N_site_j equal.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(num_each_site > 0, N_SITE / num_each_site, np.inf)
    site_prefactor = ratio.min(axis=1)
    site_prefactor = np.where(np.isfinite(site_prefactor), site_prefactor, N_SITE)
    tot_prefactor = degen_cfg * degen_elec * site_prefactor       # degen factors are 1.0 here

    return DefectDB(dHo=dHo, charge=charge, dm=dm, cs_ID=cs_ID, prefactor=tot_prefactor,
                    sum_dm=dm.sum(axis=1).astype(np.float64), elements=list(ELEMENTS),
                    nD=int(z["num_defects"]), nCS=int(z["num_chargestates"]))


def write_provenance_db(out_dir: str | Path = PROJ / "data/energetics") -> dict:
    """Persist the numeric DB + element order + provenance to data/energetics/ (Tier 1.3)."""
    db = load()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "kroger_ga2o3_092724.npz",
        dHo=db.dHo, charge=db.charge, dm=db.dm, cs_ID=db.cs_ID,
        prefactor=db.prefactor, sum_dm=db.sum_dm,
    )
    meta = {
        "schema_version": 1,
        "source": "KROGER Ga2O3_Varley_all_defects_new_092724.mat (Arnab et al., PCCP 2025, "
                  "DOI 10.1039/D4CP04817B; github.com/mikescarpulla/KROGER, UNLICENSED)",
        "verified_counts": {"charge_states": db.nCS, "defects": db.nD, "elements": len(db.elements)},
        "element_order_0indexed": db.elements,
        "element_order_provenance": "KROGER_Set_Ga2O3_Thermo_Conditions.m line 177 + "
                                    "Ga2O3_stoich.m line 56 (host Ga,O then 17 dopants incl Pt,Rh)",
        "formation_energy_convention": "dG = cs_dHo + q*EF - 3 kBT s_vib(T) sum(dm) - dm.mu; "
                                       "N = prefactor * exp(-dG/kBT); EF from VBM=0.",
        "lab_dopant_columns": {e: ELEM_COL[e] for e in ("Si", "Sn", "Mg", "Zn")},
        "verified": True,
        "clean_room_SI_rebuild": "PENDING — numeric arrays used as citable published data; "
                                 "authoritative from-SI/paper rebuild deferred (unlicensed repo).",
    }
    (out_dir / "kroger_ga2o3_092724.provenance.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    m = write_provenance_db()
    db = load()
    print(json.dumps(m, indent=2))
    # native V_O census (Sn=off): V_O = dm O=-1, all others 0
    o = db.col("O")
    is_VO = (db.dm[:, o] == -1) & (np.abs(db.dm).sum(axis=1) == 1)
    print("\nV_O charge states (native):")
    for i in np.where(is_VO)[0]:
        print(f"  q={db.charge[i]:+d}  dHo={db.dHo[i]:+.3f} eV  pref={db.prefactor[i]:.3e}")
