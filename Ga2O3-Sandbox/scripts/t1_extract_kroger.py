"""T1.1/T1.2 — verify KROGER 092724 (873/259/19), extract ALL numeric arrays to a
portable npz + JSON, and identify host columns / native defects WITHOUT needing the
MCOS name strings (which scipy cannot read). Formation-energy convention (from
Ga2O3_defect_equilib_script_with_frozen.m line 1049-1051):

    E_form(cs, E_F) = cs_dHo  -  cs_dm . mu_vec  +  cs_charge * E_F

cs_dm[cs, element] = atoms of each element ADDED to form the defect (removed = negative);
mu_vec in the elementnames column order. Host columns (Ga, O) are the two most-used
columns of cs_dm across all charge states.
"""
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

PROJ = Path(__file__).resolve().parents[1]
MAT = PROJ / "external/KROGER/Ga2O3/Ga2O3_Varley_all_defects_new_092724.mat"
OUT = PROJ / "results/phase0"
OUT.mkdir(parents=True, exist_ok=True)

m = sio.loadmat(MAT, squeeze_me=True, struct_as_record=False)
d = m["defects"]

nD = int(np.asarray(d.num_defects))
nCS = int(np.asarray(d.num_chargestates))
nEl = int(np.asarray(d.numelements))
report = {"file": MAT.name, "num_defects": nD, "num_chargestates": nCS, "num_elements": nEl,
          "verify_873_259_19": bool(nCS == 873 and nD == 259 and nEl == 19)}

cs_dHo = np.asarray(d.cs_dHo, dtype=np.float64)
cs_charge = np.asarray(d.cs_charge, dtype=np.int64)
cs_dm = np.asarray(d.cs_dm, dtype=np.int64)                       # [873, 19]
cs_ID = np.asarray(d.cs_ID, dtype=np.int64)                      # defect id per cs (1..259)
cs_degen_config = np.asarray(d.cs_degen_factor_config, dtype=np.float64)
cs_degen_elec = np.asarray(d.cs_degen_factor_elec, dtype=np.float64)
cs_prefactor = np.asarray(d.cs_prefactor, dtype=np.float64)
cs_site_prefactor = np.asarray(d.cs_site_prefactor, dtype=np.float64)
cs_num_each_site = np.asarray(d.cs_num_each_site, dtype=np.int64)  # [873, 9]
cs_Emig = np.asarray(d.cs_Emigration, dtype=np.float64)           # [259]
cs_indices_lo = np.asarray(d.cs_indices_lo, dtype=np.int64)
cs_indices_hi = np.asarray(d.cs_indices_hi, dtype=np.int64)
ncs_per = np.asarray(d.defect_num_chargestates_per, dtype=np.int64)

# ── identify host columns: the two most-used columns of cs_dm ─────────────────
usage = (cs_dm != 0).sum(axis=0)                                  # per-element usage count
order = np.argsort(usage)[::-1]
report["cs_dm_column_usage"] = usage.tolist()
report["two_most_used_columns(0idx)"] = order[:2].tolist()
# Ga vs O: O column should be the one where simple V_O (single -1, all else 0, small |dHo|,
# charges {0,1,2}) lives. Detect single-species -1 defects per candidate column.
host_cols = sorted(order[:2].tolist())
c0, c1 = host_cols


def single_species_rows(col, val):
    mask = (cs_dm[:, col] == val) & (np.abs(cs_dm).sum(axis=1) == abs(val))
    return np.where(mask)[0]


# V_O candidate = single -1 in a host col with charge states {0,1,2} (donor)
for col in host_cols:
    rows = single_species_rows(col, -1)
    charges = sorted(set(cs_charge[rows].tolist()))
    report.setdefault("single_minus1_by_col", {})[str(col)] = {
        "n_chargestates": int(len(rows)),
        "charges": charges,
        "dHo_examples": cs_dHo[rows][:12].round(3).tolist(),
    }

# ── native-only defects: nonzero dm only in host columns ─────────────────────
dopant_cols = [c for c in range(nEl) if c not in host_cols]
native_cs = np.where(cs_dm[:, dopant_cols].sum(axis=1) == 0)[0] if dopant_cols else np.arange(nCS)
native_cs = np.where(np.abs(cs_dm[:, dopant_cols]).sum(axis=1) == 0)[0]
report["n_native_chargestates"] = int(len(native_cs))

# quick native census: group by (dm_c0, dm_c1) signature
sig_map = {}
for i in native_cs:
    key = (int(cs_dm[i, c0]), int(cs_dm[i, c1]))
    sig_map.setdefault(key, []).append(i)
report["native_dm_signatures (dm_col%d,dm_col%d)->count" % (c0, c1)] = {
    str(k): len(v) for k, v in sorted(sig_map.items())}

# charge-state span sanity
report["charge_range"] = [int(cs_charge.min()), int(cs_charge.max())]
report["dHo_range"] = [float(cs_dHo.min()), float(cs_dHo.max())]
report["degen_config_range"] = [float(cs_degen_config.min()), float(cs_degen_config.max())]
report["degen_elec_range"] = [float(cs_degen_elec.min()), float(cs_degen_elec.max())]

# ── save portable numeric bundle ─────────────────────────────────────────────
np.savez_compressed(
    OUT / "kroger_ga2o3_092724_numeric.npz",
    cs_dHo=cs_dHo, cs_charge=cs_charge, cs_dm=cs_dm, cs_ID=cs_ID,
    cs_degen_config=cs_degen_config, cs_degen_elec=cs_degen_elec,
    cs_prefactor=cs_prefactor, cs_site_prefactor=cs_site_prefactor,
    cs_num_each_site=cs_num_each_site, cs_Emigration=cs_Emig,
    cs_indices_lo=cs_indices_lo, cs_indices_hi=cs_indices_hi,
    ncs_per=ncs_per, host_cols=np.array(host_cols),
    num_defects=nD, num_chargestates=nCS, num_elements=nEl,
)
(OUT / "kroger_extract_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
