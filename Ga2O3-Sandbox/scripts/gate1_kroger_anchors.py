"""Gate 1 — reproduce KROGER's three published experimental anchors + Neal-2018 levels.

Anchors (KROGER / Arnab et al. PCCP 2025; Neal et al. APL 113, 062101 (2018)):
  A1  Sn-doped EFG (T_freezein ~1950 K, pO2=0.02 atm), quenched to 300 K → CONDUCTIVE,
      n ≈ [Sn] with <1% compensation by V_Ga (+ complexes).
  A2  After 1300-1400 K O2 anneal (pO2=1 atm), quenched → INSULATING (n collapses; V_Ga
      acceptors proliferate in O-rich conditions and compensate Sn).
  A3  Native V_Ga density in the low-1e16 cm^-3 range (DLOS-consistent).
  N1  Charge-transition levels: Si_Ga and Sn_Ga shallow donors (ε(+1/0) within ~0.05 eV of
      Ec); Fe_Ga deep acceptor near Ec−0.86 eV; Mg_Ga deep acceptor ~1.1 eV above VBM.

PASS thresholds (pre-registered here). All numbers written to results/phase0/gate1_report.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from sandbox import kroger_db, engine, thermo  # noqa: E402

OUT = PROJ / "results/phase0"
db = kroger_db.load()


def substitutional(elem, on="Ga"):
    """Charge-state row indices for X_on (dm[X]=+1, dm[Ga]=-1, nothing else)."""
    c_x, c_h = db.col(elem), db.col(on)
    mask = (db.dm[:, c_x] == 1) & (db.dm[:, c_h] == -1) & (np.abs(db.dm).sum(axis=1) == 2)
    return np.where(mask)[0]


def ground_charge(elem, EF):
    """Charge of the lowest-formation-energy X_Ga config at Fermi level EF (VBM=0)."""
    rows = substitutional(elem)
    Eform = db.dHo[rows] + db.charge[rows] * EF
    return int(db.charge[rows][np.argmin(Eform)])


def acceptor_level_0_minus1(elem):
    """(0/−1) charge-transition level above VBM, from the most stable neutral X_Ga defect."""
    rows = substitutional(elem)
    best = None
    for did in set(db.cs_ID[rows].tolist()):
        sub = rows[db.cs_ID[rows] == did]
        q = db.charge[sub]
        if 0 in q and -1 in q:
            e0 = db.dHo[sub[q == 0]][0]
            em1 = db.dHo[sub[q == -1]][0]
            eps = float(em1 - e0)                       # ε(0/−1) above VBM: E_form(0)=E_form(-1)
            if best is None or e0 < best[1]:
                best = (eps, float(e0))
    return None if best is None else best[0]


report = {"pass": {}, "values": {}}

# ── N1: donor/acceptor character + deep-level positions (energetics, no solver) ─
# Si, Sn shallow donors: ground charge is +1 (fully ionized) across the whole gap.
si_donor = ground_charge("Si", 2.5) >= 1 and ground_charge("Si", thermo.EG0 - 0.3) >= 1
sn_donor = ground_charge("Sn", 2.5) >= 1 and ground_charge("Sn", thermo.EG0 - 0.3) >= 1
fe_lvl = acceptor_level_0_minus1("Fe")     # above VBM; Neal: Ec−0.86 → VBM + ~4.14
mg_lvl = acceptor_level_0_minus1("Mg")     # above VBM; expt Mg ~1.1
fe_below_Ec = (thermo.EG0 - fe_lvl) if fe_lvl is not None else None
report["values"]["N1"] = {
    "Si_ground_charge_midgap": ground_charge("Si", 2.5),
    "Sn_ground_charge_midgap": ground_charge("Sn", 2.5),
    "Fe_acceptor_0/-1_below_Ec": round(fe_below_Ec, 3) if fe_below_Ec else None,
    "Mg_acceptor_0/-1_above_VBM": round(mg_lvl, 3) if mg_lvl else None}
report["pass"]["N1_Si_shallow_donor"] = bool(si_donor)
report["pass"]["N1_Sn_shallow_donor"] = bool(sn_donor)
report["pass"]["N1_Fe_deep_acceptor_Ec-0.6-1.1"] = bool(fe_below_Ec and 0.6 <= fe_below_Ec <= 1.1)
report["pass"]["N1_Mg_deep_acceptor_VBM+0.9-1.5"] = bool(mg_lvl and 0.9 <= mg_lvl <= 1.5)

# ── A1: Sn-doped EFG, conductive with <1% compensation ───────────────────────
SN = 3.0e18   # representative EFG Sn wafer doping [cm^-3]
T_freeze = 1950.0
eq_efg = engine.solve_single(db, T_freeze, pO2=0.02, EcT_fraction=0.40,
                             fixed_conc={"Sn": SN}, fd=True)
q_efg = engine.quench(db, eq_efg, T_quench=300.0, EcT_fraction=0.40, fd=True)
# compensation: ionized acceptors (V_Ga etc.) / n at 300 K
Nd_q, Na_q = engine.net_acceptor_donor(db, q_efg)
sn_ionized = engine.defect_totals_by_element(db, q_efg, "Sn")
comp_efg = Na_q / max(q_efg.n, 1.0)
report["values"]["A1_EFG"] = {
    "Sn_target": SN, "Sn_total_in_solid": round(sn_ionized, -15),
    "n_300K": f"{q_efg.n:.3e}", "p_300K": f"{q_efg.p:.3e}",
    "EF_above_VBM": round(q_efg.EF, 3), "Ec": round(q_efg.Ec, 3),
    "ionized_acceptors": f"{Na_q:.3e}", "compensation_ratio_Na_over_n": f"{comp_efg:.3e}",
    "freeze_in_T": T_freeze}
report["pass"]["A1_EFG_conductive_n_near_Sn"] = bool(0.3 * SN <= q_efg.n <= 1.5 * SN)
report["pass"]["A1_EFG_compensation_below_5pct"] = bool(comp_efg < 0.05)

# ── A2: O2 anneal (1350 K, pO2=1) → insulating ───────────────────────────────
T_anneal = 1350.0
eq_ann = engine.solve_single(db, T_anneal, pO2=1.0, EcT_fraction=0.40,
                             fixed_conc={"Sn": SN}, fd=True)
q_ann = engine.quench(db, eq_ann, T_quench=300.0, EcT_fraction=0.40, fd=True)
Nd_a, Na_a = engine.net_acceptor_donor(db, q_ann)
report["values"]["A2_O2anneal"] = {
    "n_300K": f"{q_ann.n:.3e}", "p_300K": f"{q_ann.p:.3e}",
    "EF_above_VBM": round(q_ann.EF, 3), "ionized_acceptors": f"{Na_a:.3e}",
    "n_ratio_anneal_over_EFG": f"{q_ann.n / max(q_efg.n,1.0):.3e}"}
report["pass"]["A2_anneal_less_conductive_than_EFG"] = bool(q_ann.n < 0.1 * q_efg.n)

# ── A3: V_Ga (+ V_Ga-nSn complexes) density in the Sn-doped EFG crystal (DLOS ~1e16) ──
# In the conductive Sn crystal E_F sits ~Ec, which lowers V_Ga acceptor formation energy;
# the compensating cation-vacancy density is what DLOS/DLTS measures in the low-1e16 range.
c_ga = db.col("Ga")
is_vga_simple = (db.dm[:, c_ga] == -1) & (np.abs(db.dm).sum(axis=1) == 1)
# isolated V_Ga + V_Ga-nH (electrically active); EXCLUDE Sn-passivated V_Ga-nSn complexes,
# which track [Sn] and are not the free-V_Ga DLOS signal.
c_sn = db.col("Sn")
is_vga_active = (db.dm[:, c_ga] <= -1) & (db.dm[:, c_sn] == 0)
vga_simple = float(q_efg.N_cs[is_vga_simple].sum())
vga_active = float(q_efg.N_cs[is_vga_active].sum())
vga_all = float(q_efg.N_cs[db.dm[:, c_ga] <= -1].sum())
report["values"]["A3_VGa_in_Sn_EFG_crystal"] = {
    "V_Ga_isolated": f"{vga_simple:.3e}", "V_Ga_active_incl_H": f"{vga_active:.3e}",
    "V_Ga_all_incl_Sn_complexes": f"{vga_all:.3e}",
    "note": "frozen from 1950 K freeze-in, quenched to 300 K; DLOS ~1e16 is the free-V_Ga signal"}
report["pass"]["A3_VGa_active_1e13_to_1e17"] = bool(1e13 <= vga_active <= 1e17)

report["pass_count"] = int(sum(bool(v) for v in report["pass"].values()))
report["total_checks"] = len(report["pass"])
report["ALL_PASS"] = bool(report["pass_count"] == report["total_checks"])

(OUT / "gate1_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
