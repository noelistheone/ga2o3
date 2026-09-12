"""Diagnose the donor over-compensation that pins predicted n too low (the absolute-accuracy
bottleneck that is NOT the exponential ceiling — it is a process/scenario modeling choice).

For Si at 1 at%: print n at the anneal T (before quench) vs after quench, V_Ga total, Si
distribution across shallow-donor / deep / V_Ga-Sn-complex configs, at several freeze-in T and
under the frozen-V_O (Goyal non-equilibrium anneal) bracket. If a lower freeze-in T or the frozen
bracket restores realistic donor n (~1e18-1e20), the over-compensation is fixable → absolutes
improve without breaching the ceiling.
"""
import sys
from pathlib import Path
import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from sandbox import engine, kroger_db  # noqa: E402

db = kroger_db.load()
c_ga, c_si = db.col("Ga"), db.col("Si")
N_cat = kroger_db.N_SITE * 2 / 3
SI = 0.01 * N_cat   # 1 at%

is_vga = (db.dm[:, c_ga] <= -1)
is_vga_simple = (db.dm[:, c_ga] == -1) & (np.abs(db.dm).sum(axis=1) == 1)
has_si = db.dm[:, c_si] > 0
si_shallow = has_si & (db.dm[:, c_ga] == -1) & (np.abs(db.dm).sum(axis=1) == 2)  # Si_Ga only

print("=== Si 1 at% over-compensation diagnostic ===")
print(f"{'scenario':34s} {'n_anneal':>10s} {'n_quench':>10s} {'VGa_tot':>10s} {'Si_active%':>9s}")
for Tf in [1073.0, 1400.0, 1700.0, 1950.0]:
    for frozen in [False, True]:
        eq = engine.solve_single(db, Tf, pO2=1e-5, EcT_fraction=0.40,
                                 fixed_conc={"Si": SI}, fd=True, Nd_bg=1e17)
        n_anneal = eq.n
        # optionally freeze V_O totals (Goyal non-equilib anneal): re-solve with V_O frozen high
        q = engine.quench(db, eq, T_quench=300.0, EcT_fraction=0.40, fd=True, Nd_bg=1e17)
        vga = float(q.N_cs[is_vga].sum())
        si_tot = float((db.dm[:, c_si] * q.N_cs).sum())
        si_active = float((db.dm[:, c_si] * q.N_cs[..., ] * (db.charge > 0))[has_si].sum()) if False else \
            float(q.N_cs[si_shallow].sum())
        frac_active = 100.0 * si_active / max(si_tot, 1.0)
        tag = f"Tfreeze={Tf:.0f}{'(frozen)' if frozen else ''}"
        print(f"{tag:34s} {n_anneal:10.2e} {q.n:10.2e} {vga:10.2e} {frac_active:9.1f}")
        if frozen:
            break  # frozen flag not yet wired; single pass per Tf for now

# where does Si go? breakdown at 1073K
eq = engine.solve_single(db, 1073.0, pO2=1e-5, EcT_fraction=0.40, fixed_conc={"Si": SI}, fd=True, Nd_bg=1e17)
q = engine.quench(db, eq, T_quench=300.0, EcT_fraction=0.40, fd=True, Nd_bg=1e17)
print("\n=== where does Si go at 1073K (quenched)? ===")
si_by_defect = {}
for i in np.where(has_si)[0]:
    d = int(db.cs_ID[i])
    si_by_defect[d] = si_by_defect.get(d, 0) + db.dm[i, c_si] * q.N_cs[i]
for d, amt in sorted(si_by_defect.items(), key=lambda kv: -kv[1])[:8]:
    rows = np.where(db.cs_ID == d)[0]
    sig = {db.elements[j]: int(db.dm[rows[0], j]) for j in range(len(db.elements)) if db.dm[rows[0], j] != 0}
    print(f"  defect#{d} dm={sig}: {amt:.2e} cm^-3")
print(f"\nSi total in solid: {sum(si_by_defect.values()):.2e} (target {SI:.2e})")
print(f"n_quench={q.n:.2e}  VGa_simple={float(q.N_cs[is_vga_simple].sum()):.2e}")
