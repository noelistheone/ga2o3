import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sandbox import kroger_db, engine

db = kroger_db.load()
print("DB:", db.nCS, "cs,", db.nD, "defects")
for T, pO2 in [(2068, 0.02), (1400, 1.0), (1000, 1e-4), (1073, 1e-6)]:
    r = engine.solve_single(db, T, pO2=pO2, EcT_fraction=0.40, fd=False)
    Nd, Na = engine.net_acceptor_donor(db, r)
    print(f"T={T} pO2={pO2:.0e}: EF={r.EF:.3f} (Ec={r.Ec:.2f}) n={r.n:.2e} p={r.p:.2e} "
          f"sth={r.sth1 + r.sth2:.2e} Nd+={Nd:.2e} Na-={Na:.2e} resid={r.charge_residual:.1e}")
