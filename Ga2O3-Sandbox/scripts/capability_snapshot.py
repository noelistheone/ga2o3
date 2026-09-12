"""Current-engine capability snapshot: all 17 dopants at lab-like process conditions.

For each dopant X: fix total [X]=1e19 cm^-3, equilibrate at T_anneal=1073 K (lab 800 C) with
pO2=1e-4 atm (Ar-ambient placeholder nuisance — to be FIT in Tier 2, not asserted), quench to
300 K. Emit everything the Tier-1 engine natively produces per the nine ML properties:

  native now : [V_O], hall_n (and p, E_F, compensation, free V_Ga, dominant defects)
  Tier 2     : mu (Ma+BH+Seto), sigma=e n mu, Eg/dEg (Burstein-Moss), tau/dark/PDR direction flags

Output: results/phase0/capability_snapshot.json + printed table.
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from sandbox import kroger_db, engine  # noqa: E402

db = kroger_db.load()
T_ANNEAL, PO2, CONC = 1073.0, 1e-4, 1e19

c_ga, c_o = db.col("Ga"), db.col("O")
is_vo = (db.dm[:, c_o] == -1) & (np.abs(db.dm).sum(axis=1) == 1)
is_vga = (db.dm[:, c_ga] == -1) & (np.abs(db.dm).sum(axis=1) == 1)

rows = []
dopants = [e for e in db.elements if e not in ("Ga", "O")]
for X in ["undoped"] + dopants:
    try:
        fixed = {} if X == "undoped" else {X: CONC}
        eq = engine.solve_single(db, T_ANNEAL, pO2=PO2, EcT_fraction=0.40,
                                 fixed_conc=fixed, fd=True)
        q = engine.quench(db, eq, T_quench=300.0, EcT_fraction=0.40, fd=True)
        Nd, Na = engine.net_acceptor_donor(db, q)
        # dominant defect at 300 K (largest single concentration)
        top = np.argsort(q.N_defect)[::-1][:2]
        dom = [f"defect#{d+1}:{q.N_defect[d]:.1e}" for d in top if q.N_defect[d] > 1e10]
        incorporated = None
        if X != "undoped":
            incorporated = float((db.dm[:, db.col(X)] * eq.N_cs).sum())
        rows.append({
            "dopant": X,
            "conc_fixed": None if X == "undoped" else CONC,
            "solubility_limited": eq.meta.get("solubility_limited", False),
            "log10_incorporated": None if incorporated is None else round(np.log10(max(incorporated, 1.0)), 2),
            "solver_charge_residual_rel": abs(eq.charge_residual) / max(eq.n, 1e10),
            "EF_300K_belowEc": round(q.Ec - q.EF, 3),
            "log10_n_300K": round(np.log10(max(q.n, 1.0)), 2),
            "log10_p_300K": round(np.log10(max(q.p, 1e-30)), 2),
            "log10_VO_anneal": round(np.log10(max(float(eq.N_cs[is_vo].sum()), 1.0)), 2),
            "log10_VGa_free": round(np.log10(max(float(q.N_cs[is_vga].sum()), 1.0)), 2),
            "ionized_Na_300K": f"{Na:.2e}",
            "type": ("n" if q.n > 10 * q.p and q.n > 1e10 else
                     ("insulating" if max(q.n, q.p) < 1e10 else "p-ish/compensated")),
            "dominant_defects": dom,
        })
    except Exception as e:
        rows.append({"dopant": X, "error": str(e)[:120]})

snapshot = {
    "conditions": {"T_anneal_K": T_ANNEAL, "pO2_atm_placeholder": PO2, "dopant_conc_cm3": CONC,
                   "quench_K": 300.0, "EcT_fraction": 0.40,
                   "note": "pO2 for Ar ambient is a PLACEHOLDER nuisance parameter here; "
                           "Tier 2 fits it on the lab's paired intrinsic references"},
    "nine_properties_coverage_now": {
        "V_O": "NATIVE (this table; also per-site and complexed V_O available)",
        "hall_n": "NATIVE (this table; quenched 300 K free-carrier density)",
        "hall_mu": "Tier 2: Ma mu_H(N_D,T)+Brooks-Herring(N_I from this solver)+Seto GB term",
        "sigma": "Tier 2: e*n*mu composition",
        "Eg": "Tier 2: Burstein-Moss from degenerate n + renormalization sign",
        "dEg": "Tier 2: doped-minus-undoped BM shift (this table already gives both n's)",
        "tau": "Tier 2: DIRECTION-ONLY flag from deep-trap totals (quantitative refused)",
        "dark": "Tier 2: activation energy from EF vs trap catalog + order-of-magnitude rho",
        "PDR": "Tier 2: DIRECTION-ONLY flag (deep-acceptor hole-trap density lever)",
    },
    "rows": rows,
}
(PROJ / "results/phase0/capability_snapshot.json").write_text(json.dumps(snapshot, indent=2))

print(f"{'dopant':9s} {'type':12s} {'Ec-EF':>6s} {'log n':>6s} {'logVO':>6s} {'logVGa':>6s}  dominant")
for r in rows:
    if "error" in r:
        print(f"{r['dopant']:9s} ERROR {r['error']}")
        continue
    print(f"{r['dopant']:9s} {r['type']:12s} {r['EF_300K_belowEc']:6.2f} {r['log10_n_300K']:6.2f} "
          f"{r['log10_VO_anneal']:6.2f} {r['log10_VGa_free']:6.2f}  {';'.join(r['dominant_defects'][:1])}")
