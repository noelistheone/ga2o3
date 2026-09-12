"""run_designer — end-to-end demo of the per-lab experiment designer (the lab-usable lever).

Scenario: a lab has run a few characterisation experiments on its own bench. The designer:
  1. FITS the lab's per-lab calibration theta (freeze-in T, background donor) from those points,
  2. runs the NROY honesty gate (is the sim reconcilable with this bench at all?),
  3. RECOMMENDS the next K experiments (L-optimal toward the lab's stated TARGET) that most tighten
     the target prediction, showing the absolute-n band before vs after,
  4. emits the calibrated 9-property prediction for the target with genuinely-simulated bars.

Run: PYTHONPATH=src python scripts/run_designer.py
"""
import sys, json
from pathlib import Path
import numpy as np
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from sandbox import design, kroger_db

db = kroger_db.load()


def synth_lab_points(true_theta):
    """Stand in for a real lab's measured data: 3 characterisation runs at the lab's true state."""
    conds = [{"dopant": "undoped", "conc": 0.0, "film": True, "pO2": 1e-5, "obs_mask": ["log10_n"]},
             {"dopant": "Si", "conc": 0.003, "film": True, "pO2": 1e-5},
             {"dopant": "Si", "conc": 0.01, "film": True, "pO2": 1e-5}]
    pts = []
    for c in conds:
        y = design.observe(c, true_theta, db)
        meas = {"log10_n": float(y[0])}
        if c["dopant"] != "undoped":
            meas["log10_mu"] = float(y[1]); meas["log10_sigma"] = float(y[2])
        pts.append({**c, "meas": meas})
    return pts


def main():
    # the lab's (unknown-to-us) true process state
    true = {"log10_Tf": np.log10(1650.0), "E_B": 0.06, "log10_Nd_bg": np.log10(2e17)}
    lab_pts = synth_lab_points(true)
    target = {"dopant": "Sn", "conc": 0.01, "film": True, "pO2": 1e-5}   # what the lab wants to predict

    print("=" * 74)
    print("STEP 1 — fit per-lab theta from the lab's 3 characterisation runs")
    th, cov = design.theta_fit(lab_pts, db=db)
    std = np.sqrt(np.diag(cov))
    for i, k in enumerate(design.THETA_KEYS):
        val = 10 ** th[k] if k.startswith("log10") else th[k]
        print(f"   {k:12s} = {val:10.4g}   (posterior std {std[i]:.3f})")

    print("\nSTEP 2 — NROY / implausibility honesty gate")
    gate = design.implausibility(lab_pts, th, db)
    print(f"   max implausibility = {gate['max_implausibility']}  ->  {gate['verdict']}")
    if not gate["NROY_nonempty"]:
        print("   ABORT calibrated absolutes; report relative-only."); return

    print(f"\nSTEP 3 — recommend the next K=4 experiments (L-optimal toward target {target['dopant']} "
          f"{target['conc']*100:.0f}at%)")
    rec = design.design_battery(K=4, theta=th, base_points=lab_pts, criterion="L",
                                targets=[target], db=db)
    for c, g in zip(rec["battery"], rec["info_gain"]):
        print(f"   run: {c['dopant']:7s} {c.get('conc',0)*100:4.1f}at% "
              f"{'film' if c['film'] else 'xtal':4s} pO2={c['pO2']:.0e}   (Δcriterion {g})")
    print(f"   absolute-n band at target:  {rec['abs_n_dex_before']} dex  ->  {rec['abs_n_dex_after']} dex")

    print(f"\nSTEP 4 — calibrated 9-property prediction for {target['dopant']} {target['conc']*100:.0f}at% "
          f"(every value = V61 solver; bands from the per-lab posterior)")
    pu = design.predict_with_uncertainty(target, th, cov, db)
    for k, v in pu.items():
        if "band_dex" in v:
            print(f"   {k:20s} = {v['median']:.3e}   ± {v['band_dex']} dex")
        else:
            print(f"   {k:20s} = {v['median']:<10.4g} ± {v['band']}")
    print("=" * 74)
    print("Genuinely-simulated: theta is a physical per-lab process calibration; the ML/design layer "
          "only CHOOSES experiments and FITS theta. No table lookup enters the value.")


if __name__ == "__main__":
    main()
