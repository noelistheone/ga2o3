"""Validate the transport layer against published single-crystal β-Ga2O3 Hall μ(n) data."""
import json
import sys
from pathlib import Path
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from sandbox import transport as tr

# Published RT Hall mobility vs carrier density (single-crystal/epi):
#  Ma/Neal/Bhattacharyya anchors: 196 @2.3e16 ; ~150 @1e18 ; 85 @1.2e20
anchors = [(2.3e16, 196), (1e17, 180), (1e18, 150), (1e19, 90), (1.2e20, 85)]
T = 300.0
rows = []
for n, mu_exp in anchors:
    # single-crystal: N_I ≈ n (uncompensated donor), screening n
    m = tr.mobility(T, n_cm3=n, N_I_cm3=n, film=False)
    sigma = float(tr.conductivity(n, m["mu_drift"]))
    rows.append({"n": f"{n:.1e}", "mu_exp_hall": mu_exp,
                 "mu_pred_hall": round(m["mu_hall"], 1),
                 "mu_Ma": round(m["mu_Ma_hall"], 1),
                 "mu_BH_II": round(m["mu_II_BH_hall"], 1),
                 "mu_drift": round(m["mu_drift"], 1), "rH": round(m["r_hall"], 2),
                 "sigma_S/cm": round(sigma, 2),
                 "rel_err": round(abs(m["mu_hall"] - mu_exp) / mu_exp, 2)})
out = {"anchors_vs_prediction": rows,
       "note": "single-crystal N_I=n; primary mu = Ma compensation-aware closed form",
       "max_rel_err": max(r["rel_err"] for r in rows),
       "PASS_within_35pct": all(r["rel_err"] < 0.35 for r in rows)}
(PROJ / "results/tier2").mkdir(parents=True, exist_ok=True)
(PROJ / "results/tier2/validate_transport.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
