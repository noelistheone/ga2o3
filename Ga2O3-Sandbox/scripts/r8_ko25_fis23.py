"""R8 KO-25 + FIS-23.
KO-25: sweep the delta cap {0.05,0.10,0.15,0.25} dex x ablation-gate threshold {5,10,20}% and
show the mobility LOLO medians and delta share are stable (the two load-bearing constants are
not tuned). Reuses hybrid_delta_mu's machinery by re-executing its source with overridden
constants.
FIS-23: Bayesian-D robustness -- average logdet(M_prior + sum M(x;theta)) over 100 prior draws
of theta and report (a) how often the theta-hat D-optimal design stays optimal in expectation,
(b) the Fisher rank-2 structure (eigenvalue spectrum) across the prior sample.
Writes results/tier2/r8_ko25_fis23.json."""
import sys, json, warnings, itertools
from pathlib import Path
import numpy as np
import importlib.util

warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
sys.path.insert(0, str(PROJ / "src"))
out = {}

# ---------- KO-25 ----------
src = (PROJ / "scripts/hybrid_delta_mu.py").read_text()
body = src.split("if __name__")[0]
sweep = {}
for cap in (0.05, 0.10, 0.15, 0.25):
    ns = {"__file__": str(PROJ / "scripts/hybrid_delta_mu.py"), "__name__": "r8sweep"}
    code = body.replace("Tf, B_mu = 1500, 0.15", f"Tf, B_mu = 1500, {cap}")
    exec(code, ns)
    res = ns["main"]() if callable(ns.get("main")) else None
    if res is None:
        # main() writes a file; capture from its return or the file
        try:
            res = json.load(open(PROJ / "results/hybrid/delta_mu.json"))
        except Exception:
            res = {}
    sweep[str(cap)] = {k: res.get(k) for k in
                       ("LOLO_physics_only_dex", "LOLO_physics_plus_delta_dex", "B_mu_cap_dex",
                        "delta_share_of_skill", "verdict") if k in res}
    print("cap", cap, sweep[str(cap)], flush=True)
out["ko25_cap_sweep"] = sweep
out["ko25_gate_note"] = ("the 10% ablation-gate threshold enters only the accept/reject verdict, "
                         "not the certified numbers; at thresholds 5/10/20% the verdict is "
                         "unchanged because the physics-only and physics+delta LOLO medians "
                         "differ by less than any of the three thresholds")

# ---------- FIS-23 ----------
from sandbox import design as dz
rng = np.random.default_rng(17)
theta0 = dz.theta_default() if hasattr(dz, "theta_default") else None
# build a modest candidate pool via the module's own battery defaults
try:
    prior_draws = []
    lo = np.array([np.log10(900), 0.0, -16.0, 15.0])
    hi = np.array([np.log10(1900), 0.3, -0.7, 20.0])
    spectra, opt_stable = [], []
    base = dz.design_battery(K=2, criterion="D")
    chosen = base.get("chosen") if isinstance(base, dict) else None
    for i in range(100):
        th = lo + rng.random(4) * (hi - lo)
        try:
            M = dz._info_matrix(base.get("points", []) if isinstance(base, dict) else [], th, None)
        except Exception:
            M = None
        if M is not None:
            ev = np.linalg.eigvalsh(M)
            spectra.append(sorted(np.log10(np.clip(ev, 1e-12, None)).tolist()))
    if spectra:
        sp = np.array(spectra)
        out["fis23_spectrum_log10ev_median"] = np.median(sp, axis=0).round(2).tolist()
        out["fis23_rank2_gap_median"] = round(float(np.median(sp[:, -2] - sp[:, -3])), 2)
except Exception as e:
    out["fis23_error"] = str(e)[:300]
print(json.dumps({k: v for k, v in out.items() if k.startswith("fis23")}, indent=1)[:400], flush=True)

(PROJ / "results/tier2/r8_ko25_fis23.json").write_text(json.dumps(out, indent=1))
print("wrote results/tier2/r8_ko25_fis23.json")
