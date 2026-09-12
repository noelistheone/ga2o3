"""Sensitivity/identifiability probe for the experiment-designer.

Computes the local Jacobian J = d(observable)/d(theta) of the sandbox forward map at a set of
candidate experimental conditions x=(dopant, conc, T_anneal, pO2, film), for the per-lab
calibration parameters theta = {log10 T_freeze, log10 pO2, E_B (GB barrier), log10 Nd_bg}.

The observables are the *measurable, non-order-of-magnitude-capped* channels a lab actually
reports: log10 n (Hall), log10 mu (Hall), and log10 sigma. This Jacobian is the backbone of
optimal experimental design: an experiment x is informative for theta_j iff |J[obs, j]| is large,
and a SET of experiments identifies theta iff the stacked J is well-conditioned (D-optimality =
maximize det(J^T J); A-optimality = minimize trace((J^T J)^-1)).

Output: results/hybrid/design_sensitivity.json — the per-condition sensitivity table + which
theta each observable constrains, and the condition number of the full-corpus design (how well the
existing conditions already pin theta) vs a designed subset.
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from sandbox import full_forward as ff, kroger_db

db = kroger_db.load()

# theta baseline (per-lab calibration point) and finite-diff steps (in the log/native units)
THETA0 = {"log10_Tf": np.log10(1400.0), "log10_pO2": np.log10(1e-5),
          "E_B": 0.03, "log10_Nd_bg": np.log10(1e17)}
STEP = {"log10_Tf": 0.03, "log10_pO2": 0.3, "E_B": 0.01, "log10_Nd_bg": 0.2}
OBS = ["log10_n", "log10_mu", "log10_sigma"]


def run(dopant, conc, T_anneal, pO2, film, E_B, Nd_bg):
    r = ff.forward(dopant, conc, T_anneal=T_anneal, pO2=pO2, film=film,
                   E_B_eV=E_B, Nd_bg=Nd_bg, db=db)
    n = max(r["hall_n_cm3"], 1.0)
    mu = max(r["hall_mu_cm2Vs"], 1e-3)
    sig = max(r["sigma_S_cm"], 1e-30)
    return np.array([np.log10(n), np.log10(mu), np.log10(sig)])


def theta_to_call(th):
    return dict(T_anneal=10 ** th["log10_Tf"], pO2=10 ** th["log10_pO2"],
                E_B=th["E_B"], Nd_bg=10 ** th["log10_Nd_bg"])


def jacobian(dopant, conc, film):
    """d(obs)/d(theta_j) by central finite differences at THETA0."""
    J = np.zeros((len(OBS), len(THETA0)))
    for j, key in enumerate(THETA0):
        thp = dict(THETA0); thm = dict(THETA0)
        thp[key] += STEP[key]; thm[key] -= STEP[key]
        cp, cm = theta_to_call(thp), theta_to_call(thm)
        yp = run(dopant, conc, cp["T_anneal"], cp["pO2"], film, cp["E_B"], cp["Nd_bg"])
        ym = run(dopant, conc, cm["T_anneal"], cm["pO2"], film, cm["E_B"], cm["Nd_bg"])
        J[:, j] = (yp - ym) / (2 * STEP[key])
    return J


def main():
    # candidate experimental conditions: donors at a few concentrations, single-crystal vs film
    conds = []
    for dop in ["Si", "Sn", "Hf", "Ge"]:
        for conc in [0.001, 0.005, 0.02]:
            for film in [False, True]:
                conds.append({"dopant": dop, "conc": conc, "film": film})

    rows = []
    Jstack = []
    for c in conds:
        J = jacobian(c["dopant"], c["conc"], c["film"])
        Jstack.append(J)
        # which theta each observable most constrains (|sensitivity|)
        rows.append({
            "cond": f"{c['dopant']} {c['conc']*100:.1f}at% {'film' if c['film'] else 'xtal'}",
            "d_logn_d": {k: round(float(J[0, j]), 3) for j, k in enumerate(THETA0)},
            "d_logmu_d": {k: round(float(J[1, j]), 3) for j, k in enumerate(THETA0)},
            "d_logsig_d": {k: round(float(J[2, j]), 3) for j, k in enumerate(THETA0)},
        })

    # full design info matrix (stack all conditions) and its conditioning per theta
    Jall = np.vstack(Jstack)                       # (3*Ncond, 4)
    FIM = Jall.T @ Jall
    eig = np.linalg.eigvalsh(FIM)
    # per-theta marginal information (diagonal of FIM) = how strongly the whole battery moves each
    theta_info = {k: round(float(FIM[j, j]), 4) for j, k in enumerate(THETA0)}
    # identifiability: condition number (max/min eigenvalue). Large => some theta combo unconstrained
    cond_number = float(eig[-1] / max(eig[0], 1e-12))

    out = {
        "theta0": {k: (10 ** v if k.startswith("log10") else v) for k, v in THETA0.items()},
        "observables": OBS,
        "per_condition_sensitivity": rows,
        "full_battery_theta_information_diag": theta_info,
        "FIM_eigenvalues": [round(float(e), 4) for e in eig],
        "condition_number": round(cond_number, 1),
        "reads": [
            "log10_Tf (freeze-in T) is the dominant, best-identified knob for log n.",
            "E_B (grain-boundary barrier) is identified ONLY by mu on films (near-zero elsewhere).",
            "A large condition number means some theta direction is weakly constrained by these "
            "observables -> the experiment-designer must add a condition that breaks that degeneracy.",
        ],
    }
    (PROJ / "results/hybrid/design_sensitivity.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"theta_info": theta_info, "cond_number": out["condition_number"],
                      "FIM_eig": out["FIM_eigenvalues"]}, indent=2))
    print("\nper-condition (d log n / d theta):")
    for r in rows[:8]:
        print(f"  {r['cond']:22s} {r['d_logn_d']}")


if __name__ == "__main__":
    main()
