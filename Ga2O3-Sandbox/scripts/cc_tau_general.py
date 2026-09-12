"""General CC→NMP→τ finalize for any defect. Usage:
    python cc_tau_general.py <defect> <qA> <qB> <dE_eV> <N_trap_cm3>
Reads the 4 CC energies (2 relax minima + 2 cross single-points) and runs the NMP→τ chain.
"""
import json
import re
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
from sandbox import tau_nmp  # noqa: E402

CC = PROJ / "dft/qe_cc"
RY = 13.605693


def E(prefix):
    p = CC / f"{prefix}.out"
    if not p.exists() or "JOB DONE" not in p.read_text():
        return None
    e = re.findall(r"^!\s+total energy\s+=\s+([-\d.]+)", p.read_text(), re.M)
    return float(e[-1]) * RY if e else None


def main():
    defect, qA, qB = sys.argv[1], sys.argv[2], sys.argv[3]
    dE = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8
    N_trap = float(sys.argv[5]) if len(sys.argv) > 5 else 1e17
    EAA = E(f"cc_{defect}_q{qA}_relax")           # qA at R_qA
    EBB = E(f"cc_{defect}_q{qB}_relax")           # qB at R_qB
    EAB = E(f"cc_{defect}_q{qA}_at_R{qB}")        # qA at R_qB
    EBA = E(f"cc_{defect}_q{qB}_at_R{qA}")        # qB at R_qA
    miss = [n for n, v in [("AA", EAA), ("BB", EBB), ("AB", EAB), ("BA", EBA)] if v is None]
    if miss:
        print(json.dumps({"waiting_for": miss}))
        return
    ccj = json.loads((PROJ / f"results/tier2/cc_{defect}.json").read_text())
    dQ = ccj["dQ_amu_half_A"]
    lamA = EAB - EAA
    lamB = EBA - EBB
    lam = 0.5 * (lamA + lamB)
    tau = tau_nmp.tau_from_cc(dQ=dQ, lam=abs(lam), dE=dE, N_trap_cm3=N_trap)
    result = {"defect": defect, "charges": f"{qA}/{qB}", "dQ_amu_half_A": dQ,
              "lambda_qA_eV": round(lamA, 3), "lambda_qB_eV": round(lamB, 3),
              "effective_lambda_eV": round(lam, 3), "driving_force_dE_eV": dE,
              "N_trap_cm3": N_trap, **tau}
    (PROJ / f"results/tier2/tau_{defect}_result.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
