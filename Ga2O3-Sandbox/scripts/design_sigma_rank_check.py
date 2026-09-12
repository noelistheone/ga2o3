"""R6 reviewer #1 item 10: sigma = e*n*mu is row-wise constructed, so (n, mu, sigma) carries only
two independent measurement channels. Re-run the identifiability battery of
design_sensitivity_probe.py and report the FIM spectrum / rank / null directions with the
observable set {n, mu, sigma} versus {n, mu} only. If the rank-2 result and the null directions
are unchanged, the identifiability conclusion is independent of the constructed channel.
Writes results/hybrid/design_sigma_rank_check.json."""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "probe", PROJ / "scripts/design_sensitivity_probe.py")
probe = importlib.util.module_from_spec(spec)
sys.modules["probe"] = probe
# prevent main() from running on import: exec module then use its functions
src = (PROJ / "scripts/design_sensitivity_probe.py").read_text()
src = src.replace('if __name__ == "__main__":', 'if False:')
exec(compile(src, "probe", "exec"), probe.__dict__)

THETA0 = probe.THETA0

conds = []
for dop in ["Si", "Sn", "Hf", "Ge"]:
    for conc in [0.001, 0.005, 0.02]:
        for film in [False, True]:
            conds.append({"dopant": dop, "conc": conc, "film": film})

Jstack = []
for c in conds:
    J = probe.jacobian(c["dopant"], c["conc"], c["film"])
    Jstack.append(J)
    print("J done", c, flush=True)


def spectrum(rows):
    Jall = np.vstack([J[rows, :] for J in Jstack])
    FIM = Jall.T @ Jall
    eig, vec = np.linalg.eigh(FIM)
    rank = int(np.sum(eig > 1e-6 * eig[-1]))
    return FIM, eig, vec, rank


F3, e3, v3, r3 = spectrum([0, 1, 2])   # n, mu, sigma
F2, e2, v2, r2 = spectrum([0, 1])      # n, mu only

# null-space overlap: subspace spanned by the two smallest eigvectors in each case
N3 = v3[:, :2]
N2 = v2[:, :2]
principal = np.linalg.svd(N3.T @ N2, compute_uv=False)   # cosines of principal angles

out = {
    "theta_order": list(THETA0.keys()),
    "obs_n_mu_sigma": {"eigenvalues": [float(f"{x:.4g}") for x in e3],
                       "rank_1e-6rel": r3,
                       "condition_number": float(f"{e3[-1]/max(e3[0],1e-12):.3g}")},
    "obs_n_mu_only": {"eigenvalues": [float(f"{x:.4g}") for x in e2],
                      "rank_1e-6rel": r2,
                      "condition_number": float(f"{e2[-1]/max(e2[0],1e-12):.3g}")},
    "null_space_principal_cosines": [round(float(x), 4) for x in principal],
    "read": "if ranks match and principal cosines ~1, the rank-2 degeneracy and its null "
            "directions (pO2, Nd_bg) are identical with or without the constructed sigma channel",
}
(PROJ / "results/hybrid/design_sigma_rank_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
