"""R6 reviewer #1 item 2: DIRECT per-property test of the attenuation bound's orthogonality
assumption. For each of the nine properties: fit intercept-only REML (y ~ 1 + (1|study)),
extract BLUP study effects b_j, aggregate the observable feature matrix F to study level
(means), and measure how much of Var(b_j) observable covariates can explain — ridge with
leave-one-study-out CV (R2_cv, clipped at 0) plus a permutation null (999 shuffles of b_j).
ICC_perp = ICC * (1 - max(R2_cv, 0)); corrected bound rho <= sqrt(1 - ICC_perp).
Uses the Net corpus loaders READ-ONLY. Writes results/tier2/icc_orthogonality.json."""
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import importlib.util

PROJ = Path(__file__).resolve().parents[1]
NETS = "/home/lawrence/Physics/Ga2O3-Net/scripts"
sys.path.insert(0, NETS)


def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


c72 = _imp("c72", f"{NETS}/_phase72_common.py")
p109 = _imp("p109", f"{NETS}/phase109_conductivity.py")
import statsmodels.formula.api as smf
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(42)


def loaders():
    for fom in c72.FOMS:
        yield fom, c72.load_fom(fom)
    yield "conductivity_sigma", p109.load_conductivity()


def r2_cv(Xs, b):
    loo = LeaveOneOut()
    pred = np.zeros_like(b)
    for tr, te in loo.split(Xs):
        m = Ridge(alpha=1.0).fit(Xs[tr], b[tr])
        pred[te] = m.predict(Xs[te])
    ss = np.sum((b - pred) ** 2) / np.sum((b - b.mean()) ** 2)
    return 1.0 - ss


out = {}
for fom, d in loaders():
    if d is None:
        continue
    F = np.asarray(d["F"], float)
    y = np.asarray(d["y"], float)
    g = np.asarray(d["groups"]).astype(str)
    ok = np.isfinite(y) & np.all(np.isfinite(F), axis=1)
    F, y, g = F[ok], y[ok], g[ok]
    df = pd.DataFrame({"y": y, "study": g})
    if df.study.nunique() < 15:
        continue
    m0 = smf.mixedlm("y~1", df, groups=df["study"]).fit(reml=True)
    icc = float(m0.cov_re.iloc[0, 0] / (m0.cov_re.iloc[0, 0] + m0.scale))
    re = m0.random_effects
    studies = sorted(df.study.unique())
    b = np.array([float(np.asarray(re[s]).ravel()[0]) for s in studies])
    # study-level covariates = mean of observable features
    Fs = np.vstack([F[g == s].mean(axis=0) for s in studies])
    keep = Fs.std(axis=0) > 1e-12
    Xs = StandardScaler().fit_transform(Fs[:, keep])
    r2 = r2_cv(Xs, b)
    # permutation null on the CV R2
    null = np.array([r2_cv(Xs, RNG.permutation(b)) for _ in range(200)])
    p_perm = float((np.sum(null >= r2) + 1) / (len(null) + 1))
    r2c = max(r2, 0.0)
    icc_perp = icc * (1 - r2c)
    out[fom] = {"n_rows": int(len(y)), "n_studies": len(studies), "n_covariates": int(keep.sum()),
                "reml_icc": round(icc, 3), "r2_cv_covariates_on_blup": round(float(r2), 3),
                "p_perm": round(p_perm, 4), "icc_perp": round(icc_perp, 3),
                "bound_icc": round(float(np.sqrt(1 - icc)), 3),
                "bound_icc_perp": round(float(np.sqrt(1 - icc_perp)), 3)}
    print(fom, out[fom], flush=True)

(PROJ / "results/tier2/icc_orthogonality.json").write_text(json.dumps(out, indent=1))
print("wrote results/tier2/icc_orthogonality.json")
