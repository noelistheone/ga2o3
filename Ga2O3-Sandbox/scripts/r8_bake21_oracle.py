"""R8 BAKE-21 (oracle tightness): if the study identity were known, does accuracy rise to the
reliability limit? Row-level (non-grouped) 5-fold CV with a train-only study-mean feature
appended: the oracle pooled rho should approach sqrt(ICC) (the between-study share is real,
recoverable information -- just unavailable across studies), demonstrating the attenuation
bound is BINDING rather than a loose inequality. Writes results/tier2/r8_bake21_oracle.json."""
import sys, json, warnings
from pathlib import Path
import numpy as np
import importlib.util
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parents[1]
NET = PROJ.parent / "Ga2O3-Net"
sys.path.insert(0, str(NET / "scripts"))
spec = importlib.util.spec_from_file_location("c72", NET / "scripts/_phase72_common.py")
c72 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c72)

FOMS = ["photo_dark_ratio", "dark_current_pA", "vacancy_concentration", "hall_carrier_n",
        "hall_mobility_mu", "optical_bandgap_Eg", "bandgap_shift_dEg", "tau_decay"]
icc = json.loads((PROJ / "results/tier2/r8_icc_lrt.json").read_text())
rho_adopted = json.loads((NET / "results/phase79/adopted_rho_phase79.json").read_text())

out = {}
for fom in FOMS:
    d = c72.load_fom(fom)
    y = np.asarray(d["y"], float)
    g = np.asarray(d["groups"]).astype(str)
    F = np.asarray(d["F"], float)
    ok = np.isfinite(y) & np.all(np.isfinite(F), axis=1)
    y, g, F = y[ok], g[ok], F[ok]
    preds = np.full(len(y), np.nan)
    for seed in range(5):
        kf = KFold(5, shuffle=True, random_state=seed)
        p = np.full(len(y), np.nan)
        for tr, te in kf.split(y):
            gm = {}
            for gr in np.unique(g[tr]):
                gm[gr] = y[tr][g[tr] == gr].mean()
            grand = y[tr].mean()
            f_tr = np.array([gm[x] for x in g[tr]])
            f_te = np.array([gm.get(x, grand) for x in g[te]])
            Xtr = np.column_stack([F[tr], f_tr])
            Xte = np.column_stack([F[te], f_te])
            sc = StandardScaler().fit(Xtr)
            m = Ridge(alpha=1.0).fit(sc.transform(Xtr), y[tr])
            p[te] = m.predict(sc.transform(Xte))
        preds = np.nanmean(np.vstack([preds, p]), axis=0) if seed else p
    rho_o = float(stats.spearmanr(preds, y).statistic)
    a = rho_adopted[fom]
    a = a if isinstance(a, float) else a.get("rho")
    ic = icc[fom]["icc_reml"]
    out[fom] = {"rho_oracle_studyID": round(rho_o, 3),
                "rho_adopted_grouped": a,
                "sqrt_ICC": round(float(np.sqrt(ic)), 3),
                "bound_sqrt_1_minus_ICC": round(float(np.sqrt(1 - ic)), 3)}
    print(fom, out[fom], flush=True)

(PROJ / "results/tier2/r8_bake21_oracle.json").write_text(json.dumps(out, indent=1))
print("wrote results/tier2/r8_bake21_oracle.json")
