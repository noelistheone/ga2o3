"""R9 review point 6: is the 'laboratory effect' actually a 'sample-type effect'?

Homogeneous-subset REML ICC. Estimator is COPIED VERBATIM from
  <repo>/Ga2O3-Sandbox/scripts/r8_icc_lrt.py
(intercept-only 1-D profiled REML; exact restricted LRT via 2000-sim parametric
bootstrap under H0; ICC CI via 2000-sim parametric bootstrap from the fitted model;
LOO-study influence; seed 2026).

Data of record (SAME as the full-corpus fit, reached through
Ga2O3-Net/scripts/_phase72_common.load_fom -> 154_phase70_eval.load_transport):
  <repo>/Ga2O3-Net/results/phase65/transport_llm_extracted_v2.csv
Grouping key = the 'file' column (source document), NOT 'doi' (112/549 rows have no doi).

READ-ONLY w.r.t. everything outside results/revision_r9/.
"""
from pathlib import Path as _P
_SB = _P(__file__).resolve().parents[1]          # <root>/Ga2O3-Sandbox
_NET = _SB.parent / "Ga2O3-Net"                  # sibling checkout / same release
import json, sys, warnings, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")
NET_CSV = Path(str(_NET) + "/results/phase65/transport_llm_extracted_v2.csv")
OUT = Path(str(_SB) + "/results/revision_r9")
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------- estimator: verbatim from scripts/r8_icc_lrt.py -------------------------
def suffstats(y, g):
    stats = []
    for grp in np.unique(g):
        v = y[g == grp]
        stats.append((len(v), float(v.mean()), float(((v - v.mean()) ** 2).sum())))
    return stats


def reml_fit(stats):
    N = sum(n for n, _, _ in stats)
    q = len(stats)

    def neg_reml(log_rho):
        rho = np.exp(log_rho)
        w = np.array([n / (1 + rho * n) for n, _, _ in stats])
        yb = np.array([m for _, m, _ in stats])
        mu = (w * yb).sum() / w.sum()
        Q = sum(ss for _, _, ss in stats) + (w * (yb - mu) ** 2).sum()
        s2e = Q / (N - 1)
        ll = -0.5 * (sum((n - 1) * np.log(s2e) + np.log(s2e * (1 + rho * n)) for n, _, _ in stats)
                     + Q / s2e + np.log((w / s2e).sum()))
        return -ll

    r = minimize_scalar(neg_reml, bounds=(-14, 8), method="bounded", options={"xatol": 1e-7})
    rho = float(np.exp(r.x))
    w = np.array([n / (1 + rho * n) for n, _, _ in stats])
    yb = np.array([m for _, m, _ in stats])
    mu = (w * yb).sum() / w.sum()
    Q = sum(ss for _, _, ss in stats) + (w * (yb - mu) ** 2).sum()
    s2e = Q / (N - 1)
    s2b = rho * s2e
    if rho < 2e-6:
        s2b = 0.0
    icc = s2b / (s2b + s2e) if (s2b + s2e) > 0 else 0.0
    return {"mu": float(mu), "s2b": float(s2b), "s2e": float(s2e), "icc": float(icc),
            "neg_reml": float(r.fun)}


def reml_ll_null(stats):
    N = sum(n for n, _, _ in stats)
    yb = np.array([m for _, m, _ in stats])
    n = np.array([x for x, _, _ in stats])
    mu = (n * yb).sum() / N
    Q = sum(ss for _, _, ss in stats) + (n * (yb - mu) ** 2).sum()
    s2 = Q / (N - 1)
    ll = -0.5 * (N * np.log(s2) + Q / s2 + np.log(N / s2))
    return float(ll)


def simulate(stats, mu, s2b, s2e, rng):
    out = []
    for n, _, _ in stats:
        b = rng.normal(0, np.sqrt(s2b)) if s2b > 0 else 0.0
        v = mu + b + rng.normal(0, np.sqrt(s2e), n)
        out.append((n, float(v.mean()), float(((v - v.mean()) ** 2).sum())))
    return out


NSIM = 2000
rng = np.random.default_rng(2026)


def analyze(name, y, g):
    y = np.asarray(y, float)
    g = np.asarray(g)
    ok = np.isfinite(y)
    y, g = y[ok], g[ok]
    stats = suffstats(y, g)
    fit = reml_fit(stats)
    lrt_obs = 2 * (-fit["neg_reml"] - reml_ll_null(stats))
    null = np.empty(NSIM)
    for i in range(NSIM):
        st = simulate(stats, fit["mu"], 0.0, fit["s2b"] + fit["s2e"], rng)
        f1 = reml_fit(st)
        null[i] = 2 * (-f1["neg_reml"] - reml_ll_null(st))
    p_lrt = float((np.sum(null >= max(lrt_obs, 0)) + 1) / (NSIM + 1))
    boot = np.empty(NSIM)
    for i in range(NSIM):
        st = simulate(stats, fit["mu"], fit["s2b"], fit["s2e"], rng)
        boot[i] = reml_fit(st)["icc"]
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    sizes = np.array([n for n, _, _ in stats])
    iccs = []
    for j in range(len(stats)):
        iccs.append(reml_fit(stats[:j] + stats[j + 1:])["icc"])
    dmax = float(np.max(np.abs(np.array(iccs) - fit["icc"])))
    r = {"n_rows": int(len(y)), "n_studies": int(len(stats)),
         "icc_reml": round(fit["icc"], 4), "lrt_stat": round(float(max(lrt_obs, 0)), 3),
         "p_exact_restricted_lrt": p_lrt, "icc_ci95_parametric_bootstrap": [round(c, 3) for c in ci],
         "singleton_fraction": round(float((sizes == 1).mean()), 3),
         "median_cluster_size": float(np.median(sizes)),
         "loo_study_max_abs_dicc": round(dmax, 4),
         "mc_se_at_p": round(float(np.sqrt(p_lrt * (1 - p_lrt) / NSIM)), 5),
         "sigma_b2": round(fit["s2b"], 4), "sigma_e2": round(fit["s2e"], 4)}
    print(name, json.dumps(r), flush=True)
    return r


# ------------------------- sample-type classifier (rule-based, ordered) -------------------------
def classify(txt):
    """Row-level sample-type label from the free-text growth_method field.
    Order matters: implantation first (post-growth processing dominates the sample state),
    then deposition families, then melt/bulk, then unknown."""
    t = str(txt).strip().lower()
    if t in ("", "nan", "na", "none"):
        return "unknown_no_method_text"
    if ("implant" in t or "ion impl" in t) and "not implanted" not in t:
        return "implanted"
    # --- vapour/solution deposition families ---
    if "pecvd" in t or "plasma-enhanced chemical vapor" in t or "plasma-enhanced chemical vapour" in t:
        return "pecvd_film"
    if re.search(r"\bald\b|atomic layer deposition", t):
        return "ald_film"
    if "sputter" in t or "magnetron" in t:
        return "sputtered_film"
    if ("pulsed laser deposition" in t or re.search(r"\bpld\b", t)
            or "pulsed electron deposition" in t
            or "electron beam evaporation" in t or "e-beam) evaporation" in t
            or "electron beam (e-beam) evaporation" in t):
        return "pvd_other_film"
    if ("mocvd" in t or "movpe" in t or "metal organic chemical vapour" in t
            or "metalorganic vapor" in t or "metal-organic vapor" in t
            or "mocataxy" in t):
        return "epi_mocvd_movpe"
    if "lpcvd" in t or "low-pressure chemical vapor" in t:
        return "epi_lpcvd"
    if "mist" in t:
        return "epi_mistcvd"
    if "hvpe" in t or "halide vapor phase" in t:
        return "epi_hvpe"
    if re.search(r"\bmbe\b|molecular beam epitaxy|pambe|pa-mbe", t):
        return "epi_mbe"
    # --- melt-grown bulk single crystals ---
    if ("czochralski" in t or re.search(r"\bcz\b|\bcz/", t) or re.search(r"\bvgf\b", t)
            or "vertical gradient" in t
            or re.search(r"\befg\b", t) or "edge-defined" in t or "edge defined" in t
            or "floating zone" in t or "float zone" in t or re.search(r"\bofz\b", t)
            or "melt growth" in t or "melt-grown" in t
            or "bulk single crystal" in t or "single-crystal wafer" in t
            or "bulk substrate" in t
            or ("single crystal" in t and ("tamura" in t or "commercial" in t or "wafer" in t))):
        return "bulk_melt_single_crystal"
    # --- exotic / other ---
    if "exfoliation" in t:
        return "exfoliated_flake"
    if "thermal oxidation" in t or "phase transition" in t:
        return "converted_from_gan"
    return "unknown_measurement_text_only"


FAMILY = {
    "bulk_melt_single_crystal": "bulk",
    "epi_mocvd_movpe": "epi", "epi_lpcvd": "epi", "epi_mistcvd": "epi",
    "epi_hvpe": "epi", "epi_mbe": "epi",
    "sputtered_film": "pvd", "pvd_other_film": "pvd",
    "pecvd_film": "other_film", "ald_film": "other_film",
    "implanted": "implanted",
    "exfoliated_flake": "exotic", "converted_from_gan": "exotic",
    "unknown_no_method_text": "unknown", "unknown_measurement_text_only": "unknown",
}

# ------------------------- load exactly as load_fom does -------------------------
raw = pd.read_csv(NET_CSV)


def _grp(series):
    """verbatim from Ga2O3-Net/scripts/154_phase70_eval.py::_grp"""
    s = series.astype(str)
    return np.array([d if d not in ("nan", "NA", "None", "") else f"nodoi_{i}" for i, d in enumerate(s)])


raw["group"] = _grp(raw["file"])
raw["stype"] = [classify(v) for v in raw["growth_method"]]
raw["family"] = [FAMILY[s] for s in raw["stype"]]

# file-level backfill for unknown rows: if every *labelled* row in the file agrees, adopt it
backfill = {}
for f, gsub in raw.groupby("group"):
    lab = set(gsub.loc[~gsub["family"].isin(["unknown"]), "family"])
    if len(lab) == 1:
        backfill[f] = next(iter(lab))
raw["family_bf"] = [backfill.get(g, fam) if fam == "unknown" else fam
                    for g, fam in zip(raw["group"], raw["family"])]

PROPS = {"hall_mobility_mu": "mu_cm2Vs", "hall_carrier_n": "carrier_cm3"}
MIN_ROWS, MIN_STUDIES = 40, 15

audit = {}
results = {}
rejected = {}

for prop, col in PROPS.items():
    yr = pd.to_numeric(raw[col], errors="coerce")
    m = yr.notna() & (yr > 0)          # pos=True in the FOM registry
    sub = raw[m].reset_index(drop=True)
    y = np.log10(yr[m].values.astype(float))   # do_log=True
    g = sub["group"].values

    # --- reproduce full-corpus fit first (must match results/tier2/r8_icc_lrt.json) ---
    results.setdefault(prop, {})["FULL_CORPUS_reproduction"] = analyze(f"{prop}/FULL", y, g)

    # census
    cen = {}
    for fam in sorted(set(sub["family_bf"])):
        mm = (sub["family_bf"] == fam).values
        cen[fam] = {"n_rows": int(mm.sum()), "n_studies": int(pd.Series(g[mm]).nunique())}
    audit[prop] = {"family_census_after_backfill": cen,
                   "stype_census_strict": {k: int(v) for k, v in sub["stype"].value_counts().items()}}

    subsets = {
        "bulk_melt_single_crystal": (sub["family_bf"] == "bulk").values,
        "epitaxial_film_all": (sub["family_bf"] == "epi").values,
        "epitaxial_film_MOCVD_MOVPE_only": (sub["stype"] == "epi_mocvd_movpe").values,
        "epitaxial_film_MBE_only": (sub["stype"] == "epi_mbe").values,
        "epitaxial_film_LPCVD_only": (sub["stype"] == "epi_lpcvd").values,
        "pvd_sputter_pld_films": (sub["family_bf"] == "pvd").values,
        "implanted": (sub["family_bf"] == "implanted").values,
    }
    for nm, mm in subsets.items():
        nr = int(mm.sum()); ns = int(pd.Series(g[mm]).nunique())
        if nr < MIN_ROWS or ns < MIN_STUDIES:
            rejected.setdefault(prop, {})[nm] = {
                "n_rows": nr, "n_studies": ns,
                "reason": f"below power floor (need >={MIN_ROWS} rows AND >={MIN_STUDIES} studies)"}
            print(f"REJECT {prop}/{nm}: n_rows={nr} n_studies={ns}", flush=True)
            continue
        results[prop][nm] = analyze(f"{prop}/{nm}", y[mm], g[mm])

    # strict (no backfill) sensitivity for the two headline subsets
    for nm, mm in {"bulk_melt_single_crystal_STRICT_no_backfill": (sub["family"] == "bulk").values,
                   "epitaxial_film_all_STRICT_no_backfill": (sub["family"] == "epi").values}.items():
        nr = int(mm.sum()); ns = int(pd.Series(g[mm]).nunique())
        if nr < MIN_ROWS or ns < MIN_STUDIES:
            rejected.setdefault(prop, {})[nm] = {"n_rows": nr, "n_studies": ns,
                                                 "reason": "below power floor"}
            print(f"REJECT {prop}/{nm}: n_rows={nr} n_studies={ns}", flush=True)
            continue
        results[prop][nm] = analyze(f"{prop}/{nm}", y[mm], g[mm])

# per-study homogeneity: how often does a single source document mix sample types?
mixed = {}
for f, gsub in raw.groupby("group"):
    fams = set(gsub.loc[gsub["family_bf"] != "unknown", "family_bf"])
    if len(fams) > 1:
        mixed[f] = sorted(fams)
audit["_studies_mixing_sample_types"] = {"n_mixed": len(mixed), "n_total_files": int(raw["group"].nunique()),
                                         "examples": dict(list(mixed.items())[:10])}

out = {"_purpose": "R9 reviewer point 6: homogeneous-sample-type REML ICC vs full-corpus ICC",
       "_data_of_record": str(NET_CSV),
       "_grouping_key": "file (source document); doi rejected (112/549 rows blank -> pseudo-lab collapse)",
       "_estimator": ("intercept-only REML (1-D profiled), exact restricted LRT of sigma_b^2>0 via "
                      "2000-sim parametric bootstrap under H0 (Crainiceanu-Ruppert-style), ICC CI via "
                      "2000-sim parametric bootstrap from the fitted model, seed 2026 "
                      "-- functions copied verbatim from scripts/r8_icc_lrt.py"),
       "_power_floor": {"min_rows": MIN_ROWS, "min_studies": MIN_STUDIES},
       "_full_corpus_reference": str(_SB) + "/results/tier2/r8_icc_lrt.json",
       "results": results, "rejected_subsets": rejected, "audit": audit}
(OUT / "homogeneous_icc.json").write_text(json.dumps(out, indent=1))
print("wrote", OUT / "homogeneous_icc.json")
