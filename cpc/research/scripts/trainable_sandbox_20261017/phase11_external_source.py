"""Retrospective, input-conditional transfer to PRB 110, 174106 (2024).

All selections/fits use the original five sources. New labels appear only in
the explicitly allowed source anchors and the final evaluation, never fitting
the shared model or selecting a growth-temperature scenario.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys, json, shutil, argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from csv_io_v2 import read_result_csv
import phase11_prediction as study
import phase2_energy as energy

PARENT = energy.BASE / "phase11_parent_allocation"
OUT = PARENT / "external_PRB2024"
TRAIN = OUT / "model_training"


def weighted_quantile(values, weights, q):
    order = np.argsort(values, kind="stable")
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    i = np.searchsorted(np.cumsum(weights) / np.sum(weights), q, side="left")
    return float(values[min(i, len(values) - 1)])


def freeze():
    assert (PARENT / "contract_audit.json").exists(), "Finish/audit parent training first"
    paths = [Path(__file__), OUT / "pre_table_protocol.json", OUT / "paired_samples.csv",
             OUT / "source_provenance.json", PARENT / "prediction_protocol.json",
             PARENT / "contract_audit.json"]
    payload = dict(created_utc=energy.now(), version=1,
        timing="retrospective; exact table already read; pre-table protocol retained; not blind or preregistered",
        hashes={str(p.relative_to(ROOT)): energy.sha(p) for p in paths},
        selection="each method/config/source-offset lambda selected on the original five source-held-out folds; source equal MAE; no new-source labels",
        fit="fit selected configurations on all original29 RT specimens and original paired temperature labels as defined by method; new labels excluded",
        growth="training RT source-equal weighted quantiles0/.25/.5/.75/1; .5 primary; all scenarios retained without label-dependent selection",
        anchors="ascending chemical dose, lowest then middle; evaluate remaining queries and fixed highest-dose query",
        controls="identity n=C; identity plus unregularized source offset; these elementary references receive the same 0/1/2 anchors",
        scope="three reported conditions/one new source; missing growth/pO2/H and unmodeled EFG process, rH=1; not a full-input or new-element validation")
    path = OUT / "evaluation_protocol.json"
    if path.exists():
        assert json.loads(path.read_text())["hashes"] == payload["hashes"]
    else:
        energy.write(path, payload)
    TRAIN.mkdir(parents=True, exist_ok=True)
    dst = TRAIN / "prediction_protocol.json"
    if not dst.exists(): shutil.copyfile(PARENT / "prediction_protocol.json", dst)
    assert energy.sha(dst) == energy.sha(PARENT / "prediction_protocol.json")


def original_cv_selections():
    fits = [json.loads(p.read_text()) for p in (PARENT / "fits").glob("*.json")]
    lookup = {(f["method"], tuple(f["train_families"]), json.dumps(f["config"], sort_keys=True)): f for f in fits}
    choices = []
    for method in study.METHODS:
        for budget in (0, 1, 2):
            candidates = []
            for i, config in enumerate(study.grids(method)):
                canonical, cc = study.canonical(method, config)
                for lam in ([np.inf] if budget == 0 else [.1, 1., 10., np.inf]):
                    scores, keys = [], []
                    for held in study.FAMILIES:
                        families = tuple(x for x in study.FAMILIES if x != held)
                        f = lookup[canonical, families, json.dumps(energy.clean(cc), sort_keys=True)]
                        scores.append(study.evaluate(f, held, budget, lam)[0]); keys.append(f["key"])
                    candidates.append(dict(index=i, config=config, lam=lam,
                        score=float(np.mean(scores)), per_source_scores=scores, source_excluded_fit_keys=keys))
            best = min(candidates, key=lambda x: (round(x["score"], 10), -x["lam"], x["index"]))
            choices.append(dict(method=method, budget=budget, candidates=candidates, selected=best))
    energy.write(OUT / "training_only_selections.json", choices)
    return choices


def fit_one(args):
    study.OUT = TRAIN
    method, config = args
    return study.fit(study.FAMILIES, method, config)


def raw_predict(fit, rows):
    field = study.METHODS[fit["method"]]["field"]
    meta = fit["feature_metadata"]
    if field == "ridge":
        vals = (study.prior.features(rows) - np.array(meta["center"])) / np.array(meta["scale"])
        return vals @ np.array(meta["coefficients"]) + meta["intercept"], [None] * len(rows)
    if field == "dose_line":
        p = meta["coefficients"]
        return p[0] + p[1] * (np.log10(rows.incorporated_cm3) - 18), [None] * len(rows)
    if field == "activation":
        return study.activation_curve(fit["parameters"], np.log10(rows.incorporated_cm3) - 18)[0], [None] * len(rows)
    vals, diagnostics = [], []
    for row in rows.itertuples():
        try:
            v, _, dg, extra = study.point(row, fit["parameters"], fit["method"], meta)
            vals.append(v[0]); diagnostics.append(dict(**dg, **extra))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            vals.append(np.nan); diagnostics.append(dict(error=repr(exc), gradient_valid=False))
    return np.asarray(vals), diagnostics


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); args = ap.parse_args()
    freeze(); choices = original_cv_selections()
    jobs = {}
    for s in choices:
        method, config = study.canonical(s["method"], s["selected"]["config"])
        jobs[method, json.dumps(energy.clean(config), sort_keys=True)] = (method, config)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fitted = list(pool.map(fit_one, jobs.values()))
    fits = {(f["method"], json.dumps(f["config"], sort_keys=True)): f for f in fitted}
    original = study.PRED[study.PRED.kind.eq("RT")]
    weights = 1 / (original.groupby("family").sample_id.transform("count").to_numpy(float) * original.family.nunique())
    growths = {q: weighted_quantile(original.growth_T_C.to_numpy(float), weights, q) for q in (0., .25, .5, .75, 1.)}
    data = read_result_csv(OUT / "paired_samples.csv").sort_values("incorporated_cm3")
    data["H_known"] = False; data["H_cm3"] = 0.; data["carbon_known"] = False; data["carbon_cm3"] = 0.
    data["Tobs_K"] = 300.; data["element"] = "Sn"
    observed = np.log10(data.Hall_n_cm3.to_numpy(float)); outputs = []
    for q, growth in growths.items():
        rows = data.copy(); rows["growth_T_C"] = growth
        for s in choices:
            method, config = study.canonical(s["method"], s["selected"]["config"])
            f = fits[method, json.dumps(energy.clean(config), sort_keys=True)]
            raw, dg = raw_predict(f, rows); budget = s["budget"]; lam = s["selected"]["lam"]
            delta = (float(np.sum(observed[:budget] - raw[:budget]) / (budget + lam))
                     if budget and np.isfinite(lam) else 0.)
            for i, r in enumerate(rows.itertuples()):
                outputs.append(dict(method=s["method"], canonical_method=method, fit_key=f["key"],
                    budget=budget, sample_id=r.sample_id, chemical_dose_cm3=r.incorporated_cm3,
                    role="anchor" if i < budget else "query", fixed_highest_query=i == 2,
                    growth_quantile=q, primary_growth_scenario=q == .5, assumed_growth_T_C=growth,
                    observed=observed[i], raw_predicted=raw[i], offset=delta, predicted=raw[i] + delta,
                    offset_lambda=lam, diagnostics=json.dumps(energy.clean(dg[i]), ensure_ascii=False),
                    label_role="source anchor only" if i < budget else "evaluation only"))
        for method in ("identity_n_equals_C", "identity_plus_unbounded_offset"):
            raw = np.log10(rows.incorporated_cm3.to_numpy(float))
            for budget in (0, 1, 2):
                delta = float(np.mean(observed[:budget] - raw[:budget])) if budget and method.endswith("offset") else 0.
                for i, r in enumerate(rows.itertuples()):
                    outputs.append(dict(method=method, canonical_method=method, fit_key="no_shared_fit",
                        budget=budget, sample_id=r.sample_id, chemical_dose_cm3=r.incorporated_cm3,
                        role="anchor" if i < budget else "query", fixed_highest_query=i == 2,
                        growth_quantile=q, primary_growth_scenario=q == .5, assumed_growth_T_C=growth,
                        observed=observed[i], raw_predicted=raw[i], offset=delta, predicted=raw[i] + delta,
                        offset_lambda=0. if method.endswith("offset") else np.inf,
                        label_role="source anchor only" if i < budget else "evaluation only"))
    frame = pd.DataFrame(outputs); energy.csvwrite(OUT / "predictions.csv", frame.to_dict("records"))
    summaries = []
    for (method, budget, q), g in frame.groupby(["method", "budget", "growth_quantile"]):
        for scope in ("all_query", "fixed_highest_query"):
            query = g[g.role.eq("query")]
            if scope == "fixed_highest_query": query = query[query.fixed_highest_query]
            error = abs(query.predicted - query.observed)
            finite = np.isfinite(error).all()
            summaries.append(dict(method=method, budget=budget, growth_quantile=q, primary_growth_scenario=q == .5,
                scope=scope, queries=len(query), finite_queries=int(np.isfinite(error).sum()),
                MAE=float(error.mean()) if finite else np.inf, max_error=float(error.max()) if finite else np.inf))
    energy.csvwrite(OUT / "summary.csv", summaries)
    energy.write(OUT / "execution.json", dict(created_utc=energy.now(), unique_full_fits=len(fits),
        source_selection_records=len(choices), prediction_rows=len(frame), growth_scenarios=growths,
        no_new_source_in_shared_training=True, new_source_has_mu=False,
        prediction_scope="new source, known SIMS and missing growth/pO2; conditional retrospective transfer; not blind"))
    print("external source completed", len(fits), "fits", len(frame), "rows", flush=True)


if __name__ == "__main__":
    main()
