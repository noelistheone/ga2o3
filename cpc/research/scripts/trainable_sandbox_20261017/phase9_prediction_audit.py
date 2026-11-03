"""Audit saved phase9 predictions, source exclusion and numerical/physical scope.

Read-only with respect to fitting inputs/results. No model refitting or selection.
"""
from pathlib import Path
import sys, json
from collections import Counter
import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from csv_io_v2 import read_result_csv
import phase2_energy as energy

BASE = energy.BASE
OUT = BASE / "phase9_complete_temperature"


def config_key(config):
    return json.dumps(config, sort_keys=True)


def finite_mae(observed, predicted):
    error = np.abs(np.asarray(predicted, float) - np.asarray(observed, float))
    return float(np.mean(error)) if np.isfinite(error).all() else np.inf


def summary_rows(frame, target, prediction, scopes, temperature=False):
    rows = []
    for (method, budget, family), g in frame[frame.role.eq("query")].groupby(
            ["method", "budget", "family"]):
        for scope, mask in scopes.items():
            h = g[mask(g)]
            if not len(h):
                continue
            error = np.abs(h[prediction].to_numpy(float) - h[target].to_numpy(float))
            # A missing prediction makes its complete-family score infinite.
            # It cannot silently disappear through pandas' skip-NaN averaging.
            if temperature:
                mae = np.mean([finite_mae(s[target], s[prediction])
                               for _, s in h.groupby("sample_id")])
            else:
                mae = finite_mae(h[target], h[prediction])
            rows.append(dict(method=method, budget=int(budget), family=family,
                scope=scope, specimens=h.sample_id.nunique(), reads=len(h),
                numeric=int(np.isfinite(error).sum()), MAE=float(mae)))
    family = pd.DataFrame(rows)
    summary = family.groupby(["method", "budget", "scope"]).agg(
        source_groups=("family", "nunique"), specimens=("specimens", "sum"),
        reads=("reads", "sum"), numeric=("numeric", "sum"),
        MAE=("MAE", "mean"), worst_source_MAE=("MAE", "max")).reset_index()
    return family, summary


def main():
    protocol = json.loads((OUT / "prediction_protocol.json").read_text())
    for group in ("code_hashes", "input_hashes"):
        for file, sha in protocol[group].items():
            assert energy.sha(ROOT / file) == sha, file
    data = read_result_csv(BASE / "dataset_v1.csv")
    roles = read_result_csv(BASE / "episodes_v1.csv")
    obs = read_result_csv(OUT / "observations_and_inputs.csv")
    curves = read_result_csv(BASE / "phase8_process_temperature/extraction_v2/temperature_observations.csv")
    curves = curves[curves.endpoint.eq("log10_nH_cm3")]
    families = set(data.family)
    common18 = set(roles[roles.budget.eq(2) & roles.role.eq("query")].sample_id)
    bound_ids = set(curves[curves.T_upper_K.ge(40) & curves.T_lower_K.le(125)].read_id)
    fits = {p.stem: json.loads(p.read_text()) for p in (OUT / "fits").glob("*.json")}
    lookup = {(f["method"], tuple(f["train_families"]), config_key(f["config"])): f
              for f in fits.values()}
    assert len(lookup) == len(fits)
    assert len(obs) == 171 and len(common18) == 18
    assert obs.obs_id.is_unique and len(curves) == 142

    parameter_rows, optimizer_rows, fit_costs = [], [], []
    for f in fits.values():
        train = obs[obs.family.isin(f["train_families"])]
        train = train[train.kind.eq("RT")] if f["method"].endswith("_RT") else train[
            train.kind.eq("RT") | train.Tobs_K.le(275)]
        assert train.obs_id.tolist() == f["training_obs_ids"]
        assert sorted(train.sample_id.unique()) == f["training_specimens"]
        w = np.asarray(f["weight_per_observation"])
        expected = 1 / (train.groupby("sample_id").obs_id.transform("count").to_numpy(float)
                        * train.groupby("family").sample_id.transform("nunique").to_numpy(float))
        expected /= expected.mean()
        assert np.allclose(w, expected, atol=1e-13, rtol=1e-13)
        check = train[["family", "sample_id"]].assign(weight=w)
        assert check.groupby("family").weight.sum().max() - check.groupby("family").weight.sum().min() < 1e-10
        for _, group in check.groupby("family"):
            sw = group.groupby("sample_id").weight.sum()
            assert sw.max() - sw.min() < 1e-10
        assert {p["obs_id"] for p in f["predictions"]} == set(obs.obs_id)
        best = min((t for t in f["trials"] if t.get("loss") is not None),
                   key=lambda t: t["loss"], default=None)
        for i, trial in enumerate(f["trials"]):
            optimizer_rows.append(dict(fit_key=f["key"], method=f["method"], trial=i,
                training_sources=len(f["train_families"]), success=trial["success"],
                selected=bool(trial is best), loss=trial.get("loss"),
                nfev=trial.get("nfev"), message=trial.get("message", trial.get("error"))))
        if f["parameters"] is not None:
            p = np.asarray(f["parameters"], float)
            assert best is not None and np.allclose(p, best["parameters"], atol=0, rtol=0)
            parameter_rows.append(dict(fit_key=f["key"], method=f["method"],
                training_sources=len(f["train_families"]),
                excluded_sources=";".join(sorted(families - set(f["train_families"]))),
                Tf_K=10**(np.log10(600) + (np.log10(1950) - np.log10(600))*expit(p[0])),
                log10_background=p[1], binding_E0_eV=10**p[2] if len(p)>2 else
                    (.0366 if f["method"] == "fixed_binding_joint" else None),
                concentration_s=p[3] if len(p)>3 else 0,
                selected_start_converged=best["success"],
                active_bounds=";".join(map(str,best["active_mask"])) ,
                background_near_lower_bound=bool(p[1] < 14.001),
                E0_near_upper_bound=bool(len(p)>2 and 10**p[2] > .195)))
        fit_costs.append(dict(method=f["method"], fit_key=f["key"], seconds=f["seconds"],
            optimizer_starts=len(f["trials"]),
            nonconverged=sum(not t["success"] for t in f["trials"]),
            selected_nonconverged=bool(best and not best["success"]),
            state_failures=len(f["state_failures"])))

    def primary_score(fit, held, budget, lam):
        d = data[data.family.eq(held)]
        r = roles[roles.family.eq(held) & roles.budget.eq(budget)].set_index("sample_id").role
        prediction = {p["obs_id"]: p["predicted"] for p in fit["predictions"]}
        x = np.array([prediction["RT::"+s] if prediction["RT::"+s] is not None else np.nan
                      for s in d.sample_id], float)
        a = np.array([r.loc[s] == "anchor" for s in d.sample_id])
        q = np.array([r.loc[s] == "query" for s in d.sample_id])
        delta = 0. if lam is None or not a.any() else np.sum(d.observed_log_nH.to_numpy(float)[a]-x[a])/(a.sum()+lam)
        return finite_mae(d.observed_log_nH.to_numpy(float)[q], (x+delta)[q]), float(delta)

    primary_frames, temperature_frames, design_rows = [], [], []
    nested_contexts = candidate_evaluations = outer_checks = 0
    for method in protocol["models"]:
        folder = OUT / "methods" / method
        p = read_result_csv(folder / "primary_predictions.csv")
        t = read_result_csv(folder / "temperature_predictions.csv")
        assert len(p) == 87 and len(t) == 447
        assert not p.duplicated(["sample_id", "budget"]).any()
        assert not t.duplicated(["obs_id", "budget"]).any()
        for _, r in p.iterrows():
            f = fits[r.fit_key]
            assert r.family not in f["train_families"] and r.sample_id not in f["training_specimens"]
            assert roles[(roles.sample_id.eq(r.sample_id)) & roles.budget.eq(r.budget)].role.iloc[0] == r.role
            outer_checks += 1
        selections = json.loads((folder / "selections.json").read_text())
        for selection in selections:
            held, budget = selection["held"], selection["budget"]
            train_families = families - {held}
            for config in protocol["models"][method]:
                for inner in train_families:
                    f = lookup[(method, tuple(sorted(train_families-{inner})), config_key(config))]
                    assert held not in f["train_families"] and inner not in f["train_families"]
                    nested_contexts += 1
            for c in selection["candidates"]:
                scores = []
                for inner in sorted(train_families):
                    f = lookup[(method, tuple(sorted(train_families-{inner})), config_key(c["config"]))]
                    scores.append(primary_score(f, inner, budget, c["lam"])[0])
                    candidate_evaluations += 1
                assert np.isclose(np.mean(scores), c["score"], atol=3e-12, rtol=0)
            expected_best = min(selection["candidates"], key=lambda c:
                (round(c["score"], 10), -np.inf if c["lam"] is None else -c["lam"], c["index"]))
            assert expected_best == selection["selected"]
            chosen = selection["selected"]
            f = lookup[(method, tuple(sorted(train_families)), config_key(chosen["config"]))]
            score, offset = primary_score(f, held, budget, chosen["lam"])
            assert np.isclose(score, selection["query_MAE"], atol=3e-12, rtol=0)
            rows = p[p.family.eq(held) & p.budget.eq(budget)]
            assert set(rows.fit_key) == {f["key"]}
            assert np.allclose(rows.offset_dex, offset, atol=3e-12, rtol=0)
            for r in rows.itertuples():
                raw = next(x["predicted"] for x in f["predictions"] if x["obs_id"] == "RT::"+r.sample_id)
                assert np.isclose(r.predicted_log_nH, raw+offset, atol=3e-12, rtol=0)
            design_rows.append(dict(method=method, held=held, budget=budget,
                fit_key=f["key"], shrinkage_lambda=chosen["lam"], config=config_key(chosen["config"]),
                inner_MAE=chosen["score"], outer_MAE=score))
        for (held, budget, sample), g in t.groupby(["family", "budget", "sample_id"]):
            assert g.anchor_complete.all()
            assert len(g[g.role.eq("anchor")]) == budget
            s = next(x for x in selections if x["held"] == held and x["budget"] == budget)
            lam = s["selected"]["lam"]
            a = g[g.role.eq("anchor")]
            delta = 0. if not budget or lam is None else np.sum(a.observed-a.raw_prediction)/(budget+lam)
            assert np.allclose(g.offset_dex, delta, atol=3e-12, rtol=0)
            assert np.allclose(g.predicted, g.raw_prediction+delta, atol=3e-12, rtol=0)
            for r in g.itertuples():
                f = fits[r.fit_key]
                assert held not in f["train_families"] and sample not in f["training_specimens"]
                raw = next(x["predicted"] for x in f["predictions"] if x["obs_id"] == r.obs_id)
                assert np.isclose(raw, r.raw_prediction, atol=3e-12, rtol=0)
                outer_checks += 1
        primary_frames.append(p)
        temperature_frames.append(t)
    primary = pd.concat(primary_frames, ignore_index=True)
    temperature = pd.concat(temperature_frames, ignore_index=True)
    allmask = lambda g: pd.Series(True, index=g.index)
    pf, ps = summary_rows(primary, "observed_log_nH", "predicted_log_nH", dict(
        all_query=allmask, common18=lambda g:g.sample_id.isin(common18)))
    tf, ts = summary_rows(temperature, "observed", "predicted", dict(
        all_query=allmask, common40_125=lambda g:g.common40_125,
        reading_bound40_125=lambda g:g.obs_id.isin(bound_ids),
        below100K=lambda g:g.Tobs_K.lt(100), at_least100K=lambda g:g.Tobs_K.ge(100)), temperature=True)
    # Include the already frozen strong two-anchor controls as context, with
    # their original protocol. They receive no new selection or refitting.
    strong = read_result_csv(BASE / "phase8_process_temperature/simple_anchor_predictions.csv")
    strong = strong.rename(columns={"read_id":"obs_id", "T_K":"Tobs_K"})
    sf, ss = summary_rows(strong, "observed", "predicted", dict(
        all_query=allmask, common40_125=lambda g:g.common40_125,
        reading_bound40_125=lambda g:g.obs_id.isin(bound_ids)), temperature=True)
    sf["comparison_contract"] = "anchor_only_stage8_frozen_no_training"
    ss["comparison_contract"] = "anchor_only_stage8_frozen_no_training"
    for frame, name in ((primary,"primary_predictions"), (temperature,"temperature_predictions"),
        (pf,"primary_by_source"), (ps,"primary_summary"), (tf,"temperature_by_source"),
        (ts,"temperature_summary"), (sf,"anchor_controls_by_source"), (ss,"anchor_controls_summary"),
        (pd.DataFrame(parameter_rows),"fitted_parameters"), (pd.DataFrame(optimizer_rows),"optimizer_trials"),
        (pd.DataFrame(design_rows),"selected_designs"), (pd.DataFrame(fit_costs),"fit_costs")):
        frame.to_csv(OUT / (name+".csv"), index=False)

    # Separate numerical validity from physical-model assumptions. Annotate
    # every held-out RT/curve output, retaining all errors and dense inputs.
    input_index = obs.set_index("obs_id")
    annotations = []
    for frame, kind in ((primary,"primary"),(temperature,"temperature")):
        for r in frame.itertuples():
            obs_id = "RT::"+r.sample_id if kind == "primary" else r.obs_id
            source = input_index.loc[obs_id]
            f = fits[r.fit_key]
            a = next(x for x in f["predictions"] if x["obs_id"] == obs_id)
            dg = a.get("diagnostics", {})
            annotations.append(dict(task=kind, method=r.method, budget=int(r.budget),
                sample_id=r.sample_id, obs_id=obs_id, family=r.family, role=r.role,
                numeric=a["numeric"], raw_physical_valid=a.get("physical_valid"),
                raw_flag_meaning="root_gradient_and_electron_Hall_only_not_catalogue_completeness" if dg else "not_applicable_empirical",
                above_reference_floor=dg.get("above_carrier_detection_floor"),
                actual_dose_cm3=source.incorporated_cm3,
                above_Si_Mott_reference=bool(source.incorporated_cm3 > 2.4e18),
                Mott_reference_scope="Si_estimate_from_McCandless_not_validated_Sn_threshold",
                local_donor_extension=dg.get("donor_extension",False),
                isolated_donor_domain_validated=False if dg.get("donor_extension") else None,
                H_status="reported_detected_and_supplied" if source.H_known else "unknown_conditional_zero",
                carbon_detected_unmodeled=bool(source.carbon_known),
                numerical_root_log_residual=dg.get("root_residual_log"),
                jacobian_condition=dg.get("jacobian_condition")))
    ann = pd.DataFrame(annotations)
    ann.to_csv(OUT/"canonical_prediction_annotations.csv", index=False)

    # Matched source comparison; counts are descriptive, not significance tests
    # with five (or two) independent source groups.
    pairs = []
    for task, frame in (("primary",pf),("temperature",tf)):
        scope = "all_query" if task == "primary" else "common40_125"
        z = frame[frame.scope.eq(scope)]
        for baseline in ("fixed_binding_joint","no_extension_joint","no_extension_RT","direct_ridge_joint"):
            for learner in ("learned_binding_joint","learned_screen_joint"):
                for budget in (0,1,2):
                    a=z[z.method.eq(learner)&z.budget.eq(budget)].set_index("family").MAE
                    b=z[z.method.eq(baseline)&z.budget.eq(budget)].set_index("family").MAE
                    diff=a-b
                    pairs.append(dict(task=task, scope=scope, method=learner, baseline=baseline,
                        budget=budget, sources=len(diff), better=int((diff < -1e-10).sum()),
                        tied=int((abs(diff)<=1e-10).sum()), worse=int((diff>1e-10).sum()),
                        source_mean_MAE_difference=float(diff.mean())))
    pd.DataFrame(pairs).to_csv(OUT/"paired_source_comparisons.csv", index=False)
    protected=[]
    for file in ("protected_inputs_before.json","artifact_manifest_phase1.json","artifact_manifest_phase2.json",
                 "artifact_manifest_phase34.json","artifact_manifest_phase56.json","artifact_manifest_phase7.json",
                 "artifact_manifest_phase8.json"):
        x=json.loads((BASE/file).read_text());entries=x["files"] if isinstance(x,dict) else x
        changed=[v["path"] for v in entries if not (ROOT/v["path"]).is_file() or energy.sha(ROOT/v["path"])!=v["sha256"]]
        assert not changed, changed
        protected.append(dict(manifest=file,checked=len(entries),changed=changed))
    costs=pd.DataFrame(fit_costs)
    qann=ann[ann.role.eq("query") & ann.method.isin([m for m in protocol["models"] if not m.startswith(("direct","dose"))])]
    audit=dict(passed=True,created_utc=energy.now(),goal_status="active",goal_complete=False,
        direction="train internal shared relationship through full defect Sandbox for actual-dose-to-doped-property prediction",
        specimens=29,source_groups=5,temperature_specimens=7,temperature_source_groups=2,
        new_physical_specimens=0,temperature_carrier_reads=142,
        methods=len(protocol["models"]),fits=len(fits),optimizer_starts=int(costs.optimizer_starts.sum()),
        nonconverged_starts=int(costs.nonconverged.sum()),
        selected_nonconverged_fit_records=int(costs.selected_nonconverged.sum()),
        state_failures=int(costs.state_failures.sum()),summed_fit_seconds=float(costs.seconds.sum()),
        cost_scope="sum of recorded fit intervals,not wallclock; empirical/physical seconds differ; candidate cap not identical runtime",
        nested_training_contexts=nested_contexts,nested_candidate_evaluations=candidate_evaluations,
        saved_predictions_source_exclusion_checks=outer_checks,
        selection_scores_and_offsets_independently_reconstructed=True,
        observation_source_and_sample_balancing_verified=True,
        RT_query_counts=primary[primary.method.eq("no_extension_RT") & primary.role.eq("query")].groupby("budget").size().to_dict(),
        temperature_query_counts=temperature[temperature.method.eq("no_extension_RT") & temperature.role.eq("query")].groupby("budget").size().to_dict(),
        raw_numerical_or_electronHall_false_queries=int(qann.raw_physical_valid.eq(False).sum()),
        below_reference_floor_query_predictions=int(qann.above_reference_floor.eq(False).sum()),
        validation="retrospective source-held-out,all sources previously exposed;not blind",
        donor_relation="low-dimensional effective E0,s trained on real labels via IFT,not a new ML algorithm or HSE charge-state label",
        physical_scope="isolated donor two-state statistics not a Mott/impurity-band model;no point removals by regime flags",
        temperature_fit_scope="two Si curve sources only; no independent Sn temperature curve validation",
        property_chain="nH and charge budgets; no validated variable-temperature mobility; v4 unchanged",
        main_prediction_increment_established=False,
        temperature_conditional_signal="shared E(C) beats fixed E,other tested full models and current directRidge on common/all query,not strong two-anchor line at budget2",
        previous_protected=protected,
        next="source/sample-specific freezeout and plateau residuals; formulate physically consistent localized/delocalized state+Hall alternative before training; retain original29/5 endpoint",
        goal_completion_sufficient=False)
    energy.write(OUT/"contract_audit.json",audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ("previous_protected",)},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
