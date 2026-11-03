"""Audit temperature inputs and baseline prediction; no favorable refitting."""
from pathlib import Path
import os,sys,json,time
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import expit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sandbox.implicit_defects import _fd_log
from sandbox.donor_ionization import solve_ionization
from csv_io_v2 import read_result_csv
import phase2_energy as energy
import phase8_temperature_baselines as study

BASE=energy.BASE;OUT=BASE/"phase8_process_temperature"

def main():
    study.freeze()
    # This supplement was specified after seeing the first five-control study.
    # No baseline is chosen by its held-out result; all prescribed lines retained.
    extra=OUT/"simple_anchor_protocol.json"
    protocol=dict(created_utc=energy.now(),code_sha256=energy.sha(Path(__file__)),
        parent_protocol=energy.sha(study.OUT/"protocol.json"),
        timing="supplementary control design after first five-model results; exploratory, not blind",
        controls=dict(RT_constant="publishedRT fixed acrossT, ignores second allowed anchor",
            last_temperature_anchor="last allowed anchor held constant",
            two_anchor_lines="log(n/T^p)=a+b/T, fixedp0/.75/1.5, all reported; only budget2"),
        anchor_rule="same frozenRT and nearest150T points; exact same query/guard roles",
        sensitivity="40K bound using reading interval intersects40..125, no refit or reselection",
        FD_check="quadrature reference for approximated FD at21quantiles of actual fitted eta; no refit")
    if extra.exists():
        old=json.loads(extra.read_text());assert old["code_sha256"]==protocol["code_sha256"]
    else:energy.write(extra,protocol)
    d=read_result_csv(study.OUT/"predictions.csv")
    data=read_result_csv(study.DATA_DIR/"temperature_observations.csv")
    data=data[data.endpoint.eq("log10_nH_cm3")]
    specs=read_result_csv(study.DATA_DIR/"paired_specimens.csv").set_index("sample_id")
    reference=d[d.method.eq("chemical_constant")];rows=[]
    for (sample,budget),g in reference.groupby(["sample_id","budget"]):
        if budget==0:continue
        a=g[g.role.eq("anchor")].sort_values("T_K",ascending=False)
        assert len(a)==budget
        rt=a.iloc[0];last=a.iloc[-1]
        for r in g.itertuples():
            for method,pred in (("RT_constant",rt.observed),("last_temperature_anchor",last.observed)):
                row=r._asdict();row.update(method=method,predicted=float(pred),fit_key="anchor_only_no_shared_fit");rows.append(row)
            if budget==2:
                for power in (0.,.75,1.5):
                    y0=rt.observed-power*np.log10(rt.T_K/300)
                    y1=last.observed-power*np.log10(last.T_K/300)
                    slope=(y1-y0)/(1/last.T_K-1/rt.T_K)
                    pred=y0+slope*(1/r.T_K-1/rt.T_K)+power*np.log10(r.T_K/300)
                    row=r._asdict();row.update(method="two_anchor_line_p"+str(power),predicted=float(pred),
                                               fit_key="two_anchor_no_shared_fit");rows.append(row)
    add=pd.DataFrame(rows);add.to_csv(OUT/"simple_anchor_predictions.csv",index=False)
    both=pd.concat([d,add],ignore_index=True)
    by=[]
    for (m,b,f,s),g in both[both.role.eq("query")].groupby(["method","budget","family","sample_id"]):
        for scope in ("all_query","common40_125","warm100plus","coldBelow100","common_reading_bound"):
            if scope=="all_query":h=g
            elif scope=="common40_125":h=g[g.common40_125]
            elif scope=="warm100plus":h=g[g.T_K.ge(100)]
            elif scope=="coldBelow100":h=g[g.T_K.lt(100)]
            else:
                ids=set(data[(data.T_upper_K.ge(40))&data.T_lower_K.le(125)].read_id)
                h=g[g.read_id.isin(ids)]
            if not len(h):continue
            e=abs(h.predicted-h.observed)
            by.append(dict(method=m,budget=int(b),family=f,sample_id=s,scope=scope,n=len(h),
                          numeric=int(e.notna().sum()),MAE=float(e.mean()) if e.notna().all() else None))
    by=pd.DataFrame(by);by.to_csv(OUT/"combined_by_specimen.csv",index=False)
    family=by.groupby(["method","budget","family","scope"]).agg(
        specimens=("sample_id","nunique"),reads=("n","sum"),MAE=("MAE","mean")).reset_index()
    family.to_csv(OUT/"combined_by_family.csv",index=False)
    summary=family.groupby(["method","budget","scope"]).agg(
        source_groups=("family","nunique"),specimens=("specimens","sum"),reads=("reads","sum"),
        MAE=("MAE","mean"),worst=("MAE","max")).reset_index()
    summary.to_csv(OUT/"combined_summary.csv",index=False)
    fits={p.stem:json.loads(p.read_text()) for p in (study.OUT/"fits").glob("*.json")}
    scopes=0;candidate_evaluations=0
    lookup={(f["method"],tuple(f["training_specimens"]),json.dumps(f["config"],sort_keys=True)):f for f in fits.values()}
    choices=json.loads((study.OUT/"selections.json").read_text())
    for v in choices:
        held=v["held"];expected=sorted(study.SPECS[study.SPECS.family.ne(held)].sample_id)
        assert expected==v["training_specimens"]
        for inner in expected:
            tr=tuple(x for x in expected if x!=inner)
            for config in study.GRIDS[v["method"]]:
                f=lookup[(v["method"],tr,json.dumps(config,sort_keys=True))]
                assert inner not in f["training_specimens"] and held not in f["training_families"]
                scopes+=1
        candidate_evaluations+=len(v["candidates"])*len(expected)
    for row in d.itertuples():
        f=fits[row.fit_key]
        assert row.family not in f["training_families"] and row.sample_id not in f["training_specimens"]
    protected=[]
    for file in ("protected_inputs_before.json","artifact_manifest_phase1.json","artifact_manifest_phase2.json",
                 "artifact_manifest_phase34.json","artifact_manifest_phase56.json","artifact_manifest_phase7.json"):
        p=BASE/file;a=json.loads(p.read_text());entries=a["files"] if isinstance(a,dict) else a
        changed=[v["path"] for v in entries if not (ROOT/v["path"]).exists() or energy.sha(ROOT/v["path"])!=v["sha256"]]
        protected.append(dict(manifest=file,checked=len(entries),changed=changed))
    assert all(not v["changed"] for v in protected)
    # Compare direct vector read near room temperature to exact TableI; do not
    # force marker positions/ordinates to match the table.
    checks=[]
    for sample,g in data.groupby("sample_id"):
        near=g.iloc[(g.T_K-300).abs().argsort().iloc[0]]
        ref=np.log10(float(specs.loc[sample,"reported_RT_nH_cm3"]))
        checks.append(dict(sample_id=sample,nearest_plot_T_K=float(near.T_K),
            plot_read=float(near.observed),published_RT_log_nH=ref,
            difference=float(near.observed-ref),
            warning="nearby temperatures are not identical; audit only, not new labels"))
    energy.write(OUT/"room_temperature_read_audit.json",checks)
    # Quantify approximation accuracy where the fitted reduced roots lie.
    eta=[]
    for f in fits.values():
        if f["method"].startswith("learned") and len(f["training_families"])==1:
            # Include all saved states, even inner fits; no new fitting.
            w=np.asarray(f["parameters"]);c=study.PRED.incorporated_cm3.to_numpy(float)
            T=study.PRED.T_K.to_numpy(float);E=10**w[2]*np.exp(-(w[3] if len(w)>3 else 0)*(c/1e18)**(1/3))
            q=solve_ionization(c,T,background=10**w[0],compensation=float(expit(w[1])),energies_eV=E)
            eta.extend(q["eta"].tolist())
    fd=[]
    for v in np.quantile(eta,np.linspace(0,1,21)):
        exact=2/np.sqrt(np.pi)*quad(lambda e:np.sqrt(e)*expit(v-e),0,np.inf,epsabs=1e-13,epsrel=1e-10)[0]
        approx=np.exp(_fd_log(np.array(v))[0])
        fd.append(dict(eta=float(v),quadrature=exact,approximation=float(approx),
                       difference_log10=float(np.log10(approx/exact))))
    energy.write(OUT/"FD_quadrature_diagnostic.json",dict(values=fd,
        max_abs_log10_difference=max(abs(v["difference_log10"]) for v in fd),
        scope="sampled21etaquantiles; FD value error, not end-to-end prediction error bound"))
    old=read_result_csv(BASE/"dataset_v1.csv").set_index("sample_id")
    assert set(specs.index)<=set(old.index)
    assert np.allclose(specs.incorporated_cm3,[old.loc[s,"incorporated_cm3"] for s in specs.index])
    source_audit=dict(
        mauze=dict(doi="10.1063/5.0027870",primary_existing_markers=11,
            actual_process="all11Fig4b MOCATAXY,indium-catalyzed MBE",
            growth_T_C_interval=[700,900],per_marker_temperature_mapping="not found in main paper",
            common_process=dict(plasma_W=200,oxygen_foreline_Torr=60,Ga_BEP_Torr=2.5e-7,
                In_BEP_Torr=4e-7,typical_growth_nm_min=5,UID_buffer_nm=200),
            Hall_film_thickness_nm=[200,300],contact_anneal="500C,N2,1min",
            Figure5_temperature_specimen="different thick film;body~1000nm,caption900nm",
            fitted_donor_ND_8p2e17_is_chemical_input=False,
            same_specimen_SIMs_Fig5="not established",
            action="retain existing points; future per-markerT remains interval, no invented assignment",
            previous_label_annotation="PAMBE wording in phase7 figure/report was too broad; source-specific process is MOCATAXY; frozen numeric results unchanged"),
        xiang=dict(doi="10.1007/s40843-022-2167-x",paired_specimens=4,
            measured_H_detected_in=["C","D"],carbon_surface_accumulation="D",
            missing_C_H_are_zero=False,reported_SIMS_absolute_error_cm3=1e16,
            fitted_donor_level_tables_are_independent_labels=False,
            lowT_C_departure="retained; author suggests impurity band, not uniquely established here"),
        mccandless=dict(doi="10.1063/5.0101132",paired_specimens=3,
            paired_carrier_mobility_temperature=True,reported_Mott_reference_cm3=2.4e18,
            source1and2_similar_Si_different_curves="growthT and flux also differ; not a dose-only universal binding relation"),
        primary_prediction_data="7old Si specimens/2existingDOI,142new curve reads;54mu reads/3same specimens",
        no_new_physical_samples=True)
    energy.write(OUT/"source_process_audit.json",source_audit)
    audit=dict(passed=True,goal_status="active",actual_specimens=7,DOI_groups=2,new_specimens=0,
        carrier_curve_reads=142,mobility_curve_reads=54,outer_source_exclusion_checked_rows=len(d),
        nested_specimen_fit_contexts=scopes,nested_candidate_evaluations=candidate_evaluations,
        cached_fit_records=len(fits),physical_fits=sum(f["method"].startswith("learned") for f in fits.values()),
        optimizer_starts=sum(len(f["trials"]) for f in fits.values()),
        nonconverged_starts=sum(not t["success"] for f in fits.values() for t in f["trials"]),
        root_failures=sum(len(f["state_failures"]) for f in fits.values()),
        summed_fit_seconds=sum(f["seconds"] for f in fits.values()),
        cost_interpretation="sum measured fitting intervals, excludes process startup, extraction,rendering",
        original_energetic_benchmark_unchanged=True,current_predictor="v4 unchanged",
        reduced_baselines_not_integrated_KROGER=True,
        new_scope="same-dose variable-temperature prediction pilot, supplements original29/5task",
        common_primary_reads=80,
        uncertainty_endpoint_annotation="four nominal40K raster points have centralTjust<40; reading-bound sensitivity retained without model reselection",
        previous_protected=protected,
        next="full energetic frozen-stock variable-temperature observation/interface, then same-budget learned-relation tests",
        sufficient_to_complete_goal=False)
    energy.write(OUT/"contract_audit.json",audit)
    figures(summary,d,data)
    print(summary[summary.scope.eq("common40_125")].to_string(index=False))
    print(json.dumps(audit,ensure_ascii=False,indent=2))

def figures(summary,pred,data):
    folder=OUT/"figures";folder.mkdir(exist_ok=True)
    names=["chemical_constant","hydrogenic_FD","learned_binding","learned_screened_binding","direct_ridge"]
    fig,ax=plt.subplots(figsize=(8.2,4))
    for m in names:
        d=summary[summary.method.eq(m)&summary.scope.eq("common40_125")].sort_values("budget")
        ax.plot(d.budget,d.MAE,"o-",label=m)
    ax.set(xlabel="Temperature anchors PER SPECIMEN",ylabel="Specimen/source-equal Hall MAE (dex)",xticks=[0,1,2])
    ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(folder/"temperature_comparison.png",dpi=170);fig.savefig(folder/"temperature_comparison.pdf");plt.close(fig)
    ids=list(study.SPECS.sample_id);fig,axes=plt.subplots(2,4,figsize=(12,6.2))
    for ax,sample in zip(axes.flat,ids):
        g=data[data.sample_id.eq(sample)].sort_values("T_K")
        ax.plot(g.T_K,g.observed,"k.",label="Published measured trace")
        for method in ("hydrogenic_FD","learned_screened_binding","direct_ridge"):
            p=pred[pred.sample_id.eq(sample)&pred.method.eq(method)&pred.budget.eq(0)&
                   ~pred.read_id.str.startswith("tableRT::")].sort_values("T_K")
            ax.plot(p.T_K,p.predicted,label=method)
        ax.set(title=sample.split("::")[-1]+(" MOCVD" if "2167" in sample else " PA-MBE"),
               xlabel="T(K)",ylabel="log10 apparent Hall density");ax.grid(alpha=.2)
    axes.flat[-1].axis("off");axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc="center",fontsize=8)
    fig.tight_layout();fig.savefig(folder/"held_source_curves.png",dpi=170);fig.savefig(folder/"held_source_curves.pdf");plt.close(fig)

if __name__=="__main__":main()
