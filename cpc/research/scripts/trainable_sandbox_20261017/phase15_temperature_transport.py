"""Predict old chemical-paired mu(T) at its own temperatures, without mu fitting."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from csv_io_v2 import read_result_csv as read
from sandbox.coherent_fermi import published_coherent_profile,TABLE
from sandbox.coherent_transport import fermi_transport
import phase15_prediction as base
import phase15_source_adaptation as bench
import phase2_energy as energy

OUT=base.OUT/'temperature_transport'


def main():
    assert json.loads((base.OUT/'research_runtime/verification.json').read_text())['passed']
    input_dir=base.core.prior.prior.TEMPDIR
    obsfile=input_dir/'temperature_observations.csv';specfile=input_dir/'paired_specimens.csv'
    allobs=read(obsfile);obs=allobs[allobs.endpoint.eq('muH_cm2Vs')].copy()
    assert len(obs)==54 and obs.sample_id.nunique()==3 and obs.family.nunique()==1 and obs.observed.gt(0).all()
    held=obs.family.iloc[0];spec=read(specfile).set_index('sample_id');profile=published_coherent_profile()
    runtime=base.OUT/'research_runtime';index=json.loads((runtime/'index.json').read_text())
    rt_tasks={r['task_id']:r for r in json.loads((runtime/'tasks.json').read_text())}
    variants=list(bench.VARIANTS);plans=[]
    for variant,(method,kind) in bench.VARIANTS.items():
        exported=variant if kind!='offset' else method
        for budget in (0,1,2):
            entry=next(r for r in index if r['method']==exported and r['held']==held and r['budget']==budget)
            plans.append((variant,entry,rt_tasks[entry['task_id']]))
    assert len(plans)==54
    sources=(Path(__file__),ROOT/'scripts/trainable_sandbox_20261017/phase15_runtime.py',
        ROOT/'docs/ML_SANDBOX_TEMPERATURE_MOBILITY_20261018.md')
    inputs=(obsfile,specfile,runtime/'verification.json',runtime/'index.json',runtime/'tasks.json',TABLE)
    bundles=[ROOT/p[1]['bundle'] for p in plans]
    protocol=dict(created_utc=energy.now(),code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in sources},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (*inputs,*bundles)},
        observations=54,existing_specimens=3,DOIs=1,source=held,variants=variants,nH_anchor_budgets=[0,1,2],
        primary='mu at itsownT,RTqueryspecimensonly,within_curve_then_sourceequal logMAE;commonbudget2curve',
        baselines='constant205 plus n=NI=C+reportedH withsame3RTAreadouts;no mutargets',
        scope='retrospective;oldsamples/processedgraphs;no sameTmeasuredn orindependentsigma labels',goal_complete=False)
    pp=OUT/'protocol.json'
    if pp.exists():
        old=json.loads(pp.read_text())
        for key in ('code_hashes','input_hashes','variants'):assert old[key]==energy.clean(protocol[key]),key
    else:energy.write(pp,protocol)
    rt=base.core.PRED[base.core.PRED.kind.eq('RT')].set_index('sample_id')
    point_inputs={}
    for r in obs.itertuples():
        source=rt.loc[r.sample_id]
        assert abs(np.log10(source.incorporated_cm3/spec.loc[r.sample_id,'incorporated_cm3']))<1e-10
        point_inputs[r.read_id]=dict(element=source.element,incorporated_cm3=float(source.incorporated_cm3),
            growth_T_C=float(source.growth_T_C),Tobs_K=float(r.T_K),H_known=bool(source.H_known),H_cm3=float(source.H_cm3))
    tasks=[];task_mapping={}
    for variant,entry,old in plans:
        tid=f'{variant}::{held}::{entry["budget"]}';task_mapping[tid]=(variant,entry['budget'])
        tasks.append(dict(task_id=tid,bundle=old['bundle'],anchors=old['anchors'],points=[dict(obs_id=r.read_id,input=point_inputs[r.read_id]) for r in obs.itertuples()]))
    energy.write(OUT/'tasks.json',tasks);output=OUT/'independent_outputs.json'
    subprocess.run([sys.executable,str(ROOT/'scripts/trainable_sandbox_20261017/phase15_runtime.py'),
        '--worker',str(OUT/'tasks.json'),str(output)],cwd=ROOT,check=True)
    result=json.loads(output.read_text());assert not result['blocked_paths'] and result['property_table_open_guard_active']
    source_obs=obs.set_index('read_id');rows=[]
    def row(record,variant,budget,predicted,logn=None):
        roles=base.core.ROLES[base.core.ROLES.family.eq(held)&base.core.ROLES.budget.eq(budget)].set_index('sample_id').role
        return dict(method=variant,budget=budget,read_id=record.name,sample_id=record.sample_id,family=held,T_K=record.T_K,
            role=roles[record.sample_id],observed_log_mu=float(np.log10(record.observed)),predicted_log_mu=predicted,
            predicted_log_nH=logn,raw_instrument=False,new_mu_labels_used=False,
            reading_lower_log=float(np.log10(record.reading_lower)),reading_upper_log=float(np.log10(record.reading_upper)),
            low_temperature_warning=bool(record.low_temperature_warning))
    failures=[]
    for r in result['rows']:
        variant,budget=task_mapping[r['task_id']];o=source_obs.loc[r['obs_id']]
        if r['error'] is None:
            value=float(r['transport_values'][0]);logn=r['values'][0]
            assert abs(r['transport_values'][1]-(np.log10(1.602176634e-19)+logn+value))<1e-10
        else:value=np.nan;logn=np.nan;failures.append(r)
        rows.append(row(o,variant,budget,value,logn))
    for read_id,o in source_obs.iterrows():
        inputs=point_inputs[read_id];dose=inputs['incorporated_cm3']+inputs['H_cm3'];n=np.log10(dose)
        for kind in ('rH1','ionized','ionized_extra'):
            p=fermi_transport(n,n,inputs['Tobs_K'],Nc_ref_cm3=profile.Nc_ref,T_ref_K=profile.T_ref,mass_ratio=profile.electron_mass_ratio,
                extra_mobility_cm2Vs=np.inf if kind=='ionized' else 205.,hall=kind!='rH1')
            for budget in (0,1,2):rows.append(row(o,'unit_dose_'+kind,budget,float(p.values[1]),float(p.values[0])))
        for budget in (0,1,2):rows.append(row(o,'constant205',budget,float(np.log10(205.))))
    d=pd.DataFrame(rows);assert len(d)==3564
    common=set(base.core.ROLES[base.core.ROLES.family.eq(held)&base.core.ROLES.budget.eq(2)&base.core.ROLES.role.eq('query')].sample_id)&set(obs.sample_id)
    assert len(common)==1
    by=[]
    for (method,budget,sample),g in d[d.role.eq('query')].groupby(['method','budget','sample_id']):
        for scope in ('all_query','common_curve'):
            if scope=='common_curve' and sample not in common:continue
            errors=abs(g.predicted_log_mu-g.observed_log_mu)
            by.append(dict(method=method,budget=budget,sample_id=sample,family=held,scope=scope,
                MAE=float(errors.mean()) if errors.notna().all() else np.inf,queries=len(g),finite_queries=int(errors.notna().sum())))
    by=pd.DataFrame(by);summary=by.groupby(['method','budget','scope']).agg(MAE=('MAE','mean'),
        specimens=('sample_id','nunique'),queries=('queries','sum'),finite_queries=('finite_queries','sum')).reset_index()
    d.to_csv(OUT/'predictions.csv',index=False);by.to_csv(OUT/'by_specimen.csv',index=False);summary.to_csv(OUT/'summary.csv',index=False)
    energy.write(OUT/'failures.json',failures)
    energy.write(OUT/'contract_audit.json',dict(passed=True,created_utc=energy.now(),observations=54,existing_specimens=3,DOIs=1,
        new_physical_specimens=0,new_mu_labels_used=False,model_variants=18,baselines=4,independent_model_rows=len(result['rows']),
        total_saved_rows=len(d),new_sigma_labels=0,same_T_measured_n_claimed=False,
        missing_model_outputs=len(failures),missing_primary_query_rows=int(d[d.role.eq('query')].predicted_log_mu.isna().sum()),
        primary0anchor_queries=int(summary[summary.method.eq('constant205')&summary.budget.eq(0)&summary.scope.eq('all_query')].queries.iloc[0]),
        property_table_open_guard_active=True,property_tables_read=False,mu_labels_in_tasks=False,
        protocol_sha256=energy.sha(pp),scope='sameT frozenmodel forward prediction/aggregation;not newindependentstudy or materialscatteringproof',goal_complete=False))
    print('Temperature mu contract passed',len(result['rows']),'model rows, failures',len(failures))
    print(summary[summary.scope.eq('all_query')].pivot(index='method',columns='budget',values='MAE').to_string())


if __name__=='__main__':main()
