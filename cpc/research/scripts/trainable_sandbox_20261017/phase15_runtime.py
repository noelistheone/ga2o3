"""Independent coherent-state prediction and full chemical property audit."""
from pathlib import Path
import sys,json,argparse,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(tasks_path,output_path):
    forbidden=('dataset_v1.csv','episodes_v1.csv','paired_specimens.csv','temperature_observations.csv',
        '/fits/','/methods/','/validation/','/episodes/','/variants/','head_parameters.json')
    blocked=[];opened=[]
    def guard(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        path=str(args[0]);opened.append(path)
        if any(s in path for s in forbidden):blocked.append(path);raise RuntimeError('Property/fit/selection table access forbidden')
    sys.addaudithook(guard);sys.path.insert(0,str(ROOT/'src'))
    from types import SimpleNamespace
    import numpy as np
    from sandbox.coherent_hall_model import CoherentHallModel
    from sandbox.coherent_fermi import CoherentMaterialProfile,TABLE
    from sandbox.scalar_profile_adaptation import ScalarProfile
    from sandbox.kroger_db import NPZ
    tasks=json.loads(Path(tasks_path).read_text());rows=[];adaptations=[]
    for task in tasks:
        b=json.loads(Path(task['bundle']).read_text());assert sha(NPZ)==b['energetics_sha256'] and sha(TABLE)==b['fd_reference_sha256']
        model=CoherentHallModel(b['definition'],profile=CoherentMaterialProfile.from_dict(b['material_profile']),pO2_atm=b['pO2_atm'])
        w=b['parameters'];lam=np.inf if b['lambda'] is None else b['lambda'];z=0.;kind=b['source_kind']
        if task['anchors'] and np.isfinite(lam):
            if kind=='offset':
                residual=[model.point(SimpleNamespace(**a['input']),w).values[0]-a['observed_log_nH'] for a in task['anchors']]
                z=float(sum(residual)/(len(residual)+lam))
            else:
                aa=[SimpleNamespace(**a['input']) for a in task['anchors']]
                def forward(z):
                    points=[model.point(a,w,source_kind=kind,source_shift=z) for a in aa]
                    return np.array([p.values[0] for p in points]),np.array([p.source_jacobian[0] for p in points])
                fit=ScalarProfile(forward).fit(np.array([a['observed_log_nH'] for a in task['anchors']]),lam)
                if fit.status=='no_feasible_anchor_state':raise ValueError('No feasible legal anchor state')
                z=fit.shift
        adaptations.append(dict(task_id=task['task_id'],shift=z))
        for point in task['points']:
            try:
                r=model.point(SimpleNamespace(**point['input']),w,source_kind=kind,source_shift=z)
                row=dict(task_id=task['task_id'],obs_id=point['obs_id'],values=r.values.tolist(),transport_values=r.transport_values.tolist(),
                    physical_values=r.physical_values.tolist(),physical_transport_values=r.diagnostics['physical_transport_values'],
                    same_physical_state_for_output=r.diagnostics['same_physical_state_for_output'],hall_factor=r.diagnostics['hall_factor'],error=None)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                row=dict(task_id=task['task_id'],obs_id=point['obs_id'],values=None,transport_values=None,error=repr(exc))
            rows.append(row)
    Path(output_path).write_text(json.dumps(dict(rows=rows,adaptations=adaptations,property_tables_read=False,
        property_table_open_guard_active=True,blocked_paths=blocked,opened_path_count=len(opened)),ensure_ascii=False,indent=2)+'\n')


def parent(refined):
    sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
    import numpy as np
    import pandas as pd
    from csv_io_v2 import read_result_csv as read
    from sandbox.coherent_fermi import published_coherent_profile,TABLE
    from sandbox.kroger_db import NPZ
    from sandbox import transport
    import phase15_prediction as base
    import phase15_refinement as ref
    import phase15_source_adaptation as bench
    import phase2_energy as energy
    final=ref.OUT if refined else base.OUT;v=final/'validation';out=base.OUT/'research_runtime'
    assert json.loads((v/'contract_audit.json').read_text())['passed']
    assert json.loads((bench.OUT/'validation/contract_audit.json').read_text())['passed']
    bank=ref.load_composite() if refined else base.composite();pred=read(v/'primary_predictions.csv')
    extras={variant:pair for variant,pair in bench.VARIANTS.items() if pair[1]!='offset'}
    epred=read(bench.OUT/'validation/predictions.csv');epred=epred[epred.variant.isin(extras)].rename(columns={'variant':'method'})
    allpred=pd.concat([pred,epred],ignore_index=True)
    protocol=dict(created_utc=energy.now(),code_sha256=sha(Path(__file__)),refined=refined,
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in (v/'contract_audit.json',v/'primary_predictions.csv',TABLE,
            bench.OUT/'validation/contract_audit.json',bench.OUT/'validation/predictions.csv')},
        code_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/sandbox/coherent_hall_model.py',ROOT/'src/sandbox/coherent_fermi.py',ROOT/'src/sandbox/coherent_transport.py',ROOT/'src/sandbox/scalar_profile_adaptation.py')},
        scope='6newmodels+12internalsourcevariants/270bundles;rawphysmu,nHanchorsonly;no newmutargets duringfit',
        fixed_prior_scope='205/CT constants contain historicalempiricalmobility information;not purefirstprinciples nor provenindependenthistoricaldata',goal_complete=False)
    pp=out/'protocol.json'
    if pp.exists():
        old=json.loads(pp.read_text())
        for key in ('code_sha256','input_hashes','code_hashes','refined'):assert old[key]==protocol[key],key
    else:energy.write(pp,protocol)
    def inputs(r):
        return dict(element=r.element,incorporated_cm3=float(r.incorporated_cm3),growth_T_C=float(r.growth_T_C),
            Tobs_K=float(r.Tobs_K),H_known=bool(r.H_known),H_cm3=float(r.H_cm3))
    tasks=[];expected={};index=[];expected_shifts={}
    plans=[(m,m,'offset',final/'methods'/m/'selections.json') for m in base.NEW_METHODS]
    plans += [(variant,m,k,bench.OUT/'variants'/variant/'selections.json') for variant,(m,k) in extras.items()]
    for variant,method,kind,path in plans:
        choices=json.loads(path.read_text())
        for c in choices:
            held,budget=c['held'],c['budget'];f=bank[c['fit_key']];assert held not in f['train_families']
            tid=f'{variant}::{held}::{budget}';bp=out/'bundles'/(base.core.prior.old.digest(tid)+'.json')
            energy.write(bp,dict(method=variant,base_method=method,source_kind=kind,held=held,budget=budget,parameters=f['parameters'],definition=f['definition'],
                lambda_selected=c['selected']['lam'],**{'lambda':c['selected']['lam']},material_profile=published_coherent_profile().to_dict(),
                pO2_atm=1e-5,fit_key=f['key'],fit_path=f['composite_origin']['path'],fit_sha256=f['composite_origin']['sha256'],
                energetics_sha256=sha(NPZ),fd_reference_sha256=sha(TABLE),new_mu_labels_used=False,
                fixed_empirical_mobility_scale=f['definition']['readout']!='ionized'))
            d=base.core.PRED[base.core.PRED.family.eq(held)&base.core.PRED.kind.eq('RT')]
            role=base.core.ROLES[base.core.ROLES.family.eq(held)&base.core.ROLES.budget.eq(budget)].set_index('sample_id').role
            anchors=[dict(input=inputs(r),observed_log_nH=float(r.target)) for r in d.itertuples() if role[r.sample_id]=='anchor']
            assert len(anchors)==budget
            tasks.append(dict(task_id=tid,bundle=str(bp),anchors=anchors,points=[dict(obs_id=r.obs_id,input=inputs(r)) for r in d.itertuples()]))
            index.append(dict(task_id=tid,method=variant,base_method=method,source_kind=kind,held=held,budget=budget,bundle=str(bp.relative_to(ROOT))))
            selected=allpred[allpred.method.eq(variant)&allpred.family.eq(held)&allpred.budget.eq(budget)]
            expected_shifts[tid]=float(c['shift']) if kind!='offset' else -float(selected.offset_dex.iloc[0])
            lookup={r['obs_id']:r for r in f['predictions']}
            for r in selected.itertuples():
                saved=lookup['RT::'+r.sample_id]
                if kind=='offset':
                    mu,sigma=saved['transport_values'];charges=saved['charge_outputs'];sigma+=r.offset_dex
                else:
                    mu,sigma=r.predicted_log_mu_H,r.predicted_log_sigma;charges=[r.logNabs,r.logNq2]
                expected[tid,'RT::'+r.sample_id]=dict(n=r.predicted,charges=charges,transport=[mu,sigma])
    energy.write(out/'tasks.json',tasks);energy.write(out/'index.json',index)
    output=out/'independent_outputs.json'
    subprocess.run([sys.executable,str(Path(__file__)),'--worker',str(out/'tasks.json'),str(output)],cwd=ROOT,check=True)
    replay=json.loads(output.read_text());actual={(r['task_id'],r['obs_id']):r for r in replay['rows']};errors=[];charge_errors=[];transport_errors=[]
    for key,wanted in expected.items():
        r=actual[key]
        if np.isnan(wanted['n']):assert r['values'] is None;continue
        assert r['error'] is None
        errors.append(abs(r['values'][0]-wanted['n']));charge_errors.extend(abs(np.array(r['values'][1:])-wanted['charges']))
        transport_errors.extend(abs(np.array(r['transport_values'])-wanted['transport']))
        assert abs(r['transport_values'][1]-(np.log10(transport.E)+r['values'][0]+r['transport_values'][0]))<1e-11
    shift_errors=[abs(r['shift']-expected_shifts[r['task_id']]) for r in replay['adaptations']]
    assert len(replay['rows'])==1566 and not replay['blocked_paths'] and max(errors)<1e-6 and max(charge_errors)<1e-6 and max(transport_errors)<1e-6 and max(shift_errors)<1e-5
    data=base.core.DATA.set_index('sample_id');eligible=set(data[data.mobility_eligible.fillna(False)&data.mu_cm2Vs.gt(0)].index)
    chain=[];pair_replays=[]
    for r in allpred[allpred.role.eq('query')&allpred.sample_id.isin(eligible)].itertuples():
        state_mu=None;raw_sigma=None;same=False
        if r.method in extras:
            model_mu=r.predicted_log_mu_H;state_mu=model_mu;raw_sigma=r.predicted_log_sigma;mechanism='coherent_internal_source_state'
        elif r.method in base.NEW_METHODS:
            f=bank[r.fit_key];saved=next(a for a in f['predictions'] if a['obs_id']=='RT::'+r.sample_id)
            state_mu=saved['transport_values'][0];raw_sigma=saved['transport_values'][1];model_mu=state_mu
            mechanism='coherent_fromNq2_andfreecompressibility'
        else:model_mu=float(np.log10(transport.mu_caughey_thomas(10**r.predicted,300)));mechanism='rawfixed_CT_nHargument'
        ls=float(np.log10(transport.E)+r.predicted+model_mu);obs_mu=float(np.log10(data.loc[r.sample_id,'mu_cm2Vs']))
        chain.append(dict(method=r.method,family=r.family,budget=r.budget,sample_id=r.sample_id,
            predicted_log_nH=r.predicted,observed_log_nH=r.observed,predicted_log_mu=model_mu,observed_log_mu=obs_mu,
            predicted_log_sigma=ls,observed_log_sigma=np.log10(transport.E)+r.observed+obs_mu,
            mu_head=mechanism,new_mu_labels_used=False,conductivity_is_derived=True,raw_physical_log_sigma=raw_sigma,
            fixed_empirical_scale=not((r.method in base.NEW_METHODS and base.NEW_METHODS[r.method]['readout']=='ionized') or
                (r.method in extras and base.NEW_METHODS[extras[r.method][0]]['readout']=='ionized')),
            same_physical_state_for_output=bool(r.same_physical_state_for_output)))
        if state_mu is not None:
            a=actual[f'{r.method}::{r.family}::{int(r.budget)}','RT::'+r.sample_id]
            pair_replays.extend(abs(np.array(a['transport_values'])-[model_mu,ls]))
    df=pd.DataFrame(chain);assert len(df)==418 and len(pair_replays)==396
    df.to_csv(out/'full_chain_predictions.csv',index=False);by=[]
    for (m,b,f),g in df.groupby(['method','budget','family']):
        row=dict(method=m,budget=b,family=f,queries=len(g))
        for prop in ('nH','mu','sigma'):row[prop+'_MAE']=float(abs(g['predicted_log_'+prop]-g['observed_log_'+prop]).mean())
        by.append(row)
    by=pd.DataFrame(by);by.to_csv(out/'full_chain_by_source.csv',index=False)
    by.groupby(['method','budget'])[['nH_MAE','mu_MAE','sigma_MAE']].mean().reset_index().to_csv(out/'full_chain_summary.csv',index=False)
    energy.write(out/'verification.json',dict(passed=True,created_utc=energy.now(),saved_bundles=270,carrier_rows=1566,
        maximum_carrier_error=max(errors),maximum_charge_error=max(charge_errors),maximum_transport_error=max(transport_errors),
        maximum_source_coordinate_error=max(shift_errors),
        chemical_paired_transport_replays=len(pair_replays)//2,maximum_paired_transport_error=max(pair_replays),
        full_chain_rows=len(df),chemical_paired_mu_specimens=5,chemical_paired_mu_DOIs=2,
        property_table_open_guard_active=True,property_tables_read=False,query_labels_in_tasks=False,new_mu_labels_used=False,
        fixed_empirical_205_and_CT_prior_disclosed=True,
        independent_sigma_labels=0,goal_complete=False))
    print('Coherent runtime passed',len(tasks),'bundles',len(replay['rows']),'carrier rows',max(errors))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',nargs=2);ap.add_argument('--refined',action='store_true');a=ap.parse_args()
    if a.worker:worker(*a.worker)
    else:parent(a.refined)


if __name__=='__main__':main()
