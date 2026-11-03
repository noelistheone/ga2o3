"""Refine every failed-start new fit uniformly; never select by test score."""
from pathlib import Path
import sys,json,time,copy,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from scipy.optimize import least_squares
import phase15_prediction as base
import phase2_energy as energy
study=base.core;ORIGINAL=base.OUT;OUT=ORIGINAL/'solver_budget_v2'


def load_composite():
    rows=base.composite()
    for path in (OUT/'fits').glob('*.json'):
        f=json.loads(path.read_text());f['composite_origin']=dict(path=str(path.relative_to(ROOT)),sha256=energy.sha(path));rows[f['key']]=f
    return rows


def refine():
    base.freeze()
    affected=[p for p in sorted((ORIGINAL/'fits').glob('*.json')) if any(not t['success'] for t in json.loads(p.read_text())['trials'])]
    if not affected:print('No failed-start fits: no refinement needed');return False
    protocol=dict(created_utc=energy.now(),code_hashes={str(Path(__file__).relative_to(ROOT)):energy.sha(Path(__file__))},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in [ORIGINAL/'prediction_protocol.json',*affected]},
        eligibility='ALLnewrecords with any failedstart,notouterqueryscore',
        starts='bothoriginalstarts+savedbest iffinite;max_nfev360;unchangedobjective/bounds/tolerances/IFT',
        goal_complete=False)
    path=OUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        for key in ('code_hashes','input_hashes'):assert old[key]==protocol[key],key
    else:energy.write(path,protocol)
    gates=[];executions=[]
    for path in affected:
        old=json.loads(path.read_text());dest=OUT/'fits'/path.name
        if dest.exists():
            saved=json.loads(dest.read_text());gates.append(saved['refinement']['gate']);continue
        f=copy.deepcopy(old);method=f['method'];model=base.mapper(method);lo,hi,_=study.specification(method)
        tr=study.training_data(f['train_families'],method);y=tr.target.to_numpy(float);sw=np.sqrt(tr.loss_weight.to_numpy(float))
        start=time.perf_counter();before=model.counters();cache={};failures=[]
        def calc(w):
            key=tuple(w)
            if key not in cache:
                values=[];jac=[]
                for row in tr.itertuples():
                    try:a=model.point(row,w)
                    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                        failures.append(dict(obs_id=row.obs_id,parameters=w,error=repr(exc)));raise
                    values.append(a.values[0]);jac.append(a.jacobian[0])
                cache.clear();cache[key]=(np.array(values),np.array(jac))
            return cache[key]
        starts=[t['initial'] for t in old['trials']]
        if old['parameters'] is not None:starts.append(old['parameters'])
        trials=[]
        for initial in starts:
            try:
                op=least_squares(lambda w:(calc(w)[0]-y)*sw,initial,jac=lambda w:calc(w)[1]*sw[:,None],bounds=(lo,hi),
                    max_nfev=360,ftol=1e-7,xtol=1e-7,gtol=1e-7)
                trials.append(dict(initial=initial,parameters=op.x,loss=float(op.fun@op.fun),success=bool(op.success),
                    nfev=int(op.nfev),njev=int(op.njev),optimality=float(op.optimality),active_mask=op.active_mask,message=op.message))
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                trials.append(dict(initial=initial,parameters=initial,loss=None,success=False,error=repr(exc)))
        finite=[t for t in trials if t['loss'] is not None]
        if not finite:
            energy.write(OUT/'failed_refinement'/path.name,dict(original=old,new_trials=trials,failures=failures));raise RuntimeError('All refinement starts failed')
        best=min(finite,key=lambda t:t['loss']);oldbest=min((t['loss'] for t in old['trials'] if t['loss'] is not None),default=np.inf)
        gate=dict(passed=best['loss']<=oldbest+1e-10,key=f['key'],method=method,original_objective=oldbest,new_objective=best['loss'],
            new_starts=len(trials),unconverged=sum(not t['success'] for t in trials),selected_converged=best['success'])
        energy.write(OUT/'gates'/path.name,gate);gates.append(gate)
        assert gate['passed'],'Preserve failed gate; do not pick by outer score'
        parameters=best['parameters'];training_counts={k:model.counters()[k]-before[k] for k in before};predictions=[]
        for row in study.PRED.itertuples():
            try:
                a=model.point(row,parameters);dg={k:a.diagnostics[k] for k in ('root_residual_log','jacobian_condition','free_n_cm3',
                    'free_p_cm3','Tf_K','measurement_T_K','Nd_cm3','profile_sha256','localized_fraction','donor_subgroups',
                    'same_physical_state_for_output','hall_factor','coherent_transport','statistics_scope')}
                predictions.append(dict(obs_id=row.obs_id,predicted=float(a.values[0]),charge_outputs=a.values[1:],numeric=True,
                    physical_valid=True,diagnostics=dg,transport_values=a.transport_values))
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                predictions.append(dict(obs_id=row.obs_id,predicted=None,numeric=False,physical_valid=False,error=repr(exc)))
        counters={k:model.counters()[k]-before[k] for k in before}
        f.update(parameters=parameters,trials=trials,original_trials=old['trials'],state_failures=failures,predictions=predictions,
            seconds=time.perf_counter()-start,counters=counters,training_counters=training_counts,
            cached_prediction_counters={k:counters[k]-training_counts[k] for k in counters},
            original_transport_enrichment_counters=old['transport_enrichment_counters'],
            original_transport_enrichment_seconds=old['transport_enrichment_seconds'],
            transport_enrichment_counters={k:0 for k in counters},transport_enrichment_seconds=0.,
            refinement=dict(original_fit_path=str(path.relative_to(ROOT)),original_fit_sha256=energy.sha(path),gate=gate))
        energy.write(dest,f);executions.append(dict(**gate,seconds=f['seconds'],counters=counters))
        print('refined',method,gate,flush=True)
    energy.write(OUT/'refinement_gate.json',dict(passed=all(g['passed'] for g in gates),affected_records=len(affected),
        additional_starts=sum(g['new_starts'] for g in gates),unconverged_starts=sum(g['unconverged'] for g in gates),
        best_nonconverged=sum(not g['selected_converged'] for g in gates),records=gates))
    energy.write(OUT/'refinement_execution_complete.json',dict(created_utc=energy.now(),records=executions,goal_complete=False))
    return True


def select():
    assert json.loads((OUT/'refinement_gate.json').read_text())['passed']
    rows=load_composite();lookup={(f['method'],tuple(sorted(f['train_families'])),json.dumps(f['config'],sort_keys=True)):f for f in rows.values()}
    def getfit(families,method,config):return lookup[method,tuple(sorted(families)),json.dumps(config,sort_keys=True)]
    study.fit=getfit;study.OUT=OUT
    for method in study.METHODS:study.select(method)
    energy.write(OUT/'selection_execution_complete.json',dict(created_utc=energy.now(),methods=len(study.METHODS),goal_complete=False))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--select-only',action='store_true');a=ap.parse_args()
    if a.select_only:select()
    elif refine():select()


if __name__=='__main__':main()
