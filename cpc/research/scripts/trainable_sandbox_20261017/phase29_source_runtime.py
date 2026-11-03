"""Isolated source adaptation, reading only a task and the physical catalogue."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,time,builtins,io,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from sandbox.joint_saved_model import SavedJointModel
from sandbox.context_source import solve_context_source,solve_affine_context
from sandbox.kroger_db import NPZ


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def evaluate_task(task,lambdas):
    clock=time.perf_counter();errors=[];cache={};model_calls=0;physical_calls=0;episodes=[]
    nodes=[SimpleNamespace(**r) for r in task['nodes']];kind=task['kind'];fit=task['fit']
    assert kind in ('offset','internal','affine')
    assert not any(k in fit for k in ('training_condition_ids','selected_training_values','training_nH_case_weights'))
    assert not any(any(k in row for k in ('target','observed','observed_log_nH','mu_cm2Vs')) for row in task['nodes'])
    try:mapper=SavedJointModel(fit)
    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
        failure=repr(exc)
        for ep in task['episodes']:
            solutions=[dict(lam=lam,raw_shift=None,valid=False,error=failure,
                records=[dict(numeric=False,valid=False) for _ in nodes])
                for lam in ([None] if ep['budget']==0 else lambdas)]
            episodes.append(dict(budget=ep['budget'],anchor_condition_ids=ep['anchor_condition_ids'],
                anchor_indices=ep['anchor_indices'],anchor_labels=ep['anchor_labels'],
                solutions=solutions,solver_metadata=None,failure=failure))
        return dict(id=task['id'],fit_key=fit['key'],kind=kind,episodes=episodes,point_failures=[failure],
            actual_model_point_calls=0,actual_physical_point_calls=0,seconds=time.perf_counter()-clock)

    def points(z):
        nonlocal model_calls,physical_calls
        z=float(z)
        if z not in cache:
            values=[]
            base=points(0.) if z!=0. and kind=='offset' else None
            for i,row in enumerate(nodes):
                try:
                    if base is not None:
                        p=None if base[i] is None else mapper.offset_point(row,base[i],z,require_observation=False)
                    else:
                        model_calls+=1;physical_calls+=int(mapper.physical)
                        p=mapper.point(row,source_kind='share' if kind=='internal' else 'offset' if kind=='offset' else 'none',
                            shift=z,require_observation=False)
                    values.append(p)
                except (ValueError,RuntimeError,ArithmeticError,np.linalg.LinAlgError) as exc:
                    errors.append(dict(shift=z,node=i,error=repr(exc)));values.append(None)
            cache[z]=values
        return cache[z]

    def forward(z):
        values=points(z)
        return (np.array([p.log_nH if p is not None else np.nan for p in values]),
            np.array([p.source_derivative if p is not None else np.nan for p in values]),
            np.array([p is not None and p.valid for p in values]))

    def payload(values):
        result=[]
        for p in values:
            if p is None:result.append(dict(numeric=False,valid=False));continue
            dg={k:p.diagnostics[k] for k in (
                'coherent_transport','Tf_K','physical_state_available','same_physical_state_for_output',
                'source_coordinate','source_kind','source_shift','observation_offset_dex',
                'single_electron_Hall_valid','foreign_atom_relative_error','root_residual_log','jacobian_condition',
                'critical_free_fraction','excess_share_logit','share_basis','actual_evaluation_method',
                'extra_mobility_cm2Vs','shared_log10_mu0_shift','transport_head','transport_head_offset_dex',
                'material_Born_validity_established','transport_scope','hall_factor') if k in p.diagnostics}
            result.append(dict(numeric=bool(np.isfinite([p.log_nH,p.log_muH,p.log_sigma]).all()),
                valid=p.valid,predicted=p.log_nH,predicted_log_mu=p.log_muH,
                predicted_log_sigma=p.log_sigma,predicted_log_mu_CT=p.log_mu_CT,
                source_derivative=p.source_derivative,
                source_transport_derivative=p.source_transport_derivative,diagnostics=dg))
        return result

    for ep in task['episodes']:
        meta=None;failure=None
        # Scenario rows repeat the same physical anchor label. Every condition
        # within a DOI has the same scenario count, so these means give equal
        # condition weight and do not increase the physical anchor budget.
        assert len(ep['anchor_condition_ids'])==ep['budget']
        if ep['budget']==0:
            solutions=[dict(lam=None,raw_shift=0.,valid=True,status='no_anchor')]
        elif kind=='affine':
            try:
                base=points(0.)
                if any(p is None or not p.valid for p in base):raise ValueError('Invalid affine baseline')
                meta=solve_affine_context([p.log_nH for p in base],
                    np.log10([r.incorporated_cm3 for r in nodes]),
                    ep['anchor_indices'],ep['anchor_labels'],lambdas)
                solutions=meta['solutions']
                for solution in solutions:
                    values=[]
                    for row,p,delta in zip(nodes,base,solution['response_delta']):
                        try:values.append(mapper.offset_point(row,p,delta,require_observation=False))
                        except (ValueError,RuntimeError,ArithmeticError,np.linalg.LinAlgError) as exc:
                            errors.append(dict(stage='affine',error=repr(exc)));values.append(None)
                    solution.update(raw_shift=None,records=payload(values),
                        valid=bool(solution['finite_response_bound_valid'] and all(p is not None and p.valid for p in values)))
            except (ValueError,RuntimeError,ArithmeticError,np.linalg.LinAlgError) as exc:
                failure=repr(exc);solutions=[]
        else:
            try:
                meta=solve_context_source(forward,ep['anchor_indices'],ep['anchor_labels'],lambdas,
                    linear=kind=='offset',nodes=33,local_candidates=3,maxiter=80)
                solutions=meta['solutions']
            except (ValueError,RuntimeError,ArithmeticError,np.linalg.LinAlgError) as exc:
                failure=repr(exc);solutions=[]
        for lam in ([None] if ep['budget']==0 else lambdas):
            if not any(s['lam']==lam for s in solutions):
                solutions.append(dict(lam=lam,raw_shift=0. if lam is None else None,
                    valid=lam is None,error=failure or 'no returned solution'))
        for solution in solutions:
            if 'records' not in solution:
                solution['records']=(payload(points(solution['raw_shift']))
                    if solution.get('valid') and solution.get('raw_shift') is not None
                    else [dict(numeric=False,valid=False) for _ in nodes])
            solution['valid']=bool(solution.get('valid') and all(p['numeric'] and p['valid'] for p in solution['records']))
            if not solution['valid']:
                for p in solution['records']:p['valid']=False
        episodes.append(dict(budget=ep['budget'],anchor_condition_ids=ep['anchor_condition_ids'],
            anchor_indices=ep['anchor_indices'],anchor_labels=ep['anchor_labels'],solutions=solutions,
            solver_metadata=meta,failure=failure))
    return dict(id=task['id'],fit_key=fit['key'],kind=kind,episodes=episodes,point_failures=errors,
        actual_model_point_calls=model_calls,actual_physical_point_calls=physical_calls,
        actual_source_vectors=len(cache),seconds=time.perf_counter()-clock)


def main():
    taskpath=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve()
    original=builtins.open;original_io=io.open;denied=[];physical=NPZ.resolve();physical_reads=[]
    def guarded(factory):
        def opened(file,mode='r',*args,**kwargs):
            if isinstance(file,(str,bytes,Path)) and ('r' in mode or '+' in mode):
                path=Path(file).resolve()
                if path==physical:physical_reads.append(str(path))
                if 'results' in path.parts and path not in (taskpath,physical):
                    denied.append(str(path));raise RuntimeError('Result/label file read forbidden: '+str(path))
            return factory(file,mode,*args,**kwargs)
        return opened
    builtins.open=guarded(original);io.open=guarded(original_io)
    packet=json.loads(taskpath.read_text())
    assert hashlib.sha256(physical.read_bytes()).hexdigest()==packet['physical_reference_sha256']
    records=[]
    for task in packet['tasks']:
        record=evaluate_task(task,packet['lambdas']);records.append(record)
        print('source29',task['id'],task['kind'],'sec',round(record['seconds'],2),
            'physical',record['actual_physical_point_calls'],flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(clean(dict(records=records,denied_reads=denied,
        physical_reference_sha256=packet['physical_reference_sha256'],
        physical_reference_read_count=len(physical_reads),runtime_query_labels=False)),ensure_ascii=False,indent=2))
    print('Source29 complete',len(records),'denied',len(denied),flush=True)


if __name__=='__main__':main()
