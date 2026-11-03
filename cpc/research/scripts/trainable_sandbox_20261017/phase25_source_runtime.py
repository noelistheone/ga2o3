"""Fresh-process source inference: task inputs and anchor labels only."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,time,builtins,io,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from sandbox.context_saved_model import SavedContextModel
from sandbox.context_source import solve_context_source,solve_affine_context
from sandbox.kroger_db import NPZ


def main():
    taskpath=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve()
    original=builtins.open;original_io=io.open;denied=[];physical=NPZ.resolve();physical_reads=[]
    def guarded(factory):
        def opened(file,mode='r',*args,**kwargs):
            if isinstance(file,(str,bytes,Path)) and ('r' in mode or '+' in mode):
                p=Path(file).resolve()
                if p==physical:physical_reads.append(str(p))
                if 'results' in p.parts and p not in (taskpath,physical):
                    denied.append(str(p));raise RuntimeError('Runtime result-table/label-file read forbidden: '+str(p))
            return factory(file,mode,*args,**kwargs)
        return opened
    builtins.open=guarded(original);io.open=guarded(original_io)
    packet=json.loads(taskpath.read_text());records=[];models={}
    assert hashlib.sha256(physical.read_bytes()).hexdigest()==packet['physical_reference_sha256']
    for task in packet['tasks']:
        clock=time.perf_counter();f=task['fit'];key=f['key'];errors=[];cache={};point_calls=0
        try:
            if key not in models:models[key]=SavedContextModel(f)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            failure=repr(exc);episodes=[]
            for ep in task['episodes']:
                lambdas=[None] if ep['budget']==0 else packet['lambdas']
                solutions=[dict(lam=lam,raw_shift=None,valid=False,error=failure,
                    records=[dict(numeric=False,valid=False) for _ in task['nodes']]) for lam in lambdas]
                episodes.append(dict(ep,solutions=solutions,solver_metadata=None,failure=failure))
            records.append(dict(id=task['id'],fit_key=key,kind=task['kind'],episodes=episodes,
                point_failures=[dict(stage='saved_model',error=failure)],actual_physical_point_calls=0,
                actual_source_vectors=0,seconds=time.perf_counter()-clock))
            continue
        model=models[key];nodes=[SimpleNamespace(**r) for r in task['nodes']]
        kind=task['kind'];base=None
        def points(z):
            nonlocal point_calls,base
            z=float(z)
            if z not in cache:
                if kind=='offset' and z!=0.:
                    b=points(0.)
                    values=[]
                    for i,row in enumerate(nodes):
                        try:values.append(None if b[i] is None else model.offset_point(row,b[i],z,require_observation=False))
                        except (ValueError,RuntimeError,np.linalg.LinAlgError,OverflowError) as exc:
                            errors.append(dict(shift=z,node=i,error=repr(exc)));values.append(None)
                else:
                    values=[]
                    for i,row in enumerate(nodes):
                        point_calls+=1
                        try:values.append(model.point(row,source_kind='none' if kind=='affine' else kind,shift=z,require_observation=False))
                        except (ValueError,RuntimeError,np.linalg.LinAlgError,OverflowError) as exc:
                            errors.append(dict(shift=z,node=i,error=repr(exc)));values.append(None)
                cache[z]=values
            return cache[z]
        def forward(z):
            p=points(z)
            return (np.array([a.log_nH if a is not None else np.nan for a in p]),
                np.array([a.source_derivative if a is not None else np.nan for a in p]),
                np.array([a is not None and a.valid for a in p]))
        def payload(z):
            values=[]
            for p in points(z):
                if p is None:values.append(dict(numeric=False,valid=False));continue
                diag={k:p.diagnostics[k] for k in ('context_field_basis','context_field_value','Tf_K','pO2_atm',
                    'foreign_atom_relative_error','max_parent_fraction','same_physical_state_for_output',
                    'observation_offset_dex','single_electron_Hall_valid','hall_factor','free_n_cm3','free_p_cm3',
                    'physical_state_available','root_residual_log','jacobian_condition') if k in p.diagnostics}
                values.append(dict(numeric=True,valid=p.valid,predicted=p.log_nH,predicted_log_mu=p.log_muH,
                    predicted_log_sigma=p.log_sigma,predicted_log_mu_CT=p.log_mu_CT,predicted_log_sigma_CT=p.log_sigma_CT,
                    source_derivative=p.source_derivative,diagnostics=diag))
            return values
        episodes=[]
        for ep in task['episodes']:
            failure=None;meta=None
            if ep['budget']==0:
                solutions=[dict(lam=None,raw_shift=0.,valid=True,status='no_anchor')]
            elif kind=='affine':
                try:
                    base=points(0.)
                    if any(p is None for p in base):raise ValueError('Affine base point failed')
                    meta=solve_affine_context([p.log_nH for p in base],np.log10([r.incorporated_cm3 for r in nodes]),
                        ep['anchor_indices'],ep['anchor_labels'],packet['lambdas'])
                    solutions=meta['solutions']
                    for solution in solutions:
                        values=[]
                        for row,b,delta in zip(nodes,base,solution['response_delta']):
                            try:
                                p=model.offset_point(row,b,delta,require_observation=False)
                                values.append(dict(numeric=True,valid=p.valid,predicted=p.log_nH,predicted_log_mu=p.log_muH,
                                    predicted_log_sigma=p.log_sigma,predicted_log_mu_CT=p.log_mu_CT,predicted_log_sigma_CT=p.log_sigma_CT,
                                    source_derivative=None,diagnostics=dict(physical_state_available=p.diagnostics.get('physical_state_available'),
                                        source_parameter_count=2,observation_offset_dex=p.diagnostics.get('observation_offset_dex'))))
                            except (ValueError,RuntimeError,np.linalg.LinAlgError,OverflowError) as exc:
                                errors.append(dict(stage='affine',error=repr(exc)));values.append(dict(numeric=False,valid=False))
                        solution.update(records=values,raw_shift=None,
                            valid=bool(solution['finite_response_bound_valid'] and all(p['numeric'] and p['valid'] for p in values)))
                        if not solution['valid']:
                            for p in solution['records']:p['valid']=False
                except (ValueError,RuntimeError,np.linalg.LinAlgError,OverflowError) as exc:
                    failure=repr(exc)
                    solutions=[dict(lam=lam,raw_shift=None,valid=False,error=failure,records=[dict(numeric=False,valid=False) for _ in nodes]) for lam in packet['lambdas']]
            else:
                try:
                    meta=solve_context_source(forward,ep['anchor_indices'],ep['anchor_labels'],packet['lambdas'],
                        linear=kind=='offset',nodes=33,local_candidates=3,maxiter=80)
                    solutions=meta['solutions']
                except (ValueError,RuntimeError,np.linalg.LinAlgError,OverflowError) as exc:
                    failure=repr(exc);solutions=[]
                # Preserve a zero-update candidate and all failed candidates.
                for lam in packet['lambdas']:
                    if not any(s['lam']==lam for s in solutions):
                        solutions.append(dict(lam=lam,raw_shift=0. if lam is None else None,
                            valid=lam is None,error=failure or 'source solve did not return candidate'))
            for solution in solutions:
                if 'records' in solution:continue
                solution['records']=payload(solution['raw_shift']) if solution.get('valid') and solution.get('raw_shift') is not None else [dict(numeric=False,valid=False) for _ in nodes]
            episodes.append(dict(budget=ep['budget'],anchor_indices=ep['anchor_indices'],anchor_labels=ep['anchor_labels'],
                solutions=solutions,solver_metadata=meta,failure=failure))
        records.append(dict(id=task['id'],fit_key=key,kind=kind,episodes=episodes,point_failures=errors,
            actual_physical_point_calls=point_calls,actual_source_vectors=len(cache),seconds=time.perf_counter()-clock))
        print('source25',task['id'],kind,'sec',round(time.perf_counter()-clock,2),'points',point_calls,flush=True)
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):return [clean(v) for v in x]
        if isinstance(x,np.ndarray):return clean(x.tolist())
        if isinstance(x,np.generic):return clean(x.item())
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(clean(dict(records=records,denied_reads=denied,physical_reference_sha256=packet['physical_reference_sha256'],
        physical_reference_read_count=len(physical_reads),runtime_query_labels=False)),ensure_ascii=False,indent=2))
    print('source25 runtime complete',len(records),'denied',len(denied),flush=True)


if __name__=='__main__':main()
