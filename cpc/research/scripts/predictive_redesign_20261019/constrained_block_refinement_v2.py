"""Training-only grid check for flat zero-texture/saturated-logit starts."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import sys,itertools,time,numpy as np,pandas as pd
from scipy.optimize import minimize
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import phase35_learning_v1 as spatial
import phase34_learning_v1 as overlay
import constrained_block_training_v1 as parent

OUT=c.BASE/'proposal_followup_20261021_v1/constrained_block_refinement_v2'
GRID=list(itertools.product([-12.,-4.,0.,4.,12.],[0.,1.,4.,12.,30.]))


def run_task(path):
    saved=c.document(path);method=saved['method'];held=saved['held'];prefix=np.array(saved['primary_parameters'])
    primary=c.document(c.ROOT/saved['primary_path']);rows=overlay.subset(primary['train_units']);start=time.perf_counter()
    q0=spatial.forward(method,np.r_[prefix,[-12.,0.]],rows,penalty=True);wn=c.learning.weights(rows)
    baseline=dict(values=q0['values'],nH_loss=float(wn@((q0['values'][:,0]-rows.target.to_numpy(float))**2)))
    probes=[];cache={};failures=[];attempts=0
    def calc(beta):
        nonlocal attempts
        kk=tuple(beta)
        if kk not in cache:
            attempts+=1
            try:q=parent.obj_and_constraints(method,prefix,beta,rows,baseline)
            except (ValueError,RuntimeError,ArithmeticError) as exc:failures.append(dict(beta=list(beta),error=repr(exc)));raise
            cache.clear();cache[kk]=q
        return cache[kk]
    candidates=[]
    if saved.get('selected'):
        candidates.append(dict(saved['selected'],kind='parent_selected'))
    for beta in GRID:
        try:
            val,g,cons,J,q=calc(np.array(beta));feasible=bool(q['valid'].all() and min(cons)>=-parent.FEAS_TOL)
            r=dict(beta=list(beta),objective=float(val),feasible=feasible,min_constraint=float(min(cons)))
            if feasible:candidates.append(dict(r,kind='feasible_grid'))
        except (ValueError,RuntimeError,ArithmeticError) as exc:r=dict(beta=list(beta),feasible=False,error=repr(exc))
        probes.append(r)
    refined=[];starts=0
    if candidates:
        best=min(candidates,key=lambda r:r['objective'])
        original=saved.get('selected');improves=original is None or best['objective']<original['objective']-1e-7
        if improves:
            starts+=1
            try:
                z=minimize(lambda b:calc(b)[0],best['beta'],jac=lambda b:calc(b)[1],method='SLSQP',bounds=[(-12.,12.),(0.,30.)],
                    constraints=[dict(type='ineq',fun=lambda b:calc(b)[2],jac=lambda b:calc(b)[3])],options=dict(maxiter=60,ftol=1e-9))
                r=calc(z.x);feasible=bool(r[4]['valid'].all() and min(r[2])>=-parent.FEAS_TOL)
                opt=dict(beta=z.x.tolist(),objective=float(r[0]),feasible=feasible,success=bool(z.success),kind='refined_SLSQP',
                    iterations=int(z.nit),message=z.message,min_constraint=float(min(r[2])))
                refined.append(opt)
                if opt['success'] and feasible:candidates.append(opt)
            except (ValueError,RuntimeError,ArithmeticError) as exc:refined.append(dict(error=repr(exc),success=False))
        selected=min(candidates,key=lambda r:r['objective'])
    else:selected=None
    result=dict(created_utc=c.e.now(),method=method,held=held,parent_path=str(path.relative_to(c.ROOT)),probes=probes,
        refined=refined,selected=selected,optimizer_starts=starts,attempted_training_forward_vectors=attempts,failures=failures,
        seconds=time.perf_counter()-start,no_query_in_grid_or_optimizer_selection=True)
    prediction=[]
    if selected:
        q=calc(np.array(selected['beta']))
        result.update(training_max_nH_drift=float(max(abs(q[4]['values'][:,0]-baseline['values'][:,0]))),
            training_nH_loss_change=float(wn@((q[4]['values'][:,0]-rows.target.to_numpy(float))**2)-baseline['nH_loss']),
            min_constraint=float(min(q[2])),training_valid=bool(q[4]['valid'].all()))
        prediction=parent.predict(method,np.r_[prefix,selected['beta']],held,method)
        if method=='ClusterCoupled':prediction.extend(parent.predict(method,np.r_[prefix,[-12.,0.]],held,'FrozenPrimary'))
    key=c.digest(dict(held=held,method=method))
    c.write(OUT/'records'/(key+'.json'),result);c.write(OUT/'predictions'/(key+'.json'),prediction)
    print('Grid-refined block',method,held,selected['kind'] if selected else None,
        'feasiblegrid',sum(r['feasible'] for r in probes),'starts',starts,'seconds',round(result['seconds'],1),flush=True)
    return result


def main():
    assert not (OUT/'protocol_before.json').exists()
    paths=sorted((parent.OUT/'secondary_fits').glob('*.json'));assert len(paths)==15
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),workers=4,grid=GRID,
        purpose='specific optimization-risk check:zero texture has even-symmetry flat derivative;association logit at-12 suppresses gradients',
        training_only_selection=True,not_independent_confirmation_after_v1=True,
        refinement='one SLSQP60 start only when grid gives a lower feasible training objective by>1e-7;retain parent and all feasible grid points',
        code_hashes=c.file_hashes([Path(__file__),Path(parent.__file__),Path(spatial.__file__),Path(overlay.__file__)]),
        input_hashes=c.file_hashes([parent.OUT/'complete.json',*paths])))
    with ProcessPoolExecutor(max_workers=4) as pool:
        records=[]
        for f in as_completed([pool.submit(run_task,p) for p in paths]):
            records.append(f.result());c.write(OUT/'status.json',dict(created_utc=c.e.now(),finished=len(records),planned=15))
    rows=[]
    for p in (OUT/'predictions').glob('*.json'):rows.extend(c.document(p))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'predictions.csv',index=False);summary=[]
    for variant,scope in itertools.product(('FrozenPrimary',*spatial.METHODS),('all_query','common20')):
        q=frame[frame.variant.eq(variant)];expected=c.cases()
        if scope=='common20':q=q[q.primary_common20.eq(True)];expected=expected[expected.primary_common20.eq(True)]
        complete=len(q)==len(expected) and set(q.case_id)==set(expected.case_id)
        r=dict(variant=variant,scope=scope,rows=len(q),complete=complete,numeric=int(q.numeric.eq(True).sum()),valid=int(q.valid.eq(True).sum()))
        for endpoint in ('nH','mu','rho'):r[endpoint+'_MAE']=c.group_score(q.to_dict('records'),endpoint,paired=endpoint!='nH',group_equal=False) if complete else None
        summary.append(r)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False);c.verify_hashes(c.document(OUT/'protocol_before.json'))
    c.write(OUT/'complete.json',dict(created_utc=c.e.now(),records=len(records),grid_forward_vectors=sum(r['attempted_training_forward_vectors'] for r in records),
        optimizer_starts=sum(r['optimizer_starts'] for r in records),prediction_rows=len(frame),new_primary_fits=0,
        no_unseen_prediction_protection_claim=True,stable_prediction_gain_established=False))


if __name__=='__main__':main()
