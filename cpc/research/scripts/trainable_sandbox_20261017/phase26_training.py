"""Train absolute-share prediction and matched strong activation controls."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,json,time,argparse
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.pool_share_model import PoolShareModel
from sandbox.pool_share_controls import PoolReferenceModel,DirectActivationModel
from sandbox.coherent_fermi import published_coherent_profile
from csv_io_v2 import read_result_csv as read
import phase25_training as parent
import phase2_energy as e

OUT=e.BASE/'phase26_pool_share';TABLE=parent.TABLE;REG=(0.,.1,1.,None)
SPECS={'GrowthReference':dict(kind='reference',mode='growth',basis=None),
    'ActivationReference':dict(kind='activation',basis='none')}
for mode,prefix in [('effective','ShareEffective'),('growth','ShareGrowth')]:
    for b in ('constant','dose','process'):SPECS[prefix+b.title()]=dict(kind='share',mode=mode,basis=b)
for b in ('constant','dose','process'):SPECS['Activation'+b.title()]=dict(kind='activation',basis=b)
METHODS=tuple(SPECS);MODELS={}


def configs(method):return [None] if method.endswith('Reference') else REG


def model(method):
    if method not in MODELS:
        if method=='ParentReference':obj=PoolReferenceModel(temperature_mode='effective',profile=published_coherent_profile())
        else:
            s=SPECS[method]
            if s['kind']=='reference':obj=PoolReferenceModel(temperature_mode=s['mode'],profile=published_coherent_profile())
            elif s['kind']=='activation':obj=DirectActivationModel(basis=s['basis'])
            else:obj=PoolShareModel(basis=s['basis'],temperature_mode=s['mode'],profile=published_coherent_profile())
        MODELS[method]=obj
    return MODELS[method]


def refmethod(method):
    s=SPECS[method]
    return 'ActivationReference' if s['kind']=='activation' else ('ParentReference' if s['mode']=='effective' else 'GrowthReference')


def forward(method,w,table,*,penalty=False):
    m=model(method);y=[];J=[];delta=[];dJ=[];valid=[]
    for row in table.itertuples():
        p=m.point(row,w,require_observation=False);y.append(p.values[0]);J.append(p.jacobian[0]);valid.append(p.diagnostics['observation_domain_valid'])
        if penalty and method not in ('GrowthReference','ActivationReference','ParentReference'):
            rm=refmethod(method);nb=2 if rm in ('ParentReference','ActivationReference') else 1
            r=model(rm).point(row,np.asarray(w)[:nb],require_observation=False)
            dj=p.jacobian[0].copy();dj[:nb]-=r.jacobian[0]
            delta.append(p.values[0]-r.values[0]);dJ.append(dj)
        elif penalty:delta.append(0.);dJ.append(np.zeros(len(w)))
    return np.array(y),np.array(J),np.array(delta),np.array(dJ),np.array(valid,bool)


def path_for(scheme,units,method,reg):
    key=parent.digest(dict(protocol=e.sha(OUT/'training_protocol.json'),scheme=scheme,units=sorted(units),method=method,reg=reg))
    return key,OUT/'fits'/(key+'.json')


def inherited_reference(scheme,units):
    key,p=parent.fit_path(scheme,units,'FixedPhysical',None)
    assert p.exists();r=json.loads(p.read_text())
    assert tuple(r['train_units'])==tuple(sorted(units)) and r['method']=='FixedPhysical'
    return r,p


def bounds(method):
    m=model(method);s=SPECS[method]
    if s['kind']=='activation':lo=[12.,-12.]+[-12.]*(len(m.parameter_names)-2);hi=[21.,12.]+[12.]*(len(m.parameter_names)-2)
    elif s['mode']=='growth':lo=[14.]+[-12.]*(len(m.parameter_names)-1);hi=[18.]+[12.]*(len(m.parameter_names)-1)
    else:lo=[-6.,14.]+[-12.]*(len(m.parameter_names)-2);hi=[6.,18.]+[12.]*(len(m.parameter_names)-2)
    return np.array(lo),np.array(hi)


def fit(scheme,units,method,reg):
    units=tuple(sorted(units));key,path=path_for(scheme,units,method,reg)
    if path.exists():return json.loads(path.read_text())
    table=parent.subset(scheme,units);obs=table.target.to_numpy(float);wg=parent.weights(table,scheme);sw=np.sqrt(wg)
    start=time.perf_counter();r=dict(key=key,scheme=scheme,train_units=units,train_families=sorted(table.family.unique()),
        training_obs_ids=table.obs_id.tolist(),method=method,reg=reg,evaluation_method=method,trials=[],state_failures=[],
        selected_success=False,training_valid=False,new_optimizer_starts=0,parameters=None)
    reference=method.endswith('Reference')
    if not reference and reg is None:
        rm=refmethod(method)
        if rm=='ParentReference':ref,rpath=inherited_reference(scheme,units)
        else:_,rpath=path_for(scheme,units,rm,None);assert rpath.exists();ref=json.loads(rpath.read_text())
        w=np.array(ref['parameters']);values,J,delta,dj,valid=forward(rm,w,table,penalty=True)
        r.update(parameters=w.tolist(),evaluation_method=rm,reference_fallback=True,
            fallback_scope='separate explicit reference candidate;not the infinite-penalty limit of the absolute-share family',
            reference_fit_path=str(rpath.relative_to(ROOT)),reference_fit_sha256=e.sha(rpath),
            selected_success=ref['selected_success'],training_valid=bool(valid.all()))
    else:
        if method=='GrowthReference':starts=[np.array([15.]),np.array([17.])]
        elif method=='ActivationReference':starts=[np.array([15.,0.]),np.array([18.,0.])]
        else:
            rm=refmethod(method)
            if rm=='ParentReference':ref,_=inherited_reference(scheme,units)
            else:_,rp=path_for(scheme,units,rm,None);assert rp.exists();ref=json.loads(rp.read_text())
            base=np.array(ref['parameters']);nfield=len(model(method).parameter_names)-len(base)
            constants=[-.7,1.4] if SPECS[method]['kind']=='activation' else [2.4,3.9]
            starts=[np.r_[base,c,np.zeros(nfield-1)] for c in constants]
        lo,hi=bounds(method);failures=[];cache={};continuations=0;calls=0
        tracked=[method]+([] if reference else [refmethod(method)])
        before={name:model(name).counters() for name in tracked}
        def calculate(w):
            nonlocal continuations,calls
            k=tuple(w)
            if k not in cache:
                calls+=1
                try:
                    q=forward(method,w,table,penalty=True);continuations+=int((~q[4]).sum());cache[k]=q
                except (ValueError,RuntimeError,ArithmeticError) as exc:
                    failures.append(dict(parameters=np.asarray(w).tolist(),error=repr(exc)));raise
            return cache[k]
        def residual(w):
            v,J,d,dJ,valid=calculate(w)
            return np.r_[(v-obs)*sw,np.sqrt(reg)*d*sw] if reg is not None and reg>0 else (v-obs)*sw
        def jac(w):
            v,J,d,dJ,valid=calculate(w)
            return np.r_[J*sw[:,None],np.sqrt(reg)*dJ*sw[:,None]] if reg is not None and reg>0 else J*sw[:,None]
        def run(initial,cap,stage):
            try:
                op=least_squares(residual,initial,jac=jac,bounds=(lo,hi),x_scale='jac',max_nfev=cap,ftol=1e-7,xtol=1e-7,gtol=1e-7)
                return dict(initial=np.asarray(initial).tolist(),parameters=op.x.tolist(),success=bool(op.success),loss=float(op.fun@op.fun),
                    nfev=int(op.nfev),optimality=float(op.optimality),message=op.message,cap=cap,stage=stage,active_bounds=op.active_mask.tolist())
            except (ValueError,RuntimeError,ArithmeticError) as exc:
                return dict(initial=np.asarray(initial).tolist(),success=False,loss=None,error=repr(exc),cap=cap,stage=stage)
        trials=[run(s,120,'initial') for s in starts]
        if any(not t['success'] for t in trials):
            finite=[t for t in trials if t['loss'] is not None];best=[min(finite,key=lambda t:t['loss'])['parameters']] if finite else []
            trials.extend(run(s,360,'uniform_refinement') for s in starts+best)
        finite=[t for t in trials if t['loss'] is not None];best=min(finite,key=lambda t:t['loss']) if finite else None
        r.update(trials=trials,state_failures=failures,new_optimizer_starts=len(trials),reference_fallback=False,
            selected_success=bool(best and best['success']),selected_loss=best['loss'] if best else None,
            numerical_continuation_trial_states=continuations,objective_forward_calls=calls)
        if best:
            w=np.array(best['parameters']);values,J,delta,dj,valid=forward(method,w,table,penalty=True)
            r.update(parameters=w.tolist(),training_valid=bool(valid.all()),
                solver_counters={name:{k:model(name).counters()[k]-value for k,value in before[name].items()} for name in tracked})
    if r['parameters'] is not None:
        w=np.array(r['parameters']);values,J,delta,dj,valid=forward(r['evaluation_method'],w,table,penalty=True)
        singular=np.linalg.svd(J*sw[:,None],compute_uv=False)
        r.update(training_nH_MAE=float(abs(values-obs)@wg),training_nH_MSE=float((values-obs)**2@wg),
            rms_response_change=float(np.sqrt(delta**2@wg)),selected_training_values=values.tolist(),selected_training_valid=valid.tolist(),
            weighted_prediction_jacobian_singular_values=singular.tolist(),prediction_jacobian_numeric_rank=int(np.linalg.matrix_rank(J*sw[:,None])))
    r['seconds']=time.perf_counter()-start;e.write(path,r)
    print('fit26',scheme,method,reg,len(units),'sec',round(r['seconds'],2),'valid',r['training_valid'],'stopped',r['selected_success'],flush=True)
    return e.clean(r)


def freeze():
    for name in ('numerical_gates.json','learning_gates_v2.json'):
        a=json.loads((OUT/name).read_text());assert a['passed']
        for n,h in a['code_hashes'].items():assert e.sha(ROOT/n)==h,n
    assert json.loads((OUT/'reachability/complete.json').read_text())['passed']
    code=[Path(__file__),ROOT/'src/sandbox/pool_share_model.py',ROOT/'src/sandbox/pool_share_controls.py',ROOT/'src/sandbox/absolute_pool_share.py',
        ROOT/'docs/ML_SANDBOX_POOL_SHARE_PROTOCOL_20261018.md']
    inputs=[OUT/'numerical_gates.json',OUT/'learning_gates_v2.json',OUT/'reachability/complete.json',
        e.BASE/'phase25_shared_process/data/conditions33.csv',e.BASE/'phase25_shared_process/data/episodes33.csv',e.BASE/'artifact_manifest_phase25.json']
    refs={}
    for s in ('DOI','cluster'):
        for units in parent.contexts(s):
            _,p=inherited_reference(s,units);refs[str(p.relative_to(ROOT))]=e.sha(p)
    p=dict(created_utc=e.now(),methods=SPECS,regularization=REG,reference_fallback='explicit discrete candidate;not infinite limit forabsolute-share',
        dataset_conditions=33,DOIs=6,related_groups=5,primary_scenario='effective;only internal coordinate changes from previous main task',
        secondary_scenario='nominalgrowth reference,not measuredfreezinghistory;not promoted by outerresults',
        old19controls_preserved=True,new_methods=len(METHODS),planned_fit_records=1444,planned_nonlinear_fits=1102,
        supervision='nHonly;no newmu/Tlabels;sourcebudgets remain perDOI0/1/2,common20',
        limits='two starts120,uniform original2+best360 onanynonstop;allstates/failures retained;max_nfev upper not equalactualcost',
        gamma_selection='training-inner units only;zeroanchor macroMAE;source lambda later selected onallowed roles only',
        activation_bounds='empirical background12..21,Hlogit/theta[-12,12];not measured donortruth',
        physical_bounds='effective logitTf[-6,6],bg14..18,absolute theta[-12,12],combined numericpsi[-80,80]',
        cost='oldphysicalenergyprior separate from sameHalllabel/optimizer ceilings',
        goal_complete=False,no_new_DFT_MD=True,GPU_authorized='GPU1only',GPU_used=False,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in code},input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in inputs},
        inherited_parent_fit_sha256=refs)
    path=OUT/'training_protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        for k in ('code_hashes','input_hashes','inherited_parent_fit_sha256'):assert old[k]==p[k],k
    else:e.write(path,p)


def task(args):return fit(*args)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();freeze()
    refs=[];others=[]
    for scheme in ('DOI','cluster'):
        for units in parent.contexts(scheme):
            for m in METHODS:
                for reg in configs(m):(refs if m.endswith('Reference') else others).append((scheme,units,m,reg))
    assert len(refs)==76 and len(others)==1368
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        completed=list(pool.map(task,refs));completed+=list(pool.map(task,others))
    assert len(completed)==1444
    value=dict(created_utc=e.now(),fit_records=len(completed),actual_optimizer_starts=sum(r['new_optimizer_starts'] for r in completed),
        selected_not_stopped=sum(not r['selected_success'] for r in completed),invalid_training_fits=sum(not r['training_valid'] for r in completed),goal_complete=False)
    e.write(OUT/'training_complete.json',value);print('Training26 complete',value,flush=True)


if __name__=='__main__':main()
