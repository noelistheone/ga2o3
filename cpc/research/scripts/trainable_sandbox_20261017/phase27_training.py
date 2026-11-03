"""Train charge-feasible inventory prediction and matched endpoint controls."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,json,time,argparse
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.charge_feasible_model_v2 import ChargeFeasibleModel
from sandbox.pool_share_controls import PoolReferenceModel,DirectActivationModel
from sandbox.coherent_fermi import published_coherent_profile
from csv_io_v2 import read_result_csv as read
import phase25_training as parent
import phase2_energy as e

OUT=e.BASE/'phase27_charge_feasible';TABLE=parent.TABLE;REG=(0.,.1,1.,None)
SPECS={}
for prefix,readout in [('Feasible','physical'),('Envelope','endpoint_envelope')]:
    for basis in ('constant','dose','process'):
        SPECS[prefix+basis.title()]=dict(kind='share' if prefix=='Feasible' else 'envelope',mode='effective',basis=basis,readout=readout)
METHODS=tuple(SPECS);MODELS={}


def configs(method):return [None] if method.endswith('Reference') else REG


def model(method):
    if method not in MODELS:
        if method=='ParentReference':obj=PoolReferenceModel(temperature_mode='effective',profile=published_coherent_profile())
        else:
            spec=SPECS[method]
            obj=ChargeFeasibleModel(basis=spec['basis'],profile=published_coherent_profile(),readout=spec['readout'])
        MODELS[method]=obj
    return MODELS[method]


def refmethod(method):return 'ParentReference'


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
            constants=[-.7,1.4]
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
    print('fit27',scheme,method,reg,len(units),'sec',round(r['seconds'],2),'valid',r['training_valid'],'stopped',r['selected_success'],flush=True)
    return e.clean(r)


def freeze():
    gate_names=('numerical_gates_v2.json','readout_gates.json','learning_gates.json')
    for name in gate_names:
        audit=json.loads((OUT/name).read_text());assert audit['passed'],name
        for path,h in audit['code_hashes'].items():assert e.sha(ROOT/path)==h,path
    manifest=e.BASE/'artifact_manifest_phase26.json'
    assert e.sha(manifest)=='4489891819c035c45b3bae9c128f6a9e79b6f63cd79742c755d3a42eaacc6bcc'
    code=[Path(__file__),ROOT/'src/sandbox/charge_feasible_model_v2.py',ROOT/'src/sandbox/charge_feasible_model.py',
        ROOT/'src/sandbox/charge_feasible_share.py',ROOT/'src/sandbox/charge_feasible_saved_model.py',
        ROOT/'src/sandbox/pool_share_controls.py',ROOT/'docs/ML_SANDBOX_CHARGE_FEASIBLE_PROTOCOL_20261018.md']
    inputs=[OUT/p for p in gate_names]+[parent.OUT/'data/conditions33.csv',parent.OUT/'data/episodes33.csv',manifest]
    refs={}
    for scheme in ('DOI','cluster'):
        for units in parent.contexts(scheme):
            _,path=inherited_reference(scheme,units);refs[str(path.relative_to(ROOT))]=e.sha(path)
    protocol=dict(created_utc=e.now(),methods=SPECS,regularization=REG,primary='FeasibleProcess_internal',
        reference_fallback='explicit same-training-unit reference;not a continuous infinite regularization limit',
        dataset_conditions=33,DOIs=6,related_groups=5,old36controls_preserved=True,
        planned_fit_records=912,planned_nonlinear_fits=684,new_variants=12,combined_variants=48,
        supervision='nHonly;no newmu/temperaturelabels;0/1/2 perDOI anchors,common20',
        starts='same6families:reference base +constant-.7or1.4;other slopes0;different oldcoordinates not claimed identical starts',
        optimizer='two starts120,uniform original2+best360 onanynonstop;ftol/xtol/gtol1e-7',
        gamma_selection='training-inner zeroanchor macroMAE only;retain all nonfinite scores',
        physical_bounds='logitTf[-6,6],log10bg[14,18],theta[-12,12],numericalpsi[-80,80]',
        floor='conditional n-type conducting scope;free_n1e13,originalHallfloor andcarrier gates retained',
        unknowns='actualthermalhistory,Hunknown0,effectiveoxygen,materialenergy/DOS/scattering remain conditional',
        new_DFT_MD=0,new_material_measurements=0,GPU_authorized='GPU1only',GPU_used=False,goal_complete=False,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in code},
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in inputs},inherited_parent_fit_sha256=refs)
    path=OUT/'training_protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        for key in ('code_hashes','input_hashes','inherited_parent_fit_sha256'):assert old[key]==protocol[key],key
    else:e.write(path,protocol)


def task(args):return fit(*args)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();freeze()
    refs=[];others=[]
    for scheme in ('DOI','cluster'):
        for units in parent.contexts(scheme):
            for m in METHODS:
                for reg in configs(m):(refs if m.endswith('Reference') else others).append((scheme,units,m,reg))
    assert len(refs)==0 and len(others)==912
    import os
    e.write(OUT/'training_started.json',dict(created_utc=e.now(),pid=os.getpid(),workers=a.workers,planned_records=912,planned_nonlinear_fits=684))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        completed=list(pool.map(task,refs));completed+=list(pool.map(task,others))
    assert len(completed)==912
    value=dict(created_utc=e.now(),fit_records=len(completed),actual_optimizer_starts=sum(r['new_optimizer_starts'] for r in completed),
        selected_not_stopped=sum(not r['selected_success'] for r in completed),invalid_training_fits=sum(not r['training_valid'] for r in completed),goal_complete=False)
    e.write(OUT/'training_complete.json',value);print('Training27 complete',value,flush=True)


if __name__=='__main__':main()
