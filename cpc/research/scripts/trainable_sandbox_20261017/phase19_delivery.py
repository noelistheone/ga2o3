"""Fit/export a full-source collection, retaining earlier energy-learning tools.

This is deployment training on known sources, not a new accuracy benchmark.
No earlier stage is written. New-source labels are not read here.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
import os,sys,json,time,argparse,hashlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from scipy.optimize import least_squares
from scipy.special import logit
from csv_io_v2 import read_result_csv as read
from sandbox.property_predictor_v5 import make_model
from sandbox.material_profile import PUBLISHED_PROFILE
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.trainable_predictor_v2 import structure_fingerprint,supplied_energy_fingerprint
from sandbox.kroger_db import load,NPZ
from sandbox.scalar_profile_adaptation import ScalarProfile
import phase16_prediction as inputs
import phase2_energy as energy

BASE=energy.BASE;OUT=BASE/'phase19_delivery';BUNDLE=OUT/'predictor_bundle_v5'
SPECS={
 'original_RT':dict(kind='legacy',definition=dict(fraction='zero',training='RT')),
 'binding_screen_joint':dict(kind='legacy',definition=dict(fraction='one',binding='screen',training='joint')),
 'HallII_original_RT':dict(kind='coherent',definition=dict(fraction='zero',training='RT',readout='ionized')),
 'HallExtra_original_RT':dict(kind='coherent',definition=dict(fraction='zero',training='RT',readout='ionized_extra')),
 'Thermal_binding_joint_effective_g1':dict(kind='thermal',definition=dict(fraction='one',binding='screen',training='joint',
     readout='ionized_extra',thermal_mode='effective',gamma=1.)),
 'stock_const_noext_RT':dict(kind='stock',definition=dict(field='parent',features='constant',training='RT',thermal=False)),
}
ENERGY_METHODS=('ridge_energy_tuned','goal_ridge_2','goal_mean_2','ordinary_kernel','nonlinear_kernel','ridge','full_library')
LAM=(0.,.001,.01,.1,1.,10.,np.inf)
DATA=read(BASE/'dataset_v1.csv');ROLES=read(BASE/'episodes_v1.csv');TARGETS=inputs.TARGETS
FAMILIES=sorted(DATA.family.unique())


def write(p,x):energy.write(p,x)


def record(method):
    d=SPECS[method];p=published_coherent_profile() if d['kind'] in ('coherent','thermal') else PUBLISHED_PROFILE
    return dict(method=method,**d,profile=p.to_dict())


def spec(method):
    joint=SPECS[method]['definition'].get('binding')=='screen'
    lo=[-6.,14.];hi=[6.,18.];starts=[]
    if joint:lo += [np.log10(.003),0.];hi += [np.log10(.2),5.]
    stock=SPECS[method]['kind']=='stock'
    if stock:lo += [-24.];hi += [24.]
    for T,B,E,s,psi in ((1400.,16.,.03,.2,0.),(1750.,15.5,.06,1.,8.)):
        w=[logit((np.log10(T)-np.log10(600.))/(np.log10(1950.)-np.log10(600.))),B]
        if joint:w += [np.log10(E),s]
        if stock:w += [psi]
        starts.append(w)
    return lo,hi,starts


def training(method):
    d=SPECS[method]['definition'];pieces=[]
    for endpoint,kind,factor in (('nH','RT',1.),('nH','curve',.1 if d['training']=='joint' else 0.),
            ('muH','RT',d.get('gamma',0.)),('muH','curve',.1*d.get('gamma',0.))):
        if not factor:continue
        sub=TARGETS[TARGETS.endpoint.eq(endpoint)&TARGETS.kind.eq(kind)].copy()
        if kind=='curve':sub=sub[sub.Tobs_K.le(275)].copy()
        if sub.empty:continue
        sub['block']=endpoint+'_'+kind
        sub['weight']=factor/(sub.groupby('sample_id').obs_id.transform('count')*
            sub.groupby('family').sample_id.transform('nunique')*sub.family.nunique())
        pieces.append(sub)
    import pandas as pd
    return pd.concat(pieces,ignore_index=True).sort_values(['family','sample_id','Tobs_K','endpoint'])


def freeze():
    gate=json.loads((OUT/'numerical_gates.json').read_text())
    assert gate['passed'] and gate['code_sha256']==energy.sha(ROOT/'src/sandbox/property_predictor_v5.py')
    files=[Path(__file__),ROOT/'src/sandbox/property_predictor_v5.py']
    refs=[BASE/'artifact_manifest_phase18.json',BASE/'predictor_bundle_v4/manifest.json',BASE/'dataset_v1.csv',
        BASE/'episodes_v1.csv',NPZ,BASE/'phase16_thermal_joint/thermal_audit/specimens.csv',
        inputs.core.prior.prior.TEMPDIR/'temperature_observations.csv',OUT/'numerical_gates.json']
    for method in SPECS:
        if method=='stock_const_noext_RT':continue
        phase='phase16_thermal_joint' if method.startswith('Thermal_') else 'phase15_coherent_hall'
        refs.append(BASE/phase/'methods'/method/'primary_predictions.csv')
    refs += [BASE/'phase12_stock_adaptation/research_runtime/index.json',BASE/'phase12_stock_adaptation/research_runtime/tasks.json']
    payload=dict(created_utc=energy.now(),version=1,direction='trained ML + conservative Sandbox -> doped-material property prediction',
        models={m:record(m) for m in SPECS},inherited_energy_methods=ENERGY_METHODS,
        specifications={m:dict(bounds=spec(m)[:2],starts=spec(m)[2]) for m in SPECS},
        new_full_source_fits=6,initial_optimizer_starts=12,
        optimizer='two original starts;max_nfev120,tol1e-7;any failed start triggers uniform360 refinement at both starts plus saved best',
        training='all29/5 sources;RTnH plus .1 old Si curves for joint;gamma1 only designated jointmu model;source/specimen/read equal',
        calibration='all5 training-source held-out predictions select lambda;these are training choices for full-source delivery,not independent test scores',
        source_coordinates='output offset for all,additional internal parent-stock coordinate for constant-stock',
        energy_budget='internal models full873;legacy energy completion keeps exact earlier budget incl2 explicitly supplied labels when specified',
        properties='nH plus same-state Hall transport where implemented;otherwise300K empiricalCT head;derivedsigma',
        process='pO2=1e-5 assumed;no complete thermal history;growth metadata inactive in selected presets',
        external_source='Leedy2018 DOI10.1063/1.5047214 identified and abstract/methods viewed;quantitative pairing not extracted;retrospective/notblind',
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in files},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in refs},
        new_DFT_MD=0,new_material_measurements=0,GPU_used=False,goal_complete=False)
    p=OUT/'delivery_protocol.json'
    if p.exists():
        old=json.loads(p.read_text())
        for key in ('models','specifications','code_hashes','input_hashes'):assert old[key]==energy.clean(payload[key]),key
    else:write(p,payload)


def forward(model,w,tr):
    cache={};v=[];J=[]
    for row in tr.itertuples():
        key=(row.sample_id,float(row.Tobs_K))
        if key not in cache:cache[key]=model.point(row,w)
        p=cache[key]
        v.append(p.values[0] if row.endpoint=='nH' else p.transport_values[0])
        J.append(p.jacobian[0] if row.endpoint=='nH' else p.transport_jacobian[0])
    return np.array(v),np.array(J)


def fit(method):
    path=OUT/'fits'/f'{method}.json'
    if path.exists():return json.loads(path.read_text())
    tr=training(method);y=tr.target.to_numpy(float);sw=np.sqrt(tr.weight.to_numpy(float))
    db=load();model=make_model(record(method),db);lo,hi,starts=spec(method);trials=[];failures=[];clock=time.perf_counter();cache={}
    def calc(w):
        key=tuple(w)
        if key not in cache:
            try:val=forward(model,w,tr)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                failures.append(dict(parameters=w,error=repr(exc)));raise
            cache.clear();cache[key]=val
        return cache[key]
    def run(initial,n,stage):
        try:
            op=least_squares(lambda w:(calc(w)[0]-y)*sw,initial,jac=lambda w:calc(w)[1]*sw[:,None],
                bounds=(lo,hi),max_nfev=n,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            return dict(stage=stage,initial=initial,parameters=op.x,loss=float(op.fun@op.fun),success=bool(op.success),
                nfev=int(op.nfev),optimality=float(op.optimality),message=op.message)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            return dict(stage=stage,initial=initial,loss=None,success=False,error=repr(exc))
    trials=[run(w,120,'initial') for w in starts]
    if any(not t['success'] for t in trials):
        finite=[t for t in trials if t['loss'] is not None]
        refinement=starts+[min(finite,key=lambda t:t['loss'])['parameters']] if finite else starts
        trials += [run(w,360,'uniform_refinement') for w in refinement]
    finite=[t for t in trials if t['loss'] is not None]
    if not finite:
        write(path,dict(method=method,trials=trials,state_failures=failures,failed=True));raise RuntimeError('All starts failed')
    best=min(finite,key=lambda t:t['loss']);w=best['parameters'];values,_=forward(model,w,tr)
    out={**record(method),'parameters':w,'training_sources':FAMILIES,'training_obs_ids':tr.obs_id.tolist(),
        'training_blocks':[{**dict(block=b,observations=len(g),specimens=g.sample_id.nunique(),families=g.family.nunique()),
            'weight_sum':float(g.weight.sum())} for b,g in tr.groupby('block')],
        'trials':trials,'selected_success':best['success'],'state_failures':failures,
        'training_loss':float(np.sum(tr.weight.to_numpy(float)*(values-y)**2)),
        'protocol_sha256':energy.sha(OUT/'delivery_protocol.json'),'seconds':time.perf_counter()-clock}
    write(path,out)
    write(OUT/'training_targets'/f'{method}.json',[dict(obs_id=r.obs_id,target=float(r.target),weight=float(r.weight),prediction=float(v)) for r,v in zip(tr.itertuples(),values)])
    print('full5',method,'seconds',round(out['seconds'],2),'starts',len(trials),'stop',best['success'],flush=True)
    return energy.clean(out)


def choices(raw,method,kind='offset',stock_folds=None):
    rows=[];selected={}
    rt=inputs.PRED[inputs.PRED.kind.eq('RT')]
    for b in (0,1,2):
        candidates=[]
        for lam in ([np.inf] if b==0 else LAM):
            scores=[]
            for held in FAMILIES:
                d=rt[rt.family.eq(held)];roles=ROLES[ROLES.family.eq(held)&ROLES.budget.eq(b)].set_index('sample_id').role
                a=d.sample_id.map(roles).eq('anchor').to_numpy();q=~a;y=d.target.to_numpy(float)
                p=np.array([raw[r.sample_id] for r in d.itertuples()]);z=0.
                if b and np.isfinite(lam):
                    if kind=='stock':
                        model,pars=stock_folds[held];ar=list(d[a].itertuples())
                        def forward_z(v):
                            out=[model.point(r,pars,source_kind='stock',source_shift=v) for r in ar]
                            return np.array([v.values[0] for v in out]),np.array([v.source_jacobian[0] for v in out])
                        fitz=ScalarProfile(forward_z).fit(y[a],lam);z=float(fitz.shift)
                        p=np.array([model.point(r,pars,source_kind='stock',source_shift=z).values[0] for r in d.itertuples()])
                    else:z=float(np.sum(p[a]-y[a])/(b+lam));p-=z
                score=float(np.abs(p[q]-y[q]).mean());scores.append(score)
                rows.append(dict(method=method,kind=kind,budget=b,lam=None if np.isinf(lam) else lam,held=held,
                    query_count=int(q.sum()),MAE=score,source_coordinate=z))
            candidates.append((float(np.mean(scores)),lam))
        best=min(candidates,key=lambda t:(round(t[0],10),-t[1]));selected[str(b)]=None if np.isinf(best[1]) else float(best[1])
    write(OUT/'calibration_selection'/f'{method}_{kind}.json',dict(selected=selected,candidate_folds=rows,
        scope='full5 training selection;not a new independent performance estimate'))
    return selected


def export():
    db=load();BUNDLE.mkdir(parents=True,exist_ok=True);methods={};files=[];cvrefs=[]
    energy_manifest=json.loads((BASE/'predictor_bundle_v4/manifest.json').read_text())
    for m in ENERGY_METHODS:methods[m]=dict(branch='energy',learning='inherited conditional energetic completion')
    stockindex=json.loads((BASE/'phase12_stock_adaptation/research_runtime/index.json').read_text())
    stocktasks=json.loads((BASE/'phase12_stock_adaptation/research_runtime/tasks.json').read_text())
    stockfolds={};stockraw={}
    for e in stockindex:
        if e['model']!='constant_stock' or e['budget']!=0:continue
        path=ROOT/e['bundle'];f=json.loads(path.read_text());cvrefs.append(dict(path=e['bundle'],sha256=energy.sha(path)))
        model=make_model(record('stock_const_noext_RT'),db);stockfolds[e['held']]=(model,f['parameters'])
        task=next(t for t in stocktasks if t['task_id']==e['task_id'])
        for p in task['points']:
            row=SimpleNamespace(**p['input']);stockraw[p['obs_id'].removeprefix('RT::')]=float(model.point(row,f['parameters']).values[0])
    for m in SPECS:
        f=fit(m);path=BUNDLE/'models'/f'{m}.json';write(path,f)
        files.append(dict(path=str(path.relative_to(BUNDLE)),sha256=energy.sha(path)))
        if m=='stock_const_noext_RT':raw=stockraw
        else:
            phase='phase16_thermal_joint' if m.startswith('Thermal_') else 'phase15_coherent_hall'
            path_cv=BASE/phase/'methods'/m/'primary_predictions.csv';cv=read(path_cv)
            cv=cv[cv.budget.eq(0)];assert len(cv)==29 and cv.sample_id.is_unique
            raw=dict(zip(cv.sample_id,cv.predicted_log_nH))
        active=['element','actual_incorporated_dose','observation_T','declared_H_supply','anchor_nH']
        methods[m]=dict(branch='internal',model_key=m,source_kind='offset',anchor_penalty=choices(raw,m),
            energy_budget=dict(supplied_original_charge_energies=873,extra_computations_this_stage=0),
            active_inputs=active,learning='shared internal relation trained on old property labels;see model definition',
            transport='same-state coherent FD Hall/Born/RTA' if SPECS[m]['kind'] in ('coherent','thermal') else '300K empirical CT plus saved scalar calibration')
        if m=='stock_const_noext_RT':
            methods['stock_const_internal']={**methods[m],'source_kind':'stock',
                'anchor_penalty':choices(raw,m,'stock',stockfolds)}
    rt=inputs.PRED[inputs.PRED.kind.eq('RT')];weights=1/rt.family.map(rt.family.value_counts()).to_numpy(float)
    from transport_chain import median
    activation=median(rt.target.to_numpy(float)-np.log10(rt.incorporated_cm3.to_numpy(float)),weights)
    for method,delta in (('unit_dose',0.),('median_activation',float(activation))):
        raw={}
        for held in FAMILIES:
            tr=rt[rt.family.ne(held)];w=1/tr.family.map(tr.family.value_counts()).to_numpy(float)
            shift=0. if method=='unit_dose' else median(tr.target.to_numpy(float)-np.log10(tr.incorporated_cm3.to_numpy(float)),w)
            for r in rt[rt.family.eq(held)].itertuples():raw[r.sample_id]=float(np.log10(r.incorporated_cm3)+shift)
        methods[method]=dict(branch='empirical',activation_offset_dex=delta,anchor_penalty=choices(raw,method),
            source_kind='offset',energy_budget=dict(supplied_original_charge_energies=0),
            active_inputs=['actual_incorporated_dose','anchor_nH'],learning='unit dose or source-weighted median activation control',
            transport='300K CT plus saved scalar calibration')
    domain=dict(chemical_cm3={e:[float(g.incorporated_cm3.min()),float(g.incorporated_cm3.max())] for e,g in DATA.groupby('element')},
        nH_training_sources=FAMILIES,nH_RT_specimens=29,old_Si_temperature_specimens=7,
        complete_process_input_specimens=0,actual_full_thermal_histories=0,
        paired_RT_transport_specimens=5,paired_RT_transport_sources=2,muT_reads=54,muT_specimens=3,
        rare_earth_validated=False,growth_T_is_active_feature=False,calibrated_uncertainty=False)
    manifest=dict(version=5,created_utc=energy.now(),structure_fingerprint=structure_fingerprint(db),
        full_energy_fingerprint=supplied_energy_fingerprint(db),files=files,methods=methods,domain=domain,
        energy_bundle_relative=os.path.relpath(BASE/'predictor_bundle_v4',BUNDLE),
        energy_manifest_sha256=energy.sha(BASE/'predictor_bundle_v4/manifest.json'),
        transport_offset_dex=energy_manifest['transport']['offset_dex'],
        calibration_cv_bundle_references=cvrefs,delivery_protocol_sha256=energy.sha(OUT/'delivery_protocol.json'),
        full_source_training_is_not_new_validation=True,goal_complete=False)
    write(BUNDLE/'manifest.json',manifest);print('export',len(methods),'presets',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--initialize',action='store_true');ap.add_argument('--train',action='store_true')
    ap.add_argument('--export',action='store_true');ap.add_argument('--workers',type=int,default=2);a=ap.parse_args();freeze()
    if a.train:
        write(OUT/'training_started.json',dict(created_utc=energy.now(),fits=6,workers=a.workers))
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for _ in pool.map(fit,SPECS):pass
        write(OUT/'training_complete.json',dict(created_utc=energy.now(),fits=6,starts=sum(len(fit(m)['trials']) for m in SPECS)))
    if a.export:export()


if __name__=='__main__':main()
