"""Train conditional internal inventory fields with matched source/cluster controls."""
from pathlib import Path
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
import os, sys, json, time, hashlib, argparse
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import logit
from sklearn.linear_model import Ridge
from sandbox.context_inventory_model import ContextInventoryModel, bounded_field, field_features
from sandbox.coherent_fermi import published_coherent_profile
import phase25_data as data
import phase2_energy as e
from transport_chain import median

OUT = data.OUT
BASE = data.BASE
TABLE = data.table()
ROLES = data.roles()
REG = (0., .1, 1.)
RIDGE = (.01, .1, 1., 10., 100., 1e4)
SPECS = {'FixedPhysical':dict(kind='inventory', basis='none')}
for kind in ('Inventory', 'Output'):
    for basis in ('constant', 'dose', 'process'):
        SPECS[kind + basis.title()] = dict(kind=kind.lower(), basis=basis)
for basis in ('dose', 'process'):
    SPECS['Direct' + basis.title()] = dict(kind='direct', basis=basis)
EMPIRICAL = ('DirectRidge', 'UnitDose', 'MedianActivation', 'DoseLine')
METHODS = tuple(SPECS) + EMPIRICAL
MODELS = {}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


def unit_column(scheme):
    if scheme not in ('DOI', 'cluster'):
        raise ValueError('Explicit DOI/cluster validation required')
    return 'family' if scheme == 'DOI' else 'related_cluster'


def units(scheme):
    return tuple(sorted(TABLE[unit_column(scheme)].unique()))


def subset(scheme, training_units):
    return TABLE[TABLE[unit_column(scheme)].isin(training_units)].copy()


def weights(table, scheme):
    col = unit_column(scheme)
    nunit = table[col].nunique()
    nfamily = table.groupby(col).family.transform('nunique').to_numpy(float)
    ncondition = table.groupby('family').sample_id.transform('count').to_numpy(float)
    result = 1. / (nunit * nfamily * ncondition)
    assert abs(result.sum() - 1.) < 1e-12
    return result


def contexts(scheme):
    u = units(scheme)
    return [tuple(c) for n in (len(u)-2, len(u)-1, len(u)) for c in combinations(u,n)]


def configs(method):
    if method in SPECS and method != 'FixedPhysical':
        return list(REG) + [None]
    return list(RIDGE) if method == 'DirectRidge' else [None]


def mapper(method):
    spec = SPECS[method]
    if spec['kind'] == 'direct':
        raise ValueError('Direct controls do not have a physical state mapper')
    if method not in MODELS:
        MODELS[method] = ContextInventoryModel(basis=spec['basis'], field=spec['kind'], profile=published_coherent_profile())
    return MODELS[method]


def direct_features(table):
    c = np.log10(table.incorporated_cm3.to_numpy(float)) - 18.
    hknown = table.H_known.to_numpy(float)
    cknown = table.carbon_known.to_numpy(float)
    h = np.log10(np.maximum(table.H_cm3.to_numpy(float), 1.)) - 17.
    carbon = np.log10(np.maximum(table.carbon_cm3.to_numpy(float), 1.)) - 17.
    p = np.log10(table.scenario_pO2_atm.to_numpy(float)) + 5.
    reported = table.family.eq(data.LEEDY).to_numpy(float)
    return np.c_[c, c*c, (table.growth_T_C.to_numpy(float)-700.)/300., table.element.eq('Sn').to_numpy(float),
        hknown, h*hknown, cknown, carbon*cknown, p, reported]


def forward(method, parameters, table, *, penalty=False, source_kind='none', source_shift=0.):
    w = np.asarray(parameters, float)
    spec = SPECS[method]
    values, jac, changes, change_jac, valid = [], [], [], [], []
    if spec['kind'] == 'direct':
        for row in table.itertuples():
            h, dh, _ = bounded_field(row, w[2:], spec['basis'])
            x = float(np.log10(row.incorporated_cm3)-18.)
            values.append(w[0] + w[1]*x + h + source_shift)
            jac.append(np.r_[1., x, dh])
            valid.append(values[-1] >= 13.)
            if penalty:
                changes.append(h)
                change_jac.append(np.r_[0., 0., dh])
    else:
        m = mapper(method)
        ref = mapper('FixedPhysical')
        for row in table.itertuples():
            p = m.point(row,w,source_kind=source_kind,source_shift=source_shift,require_observation=False)
            values.append(p.values[0]); jac.append(p.jacobian[0]); valid.append(p.diagnostics['observation_domain_valid'])
            if penalty:
                q = ref.point(row,w[:2],require_observation=False)
                dq = p.jacobian[0].copy(); dq[:2] -= q.jacobian[0]
                changes.append(p.values[0]-q.values[0]); change_jac.append(dq)
    return np.asarray(values), np.asarray(jac), np.asarray(changes), np.asarray(change_jac), np.asarray(valid,bool)


def fit_path(scheme, training_units, method, reg):
    key = digest(dict(protocol=e.sha(OUT/'training_protocol.json'),scheme=scheme,units=sorted(training_units),method=method,reg=reg))
    return key, OUT/'fits'/(key+'.json')


def fit(scheme, training_units, method, reg):
    training_units = tuple(sorted(training_units))
    key, path = fit_path(scheme,training_units,method,reg)
    if path.exists():
        return json.loads(path.read_text())
    tr = subset(scheme,training_units)
    y = tr.target.to_numpy(float); wg = weights(tr,scheme); sw = np.sqrt(wg)
    clock = time.perf_counter()
    record = dict(key=key,scheme=scheme,train_units=training_units,train_families=sorted(tr.family.unique()),
        training_obs_ids=tr.obs_id.tolist(),method=method,reg=reg,trials=[],state_failures=[],training_valid=False,
        parameters=None,new_optimizer_starts=0)
    if method in SPECS:
        spec = SPECS[method]
        if method != 'FixedPhysical' and reg is None:
            fallback_method = 'DoseLine' if spec['kind']=='direct' else 'FixedPhysical'
            base = fit(scheme,training_units,fallback_method,None)
            nfield = dict(constant=1,dose=2,process=4)[spec['basis']]
            pars = (list(base['parameters']) if spec['kind']!='direct' else list(base['coefficients'])) + [0.]*nfield
            record.update(parameters=pars,training_valid=base['training_valid'],selected_success=base.get('selected_success',True),
                exact_reference_fallback=True,reference_key=base['key'],selected_loss=None)
        else:
            nfield = 0 if method=='FixedPhysical' else dict(constant=1,dose=2,process=4)[spec['basis']]
            tracked = {name:mapper(name).counters() for name in sorted({method,'FixedPhysical'})} if spec['kind']!='direct' else {}
            if spec['kind']=='direct':
                lo=np.r_[13.,-2.,np.repeat(-12.,nfield)]; hi=np.r_[24.,4.,np.repeat(12.,nfield)]
                starts=[np.r_[18.,1.,np.zeros(nfield)],np.r_[17.,.7,np.repeat(.1,nfield)]]
            else:
                lo=np.r_[-6.,14.,np.repeat(-12.,nfield)]; hi=np.r_[6.,18.,np.repeat(12.,nfield)]
                second=logit((np.log10(1750.)-np.log10(600.))/(np.log10(1950.)-np.log10(600.)))
                if method=='FixedPhysical':
                    starts=[np.array([.5,16.]),np.array([second,15.5])]
                else:
                    base=fit(scheme,training_units,'FixedPhysical',None)
                    starts=[np.r_[base['parameters'],np.zeros(nfield)],np.r_[second,15.5,np.repeat(.1,nfield)]]
            failures=[]; continuations=0; cache={}; calls=0
            def calculate(w):
                nonlocal continuations,calls
                k=tuple(w)
                if k not in cache:
                    try:
                        v=forward(method,w,tr,penalty=bool(reg and reg>0))
                    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                        failures.append(dict(parameters=w.tolist(),error=repr(exc)))
                        raise
                    calls+=1; continuations+=int((~v[4]).sum()); cache.clear();cache[k]=v
                return cache[k]
            def residual(w):
                v,j,delta,dj,valid=calculate(w)
                return np.r_[(v-y)*sw,np.sqrt(reg)*delta*sw] if reg and reg>0 else (v-y)*sw
            def jacobian(w):
                v,j,delta,dj,valid=calculate(w)
                return np.r_[j*sw[:,None],np.sqrt(reg)*dj*sw[:,None]] if reg and reg>0 else j*sw[:,None]
            def run(initial,cap,stage):
                try:
                    op=least_squares(residual,initial,jac=jacobian,bounds=(lo,hi),x_scale='jac',max_nfev=cap,
                        ftol=1e-7,xtol=1e-7,gtol=1e-7)
                    return dict(initial=np.asarray(initial).tolist(),parameters=op.x.tolist(),loss=float(op.fun@op.fun),
                        success=bool(op.success),nfev=int(op.nfev),optimality=float(op.optimality),message=op.message,
                        stage=stage,cap=cap,active_bounds=op.active_mask.tolist())
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                    return dict(initial=np.asarray(initial).tolist(),success=False,loss=None,error=repr(exc),stage=stage,cap=cap)
            trials=[run(s,120,'initial') for s in starts]
            if any(not t['success'] for t in trials):
                finite=[t for t in trials if t['loss'] is not None]
                more=[min(finite,key=lambda t:t['loss'])['parameters']] if finite else []
                trials.extend(run(s,360,'uniform_refinement') for s in starts+more)
            finite=[t for t in trials if t['loss'] is not None]
            best=min(finite,key=lambda t:t['loss']) if finite else None
            record.update(trials=trials,state_failures=failures,new_optimizer_starts=len(trials),
                exact_reference_fallback=False,selected_success=bool(best and best['success']),
                numerical_continuation_trial_states=continuations,objective_forward_calls=calls,
                selected_loss=best['loss'] if best else None)
            if best:
                w=np.asarray(best['parameters']);v,J,delta,dj,valid=forward(method,w,tr,penalty=True)
                svals=np.linalg.svd(J*sw[:,None],compute_uv=False)
                record.update(parameters=w.tolist(),training_valid=bool(valid.all()),training_nH_MAE=float(np.abs(v-y)@wg),
                    training_nH_MSE=float(((v-y)**2)@wg),rms_response_change=float(np.sqrt(delta**2@wg)),
                    weighted_prediction_jacobian_singular_values=svals.tolist(),
                    prediction_jacobian_numeric_rank=int(np.linalg.matrix_rank(J*sw[:,None])),
                    selected_training_values=v.tolist(),selected_training_valid=valid.tolist(),
                    solver_counters={name:{k:mapper(name).counters()[k]-v for k,v in before.items()}
                        for name,before in tracked.items()})
    elif method=='DirectRidge':
        X=direct_features(tr);center=np.average(X,axis=0,weights=wg)
        scale=np.sqrt(np.average((X-center)**2,axis=0,weights=wg));scale[scale<1e-12]=1.
        m=Ridge(alpha=reg).fit((X-center)/scale,y,sample_weight=wg*len(tr))
        record.update(training_valid=True,coefficients=m.coef_.tolist(),intercept=float(m.intercept_),center=center.tolist(),scale=scale.tolist(),selected_success=True)
    else:
        x=np.log10(tr.incorporated_cm3.to_numpy(float));offset=0.;coeff=None
        if method=='MedianActivation':offset=float(median(y-x,wg))
        if method=='DoseLine':
            X=np.c_[np.ones(len(tr)),x-18.];coeff=np.linalg.lstsq(X*sw[:,None],y*sw,rcond=None)[0].tolist()
        record.update(training_valid=True,coefficients=coeff,offset_dex=offset,selected_success=True)
    record['seconds']=time.perf_counter()-clock
    e.write(path,record)
    print('fit25',scheme,method,reg,len(training_units),'sec',round(record['seconds'],2),
        'valid',record['training_valid'],'stopped',record.get('selected_success'),flush=True)
    return e.clean(record)


def predict(f, table, *, source_kind='none', shift=0., require_observation=True):
    method=f['method']
    if not f['training_valid'] or not f.get('selected_success',True):
        raise ValueError('Fit is invalid or its selected optimizer start did not stop')
    if method in SPECS:
        if SPECS[method]['kind']!='direct':
            points=[mapper(method).point(r,f['parameters'],source_kind=source_kind,source_shift=shift,
                require_observation=require_observation) for r in table.itertuples()]
            return np.array([p.values[0] for p in points]),np.array([p.source_jacobian[0] for p in points]),np.array([p.diagnostics['observation_domain_valid'] for p in points])
        values=forward(method,f['parameters'],table,source_shift=shift)
        p=values[0];valid=values[4]
    elif method=='DirectRidge':
        X=(direct_features(table)-np.asarray(f['center']))/np.asarray(f['scale'])
        p=f['intercept']+X@np.asarray(f['coefficients'])+shift;valid=p>=13.
    else:
        x=np.log10(table.incorporated_cm3.to_numpy(float))
        p=f['coefficients'][0]+f['coefficients'][1]*(x-18.) if method=='DoseLine' else x+f['offset_dex']
        p=p+shift;valid=p>=13.
    if require_observation and not valid.all():
        raise ValueError('Empirical prediction below declared Hall detection floor')
    return p,np.ones(len(p)),valid


def unit_score(f, held_unit):
    d=TABLE[TABLE[unit_column(f['scheme'])].eq(held_unit)]
    try:
        y,j,v=predict(f,d)
    except (ValueError,RuntimeError,np.linalg.LinAlgError):
        return np.inf
    if not v.all():return np.inf
    errors=pd.DataFrame(dict(family=d.family.to_numpy(),error=np.abs(y-d.target.to_numpy(float))))
    return float(errors.groupby('family').error.mean().mean())


def select(training_units,scheme,method):
    trials=[]
    for reg in configs(method):
        scores=[unit_score(fit(scheme,[u for u in training_units if u!=held],method,reg),held) for held in training_units]
        trials.append(dict(reg=reg,score=float(np.mean(scores)),inner_units=list(training_units),inner_scores=scores))
    def key(t):
        r=np.inf if t['reg'] is None else t['reg']
        return round(t['score'],10),-r
    return min(trials,key=key)['reg'],trials


def freeze():
    gates=json.loads((OUT/'numerical_gates.json').read_text());assert gates['passed']
    for name,h in gates['code_hashes'].items():assert e.sha(ROOT/name)==h,name
    learning=json.loads((OUT/'learning_gates.json').read_text());assert learning['passed']
    for name,h in learning['code_hashes'].items():assert e.sha(ROOT/name)==h,name
    paths=[Path(__file__),Path(__file__).parent/'phase25_data.py',ROOT/'src/sandbox/context_inventory_model.py',
        ROOT/'docs/ML_SANDBOX_CONTEXT_INVENTORY_PROTOCOL_20261018.md']
    paths += [ROOT/p for p in gates['code_hashes']]
    inputs=[OUT/'data/conditions33.csv',OUT/'data/episodes33.csv',OUT/'data/qualification.json',OUT/'numerical_gates.json',OUT/'learning_gates.json',BASE/'artifact_manifest_phase24.json']
    p=dict(created_utc=e.now(),version=1,goal='trainable ML+Sandbox dopant-property prediction',methods=SPECS,
        controls=EMPIRICAL,regularization=REG,infinite_regularization_fallback=True,ridge_grid=RIDGE,
        schemes={s:dict(units=units(s),contexts=contexts(s),weighting='equalunit/equalDOI/equalcondition') for s in ('DOI','cluster')},
        nonlinear_starts='two predefined starts120;any failure uniformly refines original2+best360;all preserved',
        bounds=dict(logitTf=[-6,6],log10background=[14,18],theta=[-12,12],field=[-6,6],direct_intercept=[13,24],direct_slope=[-2,4]),
        source_adaptation=dict(lambda_grid=[0.,.01,.1,1.,10.,None],range=[-6,6],
            penalty='mean squared response on all declared source inputs,not onlyfirstanchor',
            grid_nodes=33,local_candidates=3,local_iterations=80,
            source_labels='same predefined0/1/2Hall anchors;queryinputsonlyinresponse design'),
        property_training='nH_RT only;no newmu or T supervision;all same source chemical/Halllabels',
        full_data_newsource_included=True,retrospective=True,independent_labs_verified=False,goal_complete=False,
        no_new_measurements=True,no_new_DFT_MD=True,GPU_used=False,GPU_authorized='GPU1 only',
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in paths},input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in inputs})
    path=OUT/'training_protocol.json'
    if path.exists():
        previous=json.loads(path.read_text())
        for k in ('code_hashes','input_hashes','methods','schemes'):assert previous[k]==e.clean(p[k]),k
    else:e.write(path,p)


def worker(job):
    return fit(*job)['key']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();freeze()
    if not args.train:return
    first=[(s,c,m,None) for s in ('DOI','cluster') for c in contexts(s) for m in ('FixedPhysical','DoseLine')]
    rest=[(s,c,m,r) for s in ('DOI','cluster') for c in contexts(s) for m in METHODS if m not in ('FixedPhysical','DoseLine') for r in configs(m)]
    plan=dict(created_utc=e.now(),workers=args.workers,baseline_jobs=len(first),remaining_jobs=len(rest),jobs=first+rest,goal_complete=False)
    e.write(OUT/'training_started.json',plan)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(worker,first):pass
        e.write(OUT/'baseline_training_complete.json',dict(created_utc=e.now(),jobs=len(first)))
        for _ in pool.map(worker,rest):pass
    records=[json.loads(p.read_text()) for p in (OUT/'fits').glob('*.json')]
    e.write(OUT/'training_complete.json',dict(created_utc=e.now(),fit_records=len(records),
        actual_optimizer_starts=sum(f['new_optimizer_starts'] for f in records),
        selected_not_stopped=sum(not f.get('selected_success',True) for f in records),
        invalid_training_fits=sum(not f['training_valid'] for f in records),goal_complete=False))
    print('Training phase25 complete',len(records),flush=True)


if __name__=='__main__':
    main()
