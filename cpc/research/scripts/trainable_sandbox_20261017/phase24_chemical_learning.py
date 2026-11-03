"""Training-only constrained chemical calibration, nested whole-stack validation and saved models."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.optimize import lsq_linear
from sandbox.chemical_process import KB,REFERENCE_K,KNOT_TSOURCE_C,feature_matrix,ChemicalCalibration
from csv_io_v2 import read_result_csv as read
import phase2_energy as e
BASE=e.BASE;OUT=BASE/'phase24_chemical_process';DEST=OUT/'chemical_learning';BUNDLE=OUT/'delivery/chemical_bundle'
METHODS=('ArrheniusMassBalance','MonotoneFlux','RegimeFlux','DirectProcessSpline')
LAMBDAS=[1e-5,1e-4,1e-3,.01,.1,1.,10.,100.,1000.]
FITS={};SELECTIONS=[]


def freeze():
    files=(Path(__file__),ROOT/'src/sandbox/chemical_process.py',ROOT/'src/sandbox/process_property_predictor.py',ROOT/'docs/ML_SANDBOX_CHEMICAL_LEARNING_PROTOCOL_20261018.md')
    inputs=(OUT/'extraction/layer_readings.csv',OUT/'extraction/protocol.json',BASE/'artifact_manifest_phase23.json')
    p=dict(created_utc=e.now(),code_hashes={str(f.relative_to(ROOT)):e.sha(f) for f in files},input_hashes={str(f.relative_to(ROOT)):e.sha(f) for f in inputs},
        methods=METHODS,lambda_grid=LAMBDAS,knots_source_C=KNOT_TSOURCE_C.tolist(),slope_energy_upper_eV=15.,
        validation='outer leave one physical stack of6layers out;inner leaveonelayer withinother6;fullmodel selection leaveonestack out',
        loss='mean squared chemical logC pluslambda mean squared adjacent slopes andoptionalratecoefficient;selectionMAE',
        no_printed_flux_prior_in_training=True,published_reference='allstack/otherXRRfit;not an independentheldchemistry baseline',
        unknown_regime='trainingcenterlogv;withonlyonevcoefficientfixed0;physicalratecoefficientprior-1,directprior0',
        numerical_bound_not_microphysics=True,Hall_targets_used_for_training_or_hyperselection=False,goal_complete=False)
    path=DEST/'protocol.json'
    if path.exists():assert json.loads(path.read_text())['code_hashes']==p['code_hashes']
    else:e.write(path,p)


def fit(d,method,lam):
    ids=list(d.layer_id);scenario=d.scenario.iloc[0];key=hashlib.sha256(json.dumps([scenario,method,float(lam),ids]).encode()).hexdigest()[:20]
    if key in FITS:return FITS[key]['model']
    T=d.Tsource_nominal_C.to_numpy(float);v=d.velocity_um_h.to_numpy(float);r=np.log10(v);center=float(r.mean());X=feature_matrix(method,T,v,center)
    y=d.log10_C_chemical.to_numpy(float)+(r if method!='DirectProcessSpline' else 0.)
    ns=1 if method=='ArrheniusMassBalance' else 5;P=[]
    for j in range(ns-1):
        row=np.zeros(X.shape[1]);row[1+j]=-1.;row[2+j]=1.;P.append(row/np.sqrt(ns-1))
    hasrate=method in ('RegimeFlux','DirectProcessSpline')
    if hasrate:
        row=np.zeros(X.shape[1]);row[-1]=1.;P.append(row)
    penalty=np.array(P).reshape(-1,X.shape[1]);active=np.ones(X.shape[1],bool)
    if hasrate and np.ptp(r)<1e-12:active[-1]=False
    A=X/np.sqrt(len(d));b=y/np.sqrt(len(d))
    if len(penalty):A=np.r_[A,np.sqrt(lam)*penalty];b=np.r_[b,np.zeros(len(penalty))]
    lower=np.full(X.shape[1],-np.inf);upper=np.full(X.shape[1],np.inf)
    if method!='DirectProcessSpline':lower[1:1+ns]=0.;upper[1:1+ns]=15/np.log(10.)
    solved=lsq_linear(A[:,active],b,bounds=(lower[active],upper[active]),method='bvls',tol=1e-12,max_iter=5000)
    coef=np.zeros(X.shape[1]);coef[active]=solved.x
    if not solved.success:raise RuntimeError('Chemical fit failed '+key+' '+solved.message)
    model=dict(version=1,method=method,coefficients=coef.tolist(),rate_center=center,lambda_=float(lam),
        source_T_support_C=[float(T.min()),float(T.max())],growth_rate_support_um_h=[float(v.min()),float(v.max())],
        train_layer_ids=ids,train_stacks=sorted(set(d.stack_id)),chemical_scenario=scenario,reference_K=REFERENCE_K,knots_source_C=KNOT_TSOURCE_C.tolist(),
        source='same DOI two SIMSstacks;nominalsource settings;positivechemicalproxy,not microscopickinetics',
        Hall_labels_used=False,unknown_training_rate_coefficient_fixed_zero=bool(hasrate and not active[-1]))
    residual=X@coef-y;reg=penalty@coef;gradient=2*(X.T@residual/len(d)+lam*penalty.T@reg)
    FITS[key]=dict(key=key,model=model,optimization=dict(success=bool(solved.success),status=int(solved.status),message=solved.message,
        cost=float(residual@residual/len(d)+lam*(reg@reg)),optimality=float(solved.optimality),gradient=gradient.tolist(),
        active_columns=active.tolist(),rank=int(np.linalg.matrix_rank(A[:,active])),columns=int(active.sum()),
        at_lower=[int(j) for j in range(1,1+ns) if method!='DirectProcessSpline' and coef[j]<1e-9],
        at_upper=[int(j) for j in range(1,1+ns) if method!='DirectProcessSpline' and coef[j]>15/np.log(10.)-1e-9]))
    return model


def predict(record,d):
    m=ChemicalCalibration(record);return np.array([m.predict(r.Tsource_nominal_C,r.velocity_um_h)['log10_C_cm3'] for r in d.itertuples()])


def choose(d,method,scope):
    candidates=[];splits=[(d[d.stack_id.ne(h)],d[d.stack_id.eq(h)],'stack:'+h) for h in sorted(set(d.stack_id))] if d.stack_id.nunique()>1 else [
        (d[d.layer_id.ne(h)],d[d.layer_id.eq(h)],'layer:'+h) for h in d.layer_id]
    for lam in ([0.] if method=='ArrheniusMassBalance' else LAMBDAS):
        folds=[]
        for train,test,name in splits:
            rec=fit(train,method,lam);error=float(np.mean(np.abs(predict(rec,test)-test.log10_C_chemical.to_numpy(float))))
            folds.append(dict(held=name,train_layer_ids=list(train.layer_id),test_layer_ids=list(test.layer_id),MAE=error))
        candidates.append(dict(lambda_=float(lam),MAE=float(np.mean([r['MAE'] for r in folds])),folds=folds))
    best=min(candidates,key=lambda a:(round(a['MAE'],12),-a['lambda_']))
    selected=dict(method=method,scenario=d.scenario.iloc[0],scope=scope,train_layer_ids=list(d.layer_id),lambda_=best['lambda_'],
        training_CV_MAE=best['MAE'],candidates=candidates);SELECTIONS.append(selected)
    return fit(d,method,best['lambda_']),selected


def main():
    freeze();d=read(OUT/'extraction/layer_readings.csv');predictions=[];full={};summary=[]
    for scenario,part in d.groupby('scenario'):
        for method in METHODS:
            for held in ('m','n'):
                train=part[part.stack_id.ne(held)];test=part[part.stack_id.eq(held)];model,choice=choose(train,method,'outer:'+held)
                for r,value in zip(test.itertuples(),predict(model,test)):
                    predictions.append(dict(method=method,scenario=scenario,held_stack=held,layer_id=r.layer_id,Tsource_nominal_C=r.Tsource_nominal_C,
                        velocity_um_h=r.velocity_um_h,observed_log10_C=r.log10_C_chemical,predicted_log10_C=float(value),lambda_=choice['lambda_'],
                        absolute_error=float(abs(value-r.log10_C_chemical)),train_layer_ids='|'.join(model['train_layer_ids'])))
            model,choice=choose(part,method,'full12');full[scenario+'::'+method]=model
            p=[r for r in predictions if r['method']==method and r['scenario']==scenario]
            summary.append(dict(method=method,scenario=scenario,chemical_conditions=12,stacks=2,DOI_groups=1,nested_LOSO_MAE=float(np.mean([r['absolute_error'] for r in p])),
                stack_m_MAE=float(np.mean([r['absolute_error'] for r in p if r['held_stack']=='m'])),stack_n_MAE=float(np.mean([r['absolute_error'] for r in p if r['held_stack']=='n'])),
                full_lambda=choice['lambda_'],full_training_MAE=float(np.mean(abs(predict(model,part)-part.log10_C_chemical.to_numpy(float))))))
    for key,row in FITS.items():e.write(DEST/'fits'/f'{key}.json',row)
    e.write(DEST/'all_selections.json',SELECTIONS);e.csvwrite(DEST/'predictions.csv',predictions);e.csvwrite(DEST/'summary.csv',summary)
    records={}
    for name,record in full.items():
        relative='models/'+name.replace('::','__')+'.json';path=BUNDLE/relative;e.write(path,record);records[name]=relative
    files=[dict(path=v,sha256=e.sha(BUNDLE/v)) for v in records.values()]
    e.write(BUNDLE/'manifest.json',dict(version=1,created_utc=e.now(),models=records,files=files,training_labels='12 chemical layer interiors perfixedreading scenario;not newHall labels',
        scope='same-source2stack chemicalcalibration;sourceTnominal;not independentlab or microscopicrates',goal_complete=False))
    e.write(DEST/'execution_complete.json',dict(created_utc=e.now(),fit_records=len(FITS),selected_contexts=len(SELECTIONS),models=len(full),outer_predictions=len(predictions),
        all_optimizers_stopped=all(r['optimization']['success'] for r in FITS.values()),new_Hall_training_labels=0,shared_process_relation_trained=True,goal_complete=False))
    print('chemical learning complete',len(FITS),'fits',len(SELECTIONS),'selectioncontexts',len(predictions),'outer predictions',flush=True)
    for r in summary:print(r,flush=True)


if __name__=='__main__':main()
