"""Known-rate temperature-level-pair validation, with a tied three-parameter rate baseline."""
from pathlib import Path
from copy import deepcopy
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.optimize import lsq_linear
from sandbox.chemical_process import source_coordinate,ChemicalCalibration
from csv_io_v2 import read_result_csv as read
import phase24_chemical_learning as parent
import phase2_energy as e
BASE=e.BASE;OUT=BASE/'phase24_chemical_process';DEST=OUT/'paired_validation';BUNDLE=OUT/'delivery/paired_chemical_bundle'
FAMILIES=(*parent.METHODS,'RateAffine');RATE_FITS={};SELECTIONS=[]
NPARAM={'ArrheniusMassBalance':2,'RateAffine':3,'MonotoneFlux':6,'RegimeFlux':7,'DirectProcessSpline':7}


def freeze():
    p=dict(created_utc=e.now(),code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in (Path(__file__),ROOT/'docs/ML_SANDBOX_PAIRED_CHEMICAL_PROTOCOL_20261018.md',Path(parent.__file__),ROOT/'src/sandbox/chemical_process.py')},
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in (OUT/'extraction/layer_readings.csv',OUT/'chemical_learning/audit.json')},
        task='known two rates,new nominal layer temperature levels;same source',outer='6pairedlevel groups;2heldchemicalconditions pergroup',
        inner='trainingpairedlevels only;noheldpairlabels',families=FAMILIES,lambda_grid=parent.LAMBDAS,
        no_Hall_labels_for_family_or_hyperselection=True,old_models_or_property_predictions_overwritten=False,goal_complete=False)
    path=DEST/'protocol.json'
    if path.exists():assert json.loads(path.read_text())['code_hashes']==p['code_hashes']
    else:e.write(path,p)


def fit(d,family,lam):
    if family!='RateAffine':
        m=deepcopy(parent.fit(d,family,lam));m['family']=family;m['independent_trained_parameters']=NPARAM[family];return m
    ids=list(d.layer_id);scenario=d.scenario.iloc[0];key=hashlib.sha256(json.dumps([scenario,family,float(lam),ids]).encode()).hexdigest()[:20]
    if key in RATE_FITS:return RATE_FITS[key]['model']
    r=np.log10(d.velocity_um_h.to_numpy());center=float(r.mean());q=source_coordinate(d.Tsource_nominal_C.to_numpy());X=np.c_[np.ones(len(d)),q,r-center];y=d.log10_C_chemical.to_numpy()+r
    penalty=np.array([[0.,0.,1.]]);A=np.r_[X/np.sqrt(len(d)),np.sqrt(lam)*penalty];b=np.r_[y/np.sqrt(len(d)),0.]
    solved=lsq_linear(A,b,bounds=([-np.inf,0.,-np.inf],[np.inf,15/np.log(10.),np.inf]),method='bvls',tol=1e-12,max_iter=5000)
    if not solved.success:raise RuntimeError('RateAffine fit did not stop')
    mapping=np.zeros((7,3));mapping[0,0]=1.;mapping[1:6,1]=1.;mapping[6,2]=1.;coef=mapping@solved.x
    m=dict(version=1,method='RegimeFlux',family=family,coefficients=coef.tolist(),coefficient_tie_matrix=mapping.tolist(),independent_coefficients=solved.x.tolist(),
        independent_trained_parameters=3,rate_center=center,lambda_=float(lam),source_T_support_C=[float(d.Tsource_nominal_C.min()),float(d.Tsource_nominal_C.max())],
        growth_rate_support_um_h=[float(d.velocity_um_h.min()),float(d.velocity_um_h.max())],train_layer_ids=ids,train_stacks=sorted(set(d.stack_id)),chemical_scenario=scenario,
        source='same-source pairedlevel chemistry supervision;not microkinetics',Hall_labels_used=False)
    res=X@solved.x-y;grad=2*(X.T@res/len(d)+lam*penalty.T@(penalty@solved.x))
    RATE_FITS[key]=dict(key=key,model=m,optimization=dict(success=bool(solved.success),status=int(solved.status),message=solved.message,
        cost=float(res@res/len(d)+lam*np.sum((penalty@solved.x)**2)),gradient=grad.tolist(),optimality=float(solved.optimality),rank=int(np.linalg.matrix_rank(A)),columns=3))
    return m


def predict(m,d):
    c=ChemicalCalibration(m);return np.array([c.predict(r.Tsource_nominal_C,r.velocity_um_h)['log10_C_cm3'] for r in d.itertuples()])


def choose(d,family,scope):
    candidates=[]
    for lam in ([0.] if family=='ArrheniusMassBalance' else parent.LAMBDAS):
        folds=[]
        for level in sorted(d.surface_order.unique()):
            train=d[d.surface_order.ne(level)];test=d[d.surface_order.eq(level)];m=fit(train,family,lam)
            folds.append(dict(held_level=int(level),train_layer_ids=list(train.layer_id),test_layer_ids=list(test.layer_id),MAE=float(np.mean(abs(predict(m,test)-test.log10_C_chemical.to_numpy())))))
        candidates.append(dict(lambda_=float(lam),MAE=float(np.mean([r['MAE'] for r in folds])),folds=folds))
    best=min(candidates,key=lambda c:(round(c['MAE'],12),-c['lambda_']));m=fit(d,family,best['lambda_'])
    rec=dict(family=family,scenario=d.scenario.iloc[0],scope=scope,train_layer_ids=list(d.layer_id),selected_lambda=best['lambda_'],selection_MAE=best['MAE'],candidates=candidates)
    SELECTIONS.append(rec);return m,rec


def family_choice(choices):return min(choices,key=lambda z:(round(z[1]['selection_MAE'],12),NPARAM[z[1]['family']],-z[1]['selected_lambda'],z[1]['family']))


def main():
    freeze();d=read(OUT/'extraction/layer_readings.csv');rows=[];summary=[];fullmodels={};deployed={};policies=[]
    for scenario,part in d.groupby('scenario'):
        for held in sorted(part.surface_order.unique()):
            train=part[part.surface_order.ne(held)];test=part[part.surface_order.eq(held)];choices=[]
            for family in FAMILIES:
                m,rec=choose(train,family,'outerpair:'+str(int(held)));choices.append((m,rec))
                for r,value in zip(test.itertuples(),predict(m,test)):
                    rows.append(dict(scenario=scenario,family=family,held_level=int(held),layer_id=r.layer_id,observed_logC=r.log10_C_chemical,predicted_logC=float(value),absolute_error=float(abs(value-r.log10_C_chemical)),selected_lambda=rec['selected_lambda']))
            chosen,rec=family_choice(choices);policies.append(dict(scenario=scenario,scope='outerpair:'+str(int(held)),selected_family=rec['family'],selected_lambda=rec['selected_lambda'],train_layer_ids=list(train.layer_id),test_layer_ids=list(test.layer_id)))
            for r,value in zip(test.itertuples(),predict(chosen,test)):
                rows.append(dict(scenario=scenario,family='SelectedPolicy',held_level=int(held),layer_id=r.layer_id,observed_logC=r.log10_C_chemical,predicted_logC=float(value),absolute_error=float(abs(value-r.log10_C_chemical)),selected_lambda=rec['selected_lambda'],selected_family=rec['family']))
        choices=[]
        for family in FAMILIES:
            m,rec=choose(part,family,'full12');choices.append((m,rec));fullmodels[scenario+'::'+family]=m
            selected=[r for r in rows if r['scenario']==scenario and r['family']==family]
            summary.append(dict(scenario=scenario,family=family,outer_predictions=12,nested_pair_MAE=float(np.mean([r['absolute_error'] for r in selected])),full_lambda=rec['selected_lambda'],full_selection_CV_MAE=rec['selection_MAE'],full_training_MAE=float(np.mean(abs(predict(m,part)-part.log10_C_chemical.to_numpy())))))
        chosen,rec=family_choice(choices);deployed[scenario+'::PairSelected']=chosen
        policies.append(dict(scenario=scenario,scope='full12',selected_family=rec['family'],selected_lambda=rec['selected_lambda'],training_CV_MAE=rec['selection_MAE'],train_layer_ids=list(part.layer_id)))
        selected=[r for r in rows if r['scenario']==scenario and r['family']=='SelectedPolicy'];summary.append(dict(scenario=scenario,family='SelectedPolicy',outer_predictions=12,nested_pair_MAE=float(np.mean([r['absolute_error'] for r in selected])),full_selected_family=rec['family'],full_lambda=rec['selected_lambda'],full_selection_CV_MAE=rec['selection_MAE']))
    for key,row in {**parent.FITS,**RATE_FITS}.items():e.write(DEST/'fits'/f'{key}.json',row)
    for name,m in fullmodels.items():e.write(DEST/'full_models'/(name.replace('::','__')+'.json'),m)
    e.write(DEST/'all_selections.json',SELECTIONS);e.write(DEST/'all_policy_choices.json',policies);e.csvwrite(DEST/'predictions.csv',rows);e.csvwrite(DEST/'summary.csv',summary)
    manifest={}
    for name,m in deployed.items():
        path='models/'+name.replace('::','__')+'.json';e.write(BUNDLE/path,m);manifest[name]=path
    e.write(BUNDLE/'manifest.json',dict(version=1,created_utc=e.now(),models=manifest,files=[dict(path=p,sha256=e.sha(BUNDLE/p)) for p in manifest.values()],
        training_labels='same12chemicalconditions;known-rate pairedlevelselection;no Halltargets',scope='same-source known0.86/1.3rates,new sourceTconditions',goal_complete=False))
    e.write(DEST/'execution_complete.json',dict(created_utc=e.now(),fit_records=len(parent.FITS)+len(RATE_FITS),selection_contexts=len(SELECTIONS),outer_predictions=len(rows),saved_full_candidates=len(fullmodels),deployed_selected_models=len(deployed),old_results_unchanged=True,goal_complete=False))
    print('paired chemical learning',len(parent.FITS)+len(RATE_FITS),'fits',len(rows),'OOFrows',flush=True)
    for r in summary:
        if r['scenario']=='central50':print(r,flush=True)


if __name__=='__main__':main()
