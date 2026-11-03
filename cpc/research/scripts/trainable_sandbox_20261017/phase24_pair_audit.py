"""Independently verify paired chemical CV, tied coefficients, convex fits and family selection."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from csv_io_v2 import read_result_csv as read
from phase24_chemical_audit import manual_matrix,manual_predict
import phase2_energy as e
BASE=e.BASE;OUT=BASE/'phase24_chemical_process';DEST=OUT/'paired_validation'
NPARAM={'ArrheniusMassBalance':2,'RateAffine':3,'MonotoneFlux':6,'RegimeFlux':7,'DirectProcessSpline':7}


def main():
    checks=[]
    def gate(name,good,**extra):checks.append(dict(name=name,passed=bool(good),**extra))
    p=json.loads((DEST/'protocol.json').read_text())
    for key in ('code_hashes','input_hashes'):
        for path,h in p[key].items():gate('paired protocol hash',e.sha(ROOT/path)==h,path=path)
    data=read(OUT/'extraction/layer_readings.csv');fits={};maximum=0.
    for path in (DEST/'fits').glob('*.json'):
        r=json.loads(path.read_text());m=r['model'];family=m.get('family',m['method']);scope=data[data.scenario.eq(m['chemical_scenario'])].set_index('layer_id',drop=False);train=scope.loc[m['train_layer_ids']]
        X=manual_matrix(m,train);coeff=np.array(m['coefficients']);ns=1 if m['method']=='ArrheniusMassBalance' else 5;P=[]
        for j in range(ns-1):
            row=np.zeros(len(coeff));row[j+1]=-1/np.sqrt(ns-1);row[j+2]=1/np.sqrt(ns-1);P.append(row)
        if m['method'] in ('RegimeFlux','DirectProcessSpline'):
            row=np.zeros(len(coeff));row[-1]=1.;P.append(row)
        P=np.array(P).reshape(-1,len(coeff));y=train.log10_C_chemical.to_numpy()+(np.log10(train.velocity_um_h.to_numpy()) if m['method']!='DirectProcessSpline' else 0.)
        if family=='RateAffine':
            M=np.array(m['coefficient_tie_matrix']);theta=np.array(m['independent_coefficients']);gate('three independent rate parameters',M.shape==(7,3) and np.max(abs(M@theta-coeff))<1e-12 and np.linalg.matrix_rank(M)==3)
            X=X@M;coeff=theta;P=np.array([[0.,0.,1.]]);bounded=[1]
        else:bounded=list(range(1,1+ns)) if m['method']!='DirectProcessSpline' else []
        residual=X@coeff-y;lam=m['lambda_'];g=2*(X.T@residual/len(y)+lam*P.T@(P@coeff))
        gate('paired convex objective',abs(residual@residual/len(y)+lam*np.sum((P@coeff)**2)-r['optimization']['cost'])<1e-10)
        maximum=max(maximum,float(np.max(abs(g-r['optimization']['gradient']))));gate('paired independent normal gradient',np.max(abs(g-r['optimization']['gradient']))<1e-8)
        for j in range(len(coeff)):
            ok=(g[j]>=-1e-7 if j in bounded and coeff[j]<1e-8 else g[j]<=1e-7 if j in bounded and coeff[j]>15/np.log(10)-1e-8 else abs(g[j])<1e-7)
            gate('paired convex KKT',ok,family=family,column=j)
        gate('two observed rates in every training fold',train.velocity_um_h.nunique()==2)
        fits[(m['chemical_scenario'],family,m['lambda_'],tuple(m['train_layer_ids']))]=m
    selections=json.loads((DEST/'all_selections.json').read_text());policy=json.loads((DEST/'all_policy_choices.json').read_text());best_by_scope={}
    for r in selections:
        scope=data[data.scenario.eq(r['scenario'])].set_index('layer_id',drop=False);allowed=set(r['train_layer_ids']);candidates=[]
        if r['scope'].startswith('outerpair:'):
            held=int(r['scope'].split(':')[1]);gate('outerlevel not in hyperselection',all(scope.loc[i].surface_order!=held for i in allowed))
        for c in r['candidates']:
            errors=[]
            for f in c['folds']:
                train,test=f['train_layer_ids'],f['test_layer_ids'];gate('paired inner label isolation',not(set(train)&set(test)) and set(train)|set(test)==allowed and len(test)==2 and set(scope.loc[test].stack_id)=={'m','n'})
                m=fits[(r['scenario'],r['family'],c['lambda_'],tuple(train))]
                error=float(np.mean([abs(manual_predict(m,scope.loc[i].Tsource_nominal_C,scope.loc[i].velocity_um_h)-scope.loc[i].log10_C_chemical) for i in test]));errors.append(error)
                gate('paired independent fold score',abs(error-f['MAE'])<1e-10)
            gate('paired candidate mean',abs(np.mean(errors)-c['MAE'])<1e-10);candidates.append(c)
        chosen=min(candidates,key=lambda c:(round(c['MAE'],12),-c['lambda_']));gate('paired lambda rule',chosen['lambda_']==r['selected_lambda'])
        best_by_scope.setdefault((r['scenario'],r['scope']),[]).append(r)
    for r in policy:
        opts=best_by_scope[(r['scenario'],r['scope'])];chosen=min(opts,key=lambda q:(round(q['selection_MAE'],12),NPARAM[q['family']],-q['selected_lambda'],q['family']))
        gate('family chosen only within trainingCV',r['selected_family']==chosen['family'] and r['selected_lambda']==chosen['selected_lambda'])
    pred=read(DEST/'predictions.csv');gate('paired OOF complete',len(pred)==216 and pred.groupby(['scenario','family']).size().eq(12).all())
    for r in pred.itertuples():
        if r.family=='SelectedPolicy':
            pol=next(x for x in policy if x['scenario']==r.scenario and x['scope']=='outerpair:'+str(int(r.held_level)));family=pol['selected_family']
        else:family=r.family
        rec=next(q for q in selections if q['family']==family and q['scenario']==r.scenario and q['scope']=='outerpair:'+str(int(r.held_level)))
        m=fits[(r.scenario,family,rec['selected_lambda'],tuple(rec['train_layer_ids']))];sample=data[data.scenario.eq(r.scenario)&data.layer_id.eq(r.layer_id)].iloc[0]
        gate('independent paired OOF replay',r.layer_id not in m['train_layer_ids'] and abs(manual_predict(m,sample.Tsource_nominal_C,sample.velocity_um_h)-r.predicted_logC)<1e-10)
    b=OUT/'delivery/paired_chemical_bundle';manifest=json.loads((b/'manifest.json').read_text());gate('three selected chemical deployments',len(manifest['models'])==3)
    for f in manifest['files']:gate('paired saved model hash',e.sha(b/f['path'])==f['sha256'])
    result=dict(passed=all(c['passed'] for c in checks),checks=len(checks),rows=checks,fit_records=len(fits),outer_predictions=len(pred),selection_contexts=len(selections),
        maximum_independent_gradient_difference=maximum,known_rates_task=True,unseen_regime_task=False,no_Hall_labels_for_selection=True,goal_complete=False,
        code_hashes={str(Path(__file__).relative_to(ROOT)):e.sha(Path(__file__))})
    e.write(DEST/'audit.json',result);print('paired chemical audit',result['passed'],len(checks),'checks',flush=True);assert result['passed'],[c for c in checks if not c['passed']][:10]


if __name__=='__main__':main()
