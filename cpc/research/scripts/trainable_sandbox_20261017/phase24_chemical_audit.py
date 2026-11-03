"""Independently integrate piecewise slopes, verify convex optima, folds and process derivatives."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from sandbox.chemical_process import ChemicalCalibration,SavedChemicalCalibrations
from csv_io_v2 import read_result_csv as read
import phase2_energy as e
BASE=e.BASE;OUT=BASE/'phase24_chemical_process';DEST=OUT/'chemical_learning'
KB=8.617333262145e-5;K=1/(KB*1500)-1/(KB*(np.array([1100.,1150.,1200.,1250.])+273.15))


def basis_integral(T,ns):
    q=1/(KB*1500)-1/(KB*(T+273.15))
    if ns==1:return np.array([q])
    edges=sorted([min(0.,q),max(0.,q)]+[float(k) for k in K if min(0.,q)<k<max(0.,q)])
    b=np.zeros(5)
    for lo,hi in zip(edges[:-1],edges[1:]):b[int(np.sum(K<(lo+hi)/2))]+=np.sign(q)*(hi-lo)
    return b


def manual_matrix(m,d):
    ns=1 if m['method']=='ArrheniusMassBalance' else 5;has=m['method'] in ('RegimeFlux','DirectProcessSpline');rows=[]
    for r in d.itertuples():
        values=[1.,*basis_integral(r.Tsource_nominal_C,ns)]
        if has:values.append(np.log10(r.velocity_um_h)-m['rate_center'])
        rows.append(values)
    return np.array(rows)


def manual_predict(m,T,v):
    ns=1 if m['method']=='ArrheniusMassBalance' else 5;coef=np.asarray(m['coefficients']);r=np.log10(v)
    y=coef[0]+coef[1:1+ns]@basis_integral(T,ns)
    if m['method'] in ('RegimeFlux','DirectProcessSpline'):y+=coef[-1]*(r-m['rate_center'])
    return float(y-(r if m['method']!='DirectProcessSpline' else 0.))


def main():
    checks=[]
    def gate(name,good,**extra):checks.append(dict(name=name,passed=bool(good),**extra))
    for rel in ('extraction/protocol.json','chemical_learning/protocol.json'):
        p=json.loads((OUT/rel).read_text())
        for key in ('code_hashes','input_hashes'):
            for f,h in p[key].items():gate('frozen protocol hash',e.sha(ROOT/f)==h,path=f)
    d=read(OUT/'extraction/layer_readings.csv');fits=[json.loads(p.read_text()) for p in (DEST/'fits').glob('*.json')];lookup={}
    for rec in fits:
        m=rec['model'];scope=d[d.scenario.eq(m['chemical_scenario'])].set_index('layer_id',drop=False);train=scope.loc[m['train_layer_ids']]
        ns=1 if m['method']=='ArrheniusMassBalance' else 5;has=m['method'] in ('RegimeFlux','DirectProcessSpline');coef=np.array(m['coefficients']);lam=m['lambda_']
        X=manual_matrix(m,train);r=np.log10(train.velocity_um_h.to_numpy());y=train.log10_C_chemical.to_numpy()+(r if m['method']!='DirectProcessSpline' else 0.)
        P=[]
        for j in range(ns-1):
            a=np.zeros(len(coef));a[j+1]=-1/np.sqrt(ns-1);a[j+2]=1/np.sqrt(ns-1);P.append(a)
        if has:a=np.zeros(len(coef));a[-1]=1.;P.append(a)
        P=np.array(P).reshape(-1,len(coef));res=X@coef-y;pen=P@coef;g=2*(X.T@res/len(y)+lam*P.T@pen)
        gate('independent convex objective',abs(res@res/len(y)+lam*(pen@pen)-rec['optimization']['cost'])<1e-10,key=rec['key'])
        gate('independent coefficient gradient',np.max(abs(g-rec['optimization']['gradient']))<1e-8,key=rec['key'])
        for j in range(len(coef)):
            if not rec['optimization']['active_columns'][j]:gate('unobserved rate coefficient zero',coef[j]==0.,key=rec['key']);continue
            bounded=m['method']!='DirectProcessSpline' and 1<=j<ns+1
            kkt=(g[j]>=-1e-7 if bounded and coef[j]<1e-8 else g[j]<=1e-7 if bounded and coef[j]>15/np.log(10)-1e-8 else abs(g[j])<1e-7)
            gate('convex KKT certificate',kkt,key=rec['key'],column=j,gradient=float(g[j]))
        gate('training rank and distinct identities',rec['optimization']['rank']==rec['optimization']['columns'] and len(set(m['train_layer_ids']))==len(train),key=rec['key'])
        lookup[(m['chemical_scenario'],m['method'],lam,tuple(m['train_layer_ids']))]=m
    selections=json.loads((DEST/'all_selections.json').read_text())
    for rec in selections:
        scope=d[d.scenario.eq(rec['scenario'])].set_index('layer_id',drop=False);allowed=set(rec['train_layer_ids']);candidates=[]
        if rec['scope'].startswith('outer:'):
            held=rec['scope'].split(':')[1];gate('whole physical stack excluded from hyperselection',all(scope.loc[x].stack_id!=held for x in allowed),scope=rec['scope'],method=rec['method'])
        for candidate in rec['candidates']:
            errs=[]
            for fold in candidate['folds']:
                train=fold['train_layer_ids'];test=fold['test_layer_ids'];gate('inner label isolation',not(set(train)&set(test)) and set(train)|set(test)==allowed,method=rec['method'])
                m=lookup[(rec['scenario'],rec['method'],candidate['lambda_'],tuple(train))]
                val=np.mean([abs(manual_predict(m,scope.loc[x].Tsource_nominal_C,scope.loc[x].velocity_um_h)-scope.loc[x].log10_C_chemical) for x in test])
                gate('reconstructed inner chemical score',abs(val-fold['MAE'])<1e-10);errs.append(float(val))
            gate('mean folds correctly scored',abs(np.mean(errs)-candidate['MAE'])<1e-10)
            candidates.append(candidate)
        chosen=min(candidates,key=lambda c:(round(c['MAE'],12),-c['lambda_']))
        gate('training-only lambda selection',chosen['lambda_']==rec['lambda_'])
    pred=read(DEST/'predictions.csv');gate('complete chemical heldout predictions',len(pred)==144 and pred.groupby(['method','scenario','held_stack']).size().eq(6).all())
    for r in pred.itertuples():
        choice=next(a for a in selections if a['method']==r.method and a['scenario']==r.scenario and a['scope']=='outer:'+r.held_stack)
        m=lookup[(r.scenario,r.method,r.lambda_,tuple(choice['train_layer_ids']))]
        expected=manual_predict(m,r.Tsource_nominal_C,r.velocity_um_h)
        gate('independent outer chemical replay',abs(expected-r.predicted_log10_C)<1e-10 and r.layer_id not in m['train_layer_ids'],id=r.layer_id)
    bundle=SavedChemicalCalibrations(OUT/'delivery/chemical_bundle');maxderiv=maxvalue=0.
    for name in bundle.manifest['models']:
        model=bundle.get(name);m=model.record
        for T in (1083.2,1136.7,1183.5,1233.4,1280.1,1319.,1350.):
            for v in (.86,1.,1.3):
                p=model.predict(T,v);expect=manual_predict(m,T,v);maxvalue=max(maxvalue,abs(expect-p['log10_C_cm3']))
                step=1e-3;fd=(model.predict(T+step,v)['log10_C_cm3']-model.predict(T-step,v)['log10_C_cm3'])/(2*step)
                dv=(model.predict(T,v*10**1e-5)['log10_C_cm3']-model.predict(T,v/10**1e-5)['log10_C_cm3'])/(2e-5)
                maxderiv=max(maxderiv,abs(fd-p['dlog10C_dsourceC']),abs(dv-p['dlog10C_dlog10v']))
                gate('independent positive process prediction',p['C_cm3']>0 and np.isfinite(p['C_cm3']) and abs(expect-p['log10_C_cm3'])<1e-10,model=name)
                gate('chemical source/rate analytic derivatives',abs(fd-p['dlog10C_dsourceC'])<1e-7 and abs(dv-p['dlog10C_dlog10v'])<1e-7,model=name)
                if m['method']!='DirectProcessSpline':gate('declared monotonicity',p['dlog10C_dsourceC']>=0,model=name)
    result=dict(passed=all(r['passed'] for r in checks),checks=len(checks),rows=checks,fit_records=len(fits),selected_contexts=len(selections),
        outer_predictions=len(pred),saved_models=len(bundle.manifest['models']),maximum_derivative_error=maxderiv,maximum_manual_prediction_error=maxvalue,
        no_Hall_training_or_hyperselection=True,independent_DOI=1,independent_stacks=2,goal_complete=False,
        code_hashes={str(Path(__file__).relative_to(ROOT)):e.sha(Path(__file__))})
    e.write(DEST/'audit.json',result);print('chemical audit',result['passed'],len(checks),'checks maxderivative',maxderiv,flush=True)
    assert result['passed'],[c for c in checks if not c['passed']][:20]


if __name__=='__main__':main()
