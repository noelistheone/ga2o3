"""Portable training, prediction and label acquisition for the CPC article.

No historical research-script globals are used for training. Physical layers
are the frozen article layers. Input tables, held-out groups, optimization
settings and all starts are recorded in each output directory.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import argparse, hashlib, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit,logit
from cpc_numerics import BASE,RESEARCH,forward
sys.path.insert(0,str(RESEARCH/'scripts/predictive_redesign_20261019'))
from si_multidonor_overlay_v1 import make_overlay
from sandbox.kroger_db import load,NPZ
from sandbox.charge_feasible_model_v2 import ChargeFeasibleModel
from sandbox.relative_inventory_model import RelativeInventoryModel
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.material_profile import PUBLISHED_PROFILE
from sandbox.temperature_defects import TemperatureDefectLayer
from sandbox.energy_learning import relative_charge_records,ChargeEnergyLearner,features,reconstruct
from sandbox.implicit_defects import ImplicitDefectLayer
from sandbox.goal_acquisition import prediction_variance_reduction,condition_coordinate

L0,L1=np.log10(600.),np.log10(1950.)
INPUTS=('element','incorporated_cm3','Tobs_K','growth_T_C','H_known','H_cm3','scenario_pO2_atm')


def write(path,data):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x))
    path.write_text(json.dumps(data,default=convert,indent=2,allow_nan=False)+'\n')


def input_row(row):
    """The physical solver never sees measured responses or source identifiers."""
    data={k:getattr(row,k) for k in INPUTS}
    if data['H_known'] not in (True,False):raise ValueError('H_known must be Boolean')
    for key in ('incorporated_cm3','Tobs_K','scenario_pO2_atm'):
        if not np.isfinite(data[key]) or data[key]<=0:
            raise ValueError(f'{key} must be finite and positive')
    if data['H_known']:
        if not np.isfinite(data['H_cm3']) or data['H_cm3']<0:
            raise ValueError('Reported H must be finite and nonnegative')
    elif data['H_cm3']>0:
        raise ValueError('Positive H concentration requires H_known=True')
    if data['element'] not in ('Si','Sn'):
        raise ValueError('The released prediction interface supports Si and Sn')
    return SimpleNamespace(**data)


class Predictor:
    def __init__(self,kind):
        self.kind=kind;self.layers={};self.calls=0
        if kind=='inventory':
            self.db=make_overlay()[0]
            self.model=ChargeFeasibleModel(basis='process',profile=published_coherent_profile(),db=self.db)
        elif kind=='thermal':self.db=load()
        else:raise ValueError(kind)

    def point(self,row,w,derivative=True,reference=False):
        row=input_row(row);self.calls+=1
        if self.kind=='inventory':
            if reference:
                key=row.scenario_pO2_atm
                if key not in self.layers:self.layers[key]=RelativeInventoryModel(dict(field='inventory'),
                    profile=published_coherent_profile(),pO2_atm=key,db=self.db)
                a=self.layers[key].point(row,np.r_[w[:2],0.],require_observation=False)
                j=np.zeros(6);j[:2]=a.jacobian[0,:2]
            else:
                a=self.model.point(row,w,require_observation=False);j=a.jacobian[0]
            return a.values[0],j,a.diagnostics['observation_domain_valid']
        key=(row.element,bool(row.H_known))
        if key not in self.layers:
            elements=(row.element,'H') if row.H_known else (row.element,)
            self.layers[key]=TemperatureDefectLayer(self.db,elements,profile=PUBLISHED_PROFILE,
                donor_extension=True,donor_element=row.element,hall='rH1')
        l=self.layers[key];s=expit(w[0]);T=10**(L0+(L1-L0)*s)
        p=l.parameters(T=T,Nd=10**w[1],doses=(row.incorporated_cm3,row.H_cm3) if row.H_known else (row.incorporated_cm3,),
            pO2=row.scenario_pO2_atm,Tobs=row.Tobs_K,binding_eV=10**w[2],screen_s=w[3])
        if not derivative:return forward(l,p),None,True
        a=l.predict(p);chain=np.zeros((l.np,4));chain[0,0]=(L1-L0)*s*(1-s)
        chain[1,1]=1.;chain[l.ie,2]=1.;chain[l.ie+1,3]=1.
        return a.values[0],a.jacobian[0]@chain,bool(a.diagnostics['single_electron_Hall_valid'])


def weights(d,kind):
    if kind=='inventory':
        return d.scenario_weight.to_numpy(float)/(d.related_cluster.nunique()*
            d.groupby('related_cluster').family.transform('nunique').to_numpy(float)*
            d.groupby('family').sample_id.transform('nunique').to_numpy(float))
    w=1/(d.groupby('sample_id').sample_id.transform('size').to_numpy(float)*
         d.groupby('family').sample_id.transform('nunique').to_numpy(float))
    return w/w.mean()


def fit(args):
    kind=args.kind;out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(args.input);unit='related_cluster' if kind=='inventory' else 'family'
    train=d[d[unit].ne(args.held)].copy();test=d[d[unit].eq(args.held)].copy()
    if train.empty or test.empty:raise ValueError('Specify an existing nonempty held-out unit')
    if kind=='thermal':train=train[train.kind.eq('RT')|train.Tobs_K.le(275)].copy()
    train=train.sort_values(['family','sample_id','Tobs_K']).reset_index(drop=True)
    alpha=.1 if kind=='inventory' else 0.
    low=[-6.,14.,-12.,-12.,-12.,-12.] if kind=='inventory' else [-6.,14.,np.log10(.003),0.]
    high=[6.,18.,12.,12.,12.,12.] if kind=='inventory' else [6.,18.,np.log10(.2),5.]
    starts=([np.array([0.,16.,0.,0.,0.,0.]),np.array([1.,15.5,1.,0.,0.,0.])]
        if kind=='inventory' else [np.array([logit((np.log10(T)-L0)/(L1-L0)),B,np.log10(E),s])
                                   for T,B,E,s in ((1400.,16.,.03,.2),(1750.,15.5,.06,1.))])
    protocol=dict(kind=kind,held=args.held,train_units=sorted(train[unit].unique()),
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),alpha=alpha,
        starts=starts,bounds=[low,high],max_nfev=args.max_nfev,derivative=args.derivative,
        tolerances=1e-7,fd_step=1e-5,weighting='equal source, specimen, observation; inventory additionally equal related groups and conditional scenarios',
        purpose='fresh portable fit; does not replace archived headline fits',
        no_query_labels_used_in_optimization=True)
    if (out/'model.json').exists():raise RuntimeError('Refusing to replace a completed fit')
    write(out/'protocol.json',protocol)
    sw=np.sqrt(weights(train,kind));target=train.target.to_numpy(float);rows=list(train.itertuples(index=False))
    mapper=Predictor(kind);cache={};calls=0
    def calculate(w):
        nonlocal calls
        key=tuple(w)
        if key not in cache:
            calls+=1;v=[];J=[];prior=[];dprior=[]
            for row in rows:
                y,j,ok=mapper.point(row,w,derivative=args.derivative=='IFT')
                v.append(y)
                if j is not None:J.append(j)
                if alpha:
                    b,bj,_=mapper.point(row,w,reference=True)
                    prior.append(y-b);dprior.append(j-bj)
            residual=sw*(np.array(v)-target)
            jacobian=sw[:,None]*np.array(J) if J else None
            if alpha:
                residual=np.r_[residual,np.sqrt(alpha)*sw*np.array(prior)]
                jacobian=np.r_[jacobian,np.sqrt(alpha)*sw[:,None]*np.array(dprior)]
            cache.clear();cache[key]=(residual,jacobian)
        return cache[key]
    if kind=='inventory' and args.derivative!='IFT':raise ValueError('FD training benchmark is supported for thermal only')
    def jac(w):
        if args.derivative=='IFT':return calculate(w)[1]
        columns=[]
        for j in range(len(w)):
            step=1e-5;a,b=w.copy(),w.copy()
            if w[j]-step<low[j]:
                a[j]+=step;b[j]+=2*step
                column=(-3*calculate(w)[0]+4*calculate(a)[0]-calculate(b)[0])/(2*step)
            elif w[j]+step>high[j]:
                a[j]-=step;b[j]-=2*step
                column=(3*calculate(w)[0]-4*calculate(a)[0]+calculate(b)[0])/(2*step)
            else:
                a[j]+=step;b[j]-=step
                column=(calculate(a)[0]-calculate(b)[0])/(2*step)
            columns.append(column)
        return np.array(columns).T
    trials=[];start=time.perf_counter()
    for initial in starts:
        st=time.perf_counter();n0=calls;c0=mapper.calls
        try:
            result=least_squares(lambda w:calculate(w)[0],initial,jac=jac,bounds=(low,high),
                max_nfev=args.max_nfev,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            trial=dict(parameters=result.x,loss=float(result.fun@result.fun),success=bool(result.success),
                message=result.message,nfev=result.nfev,njev=result.njev)
        except Exception as exc:trial=dict(loss=None,success=False,error=repr(exc),parameters=initial)
        trial.update(seconds=time.perf_counter()-st,objective_evaluations=calls-n0,physical_calls=mapper.calls-c0)
        trials.append(trial);write(out/'trials.json',trials);print(kind,args.derivative,args.held,trial,flush=True)
    good=[x for x in trials if x['loss'] is not None and x['success']]
    if not good:raise RuntimeError('No converged start; see trials.json')
    best=min(good,key=lambda x:x['loss']);w=np.array(best['parameters'])
    training_seconds=time.perf_counter()-start;training_calls=mapper.calls
    predictions=[]
    for row in test.itertuples(index=False):
        value,_,ok=mapper.point(row,w)
        predictions.append(dict(sample_id=row.sample_id,family=row.family,Tobs_K=row.Tobs_K,
            kind=getattr(row,'kind','RT'),predicted=value,observed=row.target,observation_valid=ok))
    pd.DataFrame(predictions).to_csv(out/'heldout_predictions.csv',index=False)
    write(out/'model.json',dict(kind=kind,parameters=w,held=args.held,training_seconds=training_seconds,
        training_calls=training_calls,trials=trials,selected_loss=best['loss'],
        query_observation_valid=all(x['observation_valid'] for x in predictions),
        parameter_names=(['logit_Tf','log10_background','share_intercept','share_logdose','share_growthT','share_Sn']
            if kind=='inventory' else ['logit_Tf','log10_background','log10_E0_eV','screen_s']),
        input_fields=INPUTS,model_scope='Si/Sn incorporated chemistry with explicit input scenarios'))


def predict(args):
    m=json.loads(args.model.read_text());mapper=Predictor(m['kind']);d=pd.read_csv(args.input)
    result=[]
    for i,row in enumerate(d.itertuples(index=False)):
        y,j,ok=mapper.point(row,np.asarray(m['parameters']))
        result.append(dict(row=i,log10_nH_cm3=y,observation_valid=ok,
                           parameter_gradient=json.dumps(j.tolist())))
    args.output.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(result).to_csv(args.output,index=False)


def acquire(args):
    """Train an energy prior, select labels, reveal, predict: no electrical labels."""
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    protocol=dict(element=args.element,model='extra_trees',config=dict(max_depth=None,min_samples_leaf=2),
        budget=args.budget,design_T=[850.,1325.,1700.],design_C=[1e17,1e18,1e19,1e20],
        test_T=[1100.,1500.],test_C=[3e17,3e18,3e19,3e20],
        covariance='tree ensemble covariance plus train-only leave-one-element-out mean squared discrepancy',
        labels='retrospective masked published charge energy reveal, not newly calculated DFT',
        prior_config='fixed demonstration configuration; historical selected configurations remain in archived results')
    write(out/'protocol.json',protocol);start=time.perf_counter();db=load()
    records,_=relative_charge_records(db,np.load(NPZ)['cs_num_each_site'])
    tr=[r for r in records if r['element']!=args.element];rr=[r for r in records if r['element']==args.element]
    if not 0<=args.budget<=len(rr):raise ValueError('Budget must not exceed the masked energetic coordinate count')
    learner=ChargeEnergyLearner('extra_trees',protocol['config']).fit(tr)
    errs=[]
    for e in sorted({r['element'] for r in tr}):
        a=[r for r in tr if r['element']!=e];b=[r for r in tr if r['element']==e]
        sub=ChargeEnergyLearner('extra_trees',protocol['config']).fit(a)
        errs.extend((sub.predict(b)-np.array([r['crossing_eV'] for r in b])).tolist())
    X=(features(rr)-learner.center)/learner.scale
    trees=np.array([t.predict(X)+learner.mean for t in learner.model.estimators_])
    cov=np.cov(trees,rowvar=False)+np.mean(np.array(errs)**2)*np.eye(len(rr));mean=learner.predict(rr)
    B=np.zeros((db.nCS,len(rr)))
    for j,r in enumerate(rr):B[r['index'],j]=-r['delta_q']
    test=[(T,C) for T in protocol['test_T'] for C in protocol['test_C']]
    design=[(T,C) for T in protocol['design_T'] for C in protocol['design_C']]
    truth=np.array([r['crossing_eV'] for r in rr]);oracle=ImplicitDefectLayer(db,(args.element,))
    targets=[forward(oracle,oracle.parameters(T=T,doses=(C,))) for T,C in test]
    acquired=[];rows=[]
    for budget in range(args.budget+1):
        layer=ImplicitDefectLayer(reconstruct(db,rr,mean),(args.element,),B)
        for (T,C),target in zip(test,targets):
            y=forward(layer,layer.parameters(T=T,doses=(C,)))
            rows.append(dict(budget=budget,Tf_K=T,C_cm3=C,predicted_log_nH=y,full_catalogue_log_nH=target))
        if budget==args.budget:break
        J=np.array([layer.predict(layer.parameters(T=T,doses=(C,))).jacobian[0,2:2+len(rr)] for T,C in design])
        score=prediction_variance_reduction(J,cov);score[acquired]=-np.inf;j=int(score.argmax())
        mean,cov=condition_coordinate(mean,cov,j,truth[j]);acquired.append(j)
    pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False)
    write(out/'selected.json',dict(labels=[rr[j]['label_id'] for j in acquired],seconds=time.perf_counter()-start,
        mean_absolute_property_error_by_budget={str(k):float(np.mean([abs(r['predicted_log_nH']-r['full_catalogue_log_nH']) for r in rows if r['budget']==k])) for k in range(args.budget+1)}))


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('fit');f.add_argument('--kind',choices=['inventory','thermal'],required=True)
    f.add_argument('--input',type=Path,required=True);f.add_argument('--held',required=True)
    f.add_argument('--output',type=Path,required=True);f.add_argument('--max-nfev',type=int,default=80)
    f.add_argument('--derivative',choices=['IFT','FD'],default='IFT');f.set_defaults(run=fit)
    q=sub.add_parser('predict');q.add_argument('--model',type=Path,required=True)
    q.add_argument('--input',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.set_defaults(run=predict)
    a=sub.add_parser('acquire');a.add_argument('--element',choices=['Si','Sn'],required=True)
    a.add_argument('--budget',type=int,default=2);a.add_argument('--output',type=Path,required=True);a.set_defaults(run=acquire)
    args=p.parse_args();args.run(args)


if __name__=='__main__':main()
