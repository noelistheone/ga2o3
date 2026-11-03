"""Supplementary slope/affine controls, frozen before main aggregate inspection."""
from pathlib import Path
import sys,json,copy,time,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from sandbox.affine_source import basis,fit_correction
from sandbox import transport
import phase21_prediction as s
OUT=s.OUT/'affine_controls'
CONFIG={'RidgeSlope':('DirectRidge','slope'),'RidgeAffine':('DirectRidge','affine'),'DoseAffine':('DoseLine','affine')}


def freeze():
    assert json.loads((OUT/'numerical_gates_v2.json').read_text())['passed']
    paths=(Path(__file__),ROOT/'src/sandbox/affine_source.py')
    inputs=(OUT/'amendment.json',OUT/'numerical_gates_v2.json',s.OUT/'prediction_protocol.json',s.OUT/'request_plans.json')
    p=dict(created_utc=s.energy.now(),code_hashes={str(p.relative_to(ROOT)):s.energy.sha(p) for p in paths},
        input_hashes={str(p.relative_to(ROOT)):s.energy.sha(p) for p in inputs},config=CONFIG,lambdas=s.LAM,
        source_penalty='quadraticonly;mean_design_correction_squared<=36,allnodeHall>=1e13',
        capacity='slope1coordinate,affine2coordinates;affinehasmorecapacity,sameanchorbudget',
        scope='extra controls;main14variantcode/protocol/resultsunchanged',new_shared_training=0,goal_complete=False)
    q=OUT/'protocol.json'
    if q.exists():
        old=json.loads(q.read_text())
        for k in ('code_hashes','input_hashes'):assert old[k]==p[k]
    else:s.energy.write(q,p)


def episode(variant,f,path,held,budget):
    key=s.digest(dict(protocol=s.energy.sha(OUT/'protocol.json'),variant=variant,fit=f['key'],held=held,budget=budget));dest=OUT/'episodes'/f'{key}.json'
    if dest.exists():return json.loads(dest.read_text())
    clock=time.perf_counter();d=s.table(held,budget);nodes,ai,y,aids=s.design_inputs(d);m=s.model(f);kind=CONFIG[variant][1]
    base=np.array([m.point(r,source_kind='offset',shift=0.,require_observation=False).values[0] for r in nodes]);C=np.log10([r.incorporated_cm3 for r in nodes]);results=[]
    allrows=[(r,m.point(s.SimpleNamespace(**s.inputs(r)),source_kind='offset',shift=0.,require_observation=False).values[0]) for r in d.itertuples()]
    for lam in (s.LAM if budget else (None,)):
        try:sol=dict(valid=True,coefficients=np.zeros(1 if kind=='slope' else 2),penalty=0.,status='no_update') if lam is None else fit_correction(base,C,ai,y,float(lam),kind=kind)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:sol=dict(valid=False,error=repr(exc))
        rec=[]
        for r,b in allrows:
            row=dict(sample_id=r.sample_id,family=held,element=r.element,role=r.role,observed=float(r.target))
            if sol['valid']:
                n=float(b+(basis([np.log10(r.incorporated_cm3)],kind)@sol['coefficients'])[0]);valid=n>=13.
                mu=float(np.log10(transport.mu_caughey_thomas(10**n,r.Tobs_K)))
                row.update(predicted=n,predicted_log_mu=mu,predicted_log_sigma=float(np.log10(1.602176634e-19)+n+mu),
                    numeric=True,valid=valid,error=None if valid else 'belowempiricalHallfloor')
            else:row.update(predicted=None,numeric=False,valid=False,error=sol['error'])
            rec.append(row)
        q=[r for r in rec if r['role']=='query'];mae=float(np.mean([abs(r['predicted']-r['observed']) for r in q])) if all(r['numeric'] and r['valid'] for r in q) else np.inf
        results.append(dict(prior='quadratic',lam=lam,solution=sol,MAE=mae,records=rec))
    e=dict(key=key,variant=variant,fit_key=f['key'],fit_path=str(path.relative_to(ROOT)),train_families=f['train_families'],held=held,budget=budget,
        anchor_ids=aids,anchor_labels=y,known_input_nodes=[vars(r) for r in nodes],query_labels_used_in_fit=False,results=results,seconds=time.perf_counter()-clock)
    s.energy.write(dest,e);return s.energy.clean(e)


def pick(e,lam):return next(r for r in e['results'] if r['lam']==lam)


def main():
    freeze();oldplans=json.loads((s.OUT/'request_plans.json').read_text());allchoices=[]
    for variant,(basevariant,kind) in CONFIG.items():
        rows=[];choices=[]
        for p in [r for r in oldplans if r['variant']==basevariant]:
            held,train,reg=p['held'],p['train_families'],p['reg'];f,path=s.getfit(basevariant,train,reg);selection=[]
            for budget in (0,1,2):
                candidates=[]
                if budget:
                    inner=[]
                    for i in train:
                        fi,pi=s.getfit(basevariant,[x for x in train if x!=i],reg);inner.append(episode(variant,fi,pi,i,budget))
                    for lam in s.LAM:
                        scores=[s.number(pick(e,lam)['MAE']) for e in inner]
                        candidates.append(dict(prior='quadratic',lam=lam,score=float(np.mean(scores)),inner_scores=scores,inner_keys=[e['key'] for e in inner]))
                    best=min(candidates,key=lambda r:(round(s.number(r['score']),10),-s.number(r['lam'])))
                else:best=dict(prior='quadratic',lam=None,score=None)
                c=dict(variant=variant,parent_variant=basevariant,basis=kind,held=held,train_families=train,reg=reg,fit_key=f['key'],fit_path=str(path.relative_to(ROOT)),
                    budget=budget,selected_source=best,source_candidates=candidates)
                if held is not None:
                    e=episode(variant,f,path,held,budget);z=pick(e,best['lam']);c.update(outer_episode_key=e['key'],outer_MAE=z['MAE'],source_coefficients=z['solution'].get('coefficients'))
                    rows += [dict(**r,variant=variant,budget=budget,prior='quadratic',lam=best['lam'],fit_key=f['key'],episode_key=e['key']) for r in z['records']]
                selection.append(c)
            if held is None:s.energy.write(OUT/'deployment_selection'/f'{variant}.json',selection)
            else:choices+=selection
        s.energy.csvwrite(OUT/'variants'/variant/'predictions.csv',rows);s.energy.write(OUT/'variants'/variant/'selections.json',choices);allchoices+=choices
    s.energy.write(OUT/'all_selections.json',allchoices)
    # Only copied full-source empirical records are needed for inference.
    bundle=OUT/'predictor_bundle';records={};variants={};parent=s.PARENT/'delivery_v2/predictor_bundle_v7'
    for variant,(basevariant,kind) in CONFIG.items():
        selection=json.loads((OUT/'deployment_selection'/f'{variant}.json').read_text());key=selection[0]['fit_key'];src=parent/'models'/f'{key}.json';dest=bundle/'models'/src.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert s.energy.sha(dest)==s.energy.sha(src)
        else:shutil.copyfile(src,dest)
        records[key]=dict(path=str(dest.relative_to(bundle)),sha256=s.energy.sha(dest));variants[variant]=dict(model_key=key,basis=kind,parent_variant=basevariant,
            source_selection={str(r['budget']):dict(prior='quadratic',lam=r['selected_source']['lam']) for r in selection})
    s.energy.write(bundle/'manifest.json',dict(version=1,engine='affine',created_utc=s.energy.now(),variants=variants,files=list(records.values()),
        scope='explicitfiniteRTinputdesign,0/1/2indexedRTanchors;empiricalcontrol,not internalstate',training_labels=dict(nH_RT=29,sources=5,mu=0,energetic=0),
        chemical_support_cm3=json.loads((parent/'manifest.json').read_text())['chemical_support_cm3'],goal_complete=False))
    s.energy.write(OUT/'execution_complete.json',dict(created_utc=s.energy.now(),variants=3,episodes=len(list((OUT/'episodes').glob('*.json'))),new_shared_training=0,goal_complete=False))
    print('affine controls completed',len(allchoices),'outer choices',flush=True)


if __name__=='__main__':main()
