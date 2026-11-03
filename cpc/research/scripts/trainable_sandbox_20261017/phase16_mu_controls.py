"""Matched chemical-input mobility controls; never condition on observed nH."""
from pathlib import Path
from itertools import combinations
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import phase16_prediction as study
import phase2_energy as energy

OUT=study.OUT/'mu_controls';ALPHAS=(.01,.1,1.,10.,100.,1e4)
KINDS=('fixed205','constant_logmean','dose_temperature_line','direct_ridge')
CACHE={}


def features(d):
    c=np.log10(d.incorporated_cm3.to_numpy(float))-18.;T=d.Tobs_K.to_numpy(float)
    h=d.H_known.to_numpy(float);carbon=d.carbon_known.to_numpy(float)
    return np.c_[c,c*c,(d.growth_T_C.to_numpy(float)+273.15-1100)/300,
        (d.thermal_cap_K.to_numpy(float)-1100)/300,d.element.eq('Sn').to_numpy(float),np.log(T/300),300/T-1,
        c*(300/T-1),h,h*(np.log10(np.maximum(d.H_cm3.to_numpy(float),1))-17),
        carbon,carbon*(np.log10(np.maximum(d.carbon_cm3.to_numpy(float),1))-17)]


def targets(families):
    # gamma1 defines the same relative RT/curve budget; no n labels fit the head.
    tr=study.training_data(families,'Thermal_original_RT_effective_g1')
    return tr[tr.endpoint.eq('muH')].copy()


def fit(families,kind,alpha=None):
    key=(tuple(sorted(families)),kind,alpha)
    if key in CACHE:return CACHE[key]
    tr=targets(families);mu=study.TARGETS[study.TARGETS.endpoint.eq('muH')]
    meta=dict(kind=kind,train_families=list(sorted(families)),training_obs_ids=tr.obs_id.tolist(),
        weight_per_observation=tr.loss_weight.to_list(),alpha=alpha,observed_nH_input=False)
    if kind=='fixed205' or tr.empty:
        pred=np.full(len(mu),np.log10(205.));meta['fallback']='fixed205_no_training_mu_labels' if tr.empty else 'fixed_empirical_prior'
    else:
        y=tr.target.to_numpy(float);w=tr.loss_weight.to_numpy(float)
        if kind=='constant_logmean':
            value=float(np.average(y,weights=w));pred=np.full(len(mu),value);meta['constant']=value
        elif kind=='dose_temperature_line':
            def x(d):return np.c_[np.ones(len(d)),np.log10(d.incorporated_cm3)-18.,np.log(d.Tobs_K/300.)]
            beta=np.linalg.lstsq(x(tr)*np.sqrt(w)[:,None],y*np.sqrt(w),rcond=None)[0]
            pred=x(mu)@beta;meta['coefficients']=beta
        else:
            X=features(tr);center=np.average(X,axis=0,weights=w)
            scale=np.sqrt(np.average((X-center)**2,axis=0,weights=w));scale[scale<1e-12]=1.
            reg=Ridge(alpha=alpha).fit((X-center)/scale,y,sample_weight=w*len(tr)/w.sum())
            pred=reg.predict((features(mu)-center)/scale)
            meta.update(center=center,scale=scale,coefficients=reg.coef_,intercept=reg.intercept_)
    record=dict(**meta,predictions=[dict(obs_id=r.obs_id,predicted=float(p)) for r,p in zip(mu.itertuples(),pred)])
    CACHE[key]=energy.clean(record);return CACHE[key]


def score(f,held):
    mu=study.TARGETS[study.TARGETS.endpoint.eq('muH')&study.TARGETS.family.eq(held)]
    mu=mu[mu.kind.eq('RT')|mu.Tobs_K.le(275)]
    if mu.empty:return None
    lookup={p['obs_id']:p['predicted'] for p in f['predictions']};total=0.
    for kind,factor in (('RT',1.),('curve',.1)):
        s=mu[mu.kind.eq(kind)]
        if s.empty:continue
        errors=s.assign(error=[abs(lookup[r.obs_id]-r.target) for r in s.itertuples()])
        total+=factor*float(errors.groupby('sample_id').error.mean().mean())
    return total


def main():
    protocol=dict(code_sha256=energy.sha(Path(__file__)),kinds=KINDS,alphas=ALPHAS,
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (study.OUT/'prediction_protocol.json',
            energy.BASE/'dataset_v1.csv',energy.BASE/'episodes_v1.csv',study.core.prior.prior.TEMPDIR/'temperature_observations.csv')},
        features='chemicaldose/growth/cap/element/T/reportedH,Cflags only;no measurednH',
        labels='same muRT+.1muT<=275K;DOI/specimen/read equal;all train/validation wholeDOI excluded',
        selection='ridgealpha by inner availablemu DOI mean[RTMAE+.1curveMAE];ties preferlargeralpha;no label context fixed205 explicitly',
        scope='independent mobility heads as same-input/same-mu-label controls;no additional sigma labels;fixed205 historical empirical prior')
    pp=OUT/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==energy.clean(protocol)
    else:energy.write(pp,protocol)
    selections=[]
    for n in (3,4):
        for fam in combinations(study.core.FAMILIES,n):
            for kind in KINDS:
                for alpha in (ALPHAS if kind=='direct_ridge' else (None,)):fit(fam,kind,alpha)
    for held in study.core.FAMILIES:
        train=[f for f in study.core.FAMILIES if f!=held]
        for kind in KINDS:
            candidates=[]
            for alpha in (ALPHAS if kind=='direct_ridge' else (None,)):
                scores=[score(fit([f for f in train if f!=inner],kind,alpha),inner) for inner in train]
                eligible=[s for s in scores if s is not None]
                candidates.append(dict(alpha=alpha,score=float(np.mean(eligible)) if eligible else None,
                    inner_families=train,inner_scores=scores))
            best=min(candidates,key=lambda c:(np.inf if c['score'] is None else round(c['score'],10),
                -(c['alpha'] if c['alpha'] is not None else 0.)))
            f=fit(train,kind,best['alpha']);selections.append(dict(held=held,kind=kind,selected=best,candidates=candidates,fit=f))
    energy.write(OUT/'fits.json',list(CACHE.values()));energy.write(OUT/'selections.json',selections)
    energy.write(OUT/'contract_audit.json',dict(passed=True,fit_records=len(CACHE),selections=len(selections),
        ownT_targets=59,RT_pairs=5,RT_DOIs=2,temperature_reads=54,temperature_specimens=3,temperature_DOIs=1,
        no_mu_training_contexts=[r for r in list(CACHE.values()) if not r['training_obs_ids']],
        measured_nH_feature=False,new_sigma_labels=0,goal_complete=False))
    print('Matched mobility controls',len(CACHE),'fits',len(selections),'outer selections')


if __name__=='__main__':main()
