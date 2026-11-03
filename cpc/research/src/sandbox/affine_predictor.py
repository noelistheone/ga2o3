"""Saved empirical comparison models with finite-design source corrections."""
from pathlib import Path
import json,hashlib
import numpy as np
from .affine_source import basis,fit_correction
from .functional_predictor import DopedPropertyPredictor as InputContract,row_key,sample_key
from .relative_predictor_v2 import SavedModel,features
from . import transport


class AffinePropertyPredictor:
    def __init__(self,db,bundle):
        self.bundle=Path(bundle).resolve();self.manifest=json.loads((self.bundle/'manifest.json').read_text());self.models={};self.calibrations={}
        if self.manifest.get('engine')!='affine':raise ValueError('Affine control manifest required')
        for f in self.manifest['files']:
            p=(self.bundle/f['path']).resolve()
            if self.bundle not in p.parents or hashlib.sha256(p.read_bytes()).hexdigest()!=f['sha256']:raise ValueError('Empirical model checksum/path mismatch')

    def predict(self,dopant,incorporated_cm3,*,method='DoseAffine',observation_T_K=300.,H_known=False,H_cm3=0.,
                carbon_known=False,carbon_cm3=0.,growth_T_C=None,anchors=(),source_inputs=(),pO2_atm=1e-5):
        if method not in self.manifest['variants']:raise ValueError('Unknown empirical source control')
        if not np.isfinite(pO2_atm) or pO2_atm<=0:raise ValueError('Positive finite oxygen metadata required')
        row=InputContract.parse(dopant,dict(incorporated_cm3=incorporated_cm3,observation_T_K=observation_T_K,
            H_known=H_known,H_cm3=H_cm3,carbon_known=carbon_known,carbon_cm3=carbon_cm3,growth_T_C=growth_T_C))
        nodes=[];mapping=[];seen={}
        for a in source_inputs:
            r=InputContract.parse(dopant,a)
            if r.Tobs_K!=300.:raise ValueError('Only RT source designs and anchors supported')
            k=row_key(r)
            if k not in seen:seen[k]=len(nodes);nodes.append(r)
            mapping.append(seen[k])
        if len(anchors)>2:raise ValueError('Only0/1/2 source anchors supported')
        idx=[];observed=[]
        for a in anchors:
            i=a.get('input_index')
            if not isinstance(i,(int,np.integer)) or isinstance(i,(bool,np.bool_)) or not 0<=i<len(mapping):raise ValueError('Explicit valid source input_index required')
            n=float(a['measured_nH_cm3'])
            if not np.isfinite(n) or n<1e13:raise ValueError('Uncensored Hall anchor above declared floor required')
            idx.append(mapping[i]);observed.append(float(np.log10(n)))
        if len({nodes[i].incorporated_cm3 for i in idx})!=len(idx):raise ValueError('Anchor doses must be distinct')
        if idx:
            if sample_key(row) not in {sample_key(r) for r in nodes}:raise ValueError('Query inputs outside explicitly declared design')
            order=sorted(range(len(idx)),key=lambda i:row_key(nodes[idx[i]]));idx=[idx[i] for i in order];observed=np.asarray(observed)[order]
        spec=self.manifest['variants'][method];key=spec['model_key']
        if key not in self.models:
            record=json.loads((self.bundle/'models'/f'{key}.json').read_text())
            if record['method'] not in ('ridge','line'):raise ValueError('Only empirical frozen base predictors supported')
            self.models[key]=SavedModel(record,profile=None,db=None)
        model=self.models[key]
        if model.kind=='ridge' and (growth_T_C is None or any(a.get('growth_T_C') is None for a in source_inputs)):raise ValueError('Ridge requires explicit growth metadata')
        choice=spec['source_selection'][str(len(idx))];lam=choice['lam'];coef=np.zeros(1 if spec['basis']=='slope' else 2);meta={};hit=False
        if idx and lam is not None:
            signature=(key,spec['basis'],float(lam),tuple(map(row_key,nodes)),tuple(idx),tuple(observed))
            hit=signature in self.calibrations
            if not hit:
                n0=np.array([model.point(r,require_observation=False).values[0] for r in nodes])
                solved=fit_correction(n0,np.log10([r.incorporated_cm3 for r in nodes]),idx,observed,float(lam),kind=spec['basis'])
                if not solved['valid']:raise ValueError('No valid empirical source correction')
                self.calibrations[signature]=(np.asarray(solved['coefficients']),solved)
            coef,meta=self.calibrations[signature]
        p=model.point(row,require_observation=False);n=float(p.values[0]+(basis([np.log10(row.incorporated_cm3)],spec['basis'])@coef)[0])
        if n<13.:raise ValueError('Below empirical Hall observation floor')
        mu=float(np.log10(transport.mu_caughey_thomas(10**n,row.Tobs_K)));sigma=float(np.log10(transport.E)+n+mu)
        support=self.manifest['chemical_support_cm3'][dopant];warnings=[]
        if not support[0]<=row.incorporated_cm3<=support[1]:warnings.append('outside training chemical range')
        if row.Tobs_K!=300.:warnings.append('sourceconstraints applytoRTnodes;nonewtemperaturetraining')
        if not np.isclose(np.log10(pO2_atm),-5.,atol=1e-12,rtol=0.):warnings.append('oxygen metadata unused by this empiricalcontrol')
        return dict(version=1,engine='affine',method=method,dopant=dopant,incorporated_cm3=row.incorporated_cm3,observation_T_K=row.Tobs_K,
            nH_cm3=10**n,muH_cm2Vs=10**mu,sigma_S_cm=10**sigma,log10_nH_cm3=n,log10_muH_cm2Vs=mu,log10_sigma_S_cm=sigma,
            source_coefficients=coef.tolist(),raw_source_coordinate=None,source_prior=choice,source_calibration=meta,calibration_cache_hit=hit,
            anchor_count=len(idx),source_design_size=len(nodes),raw_Sandbox_diagnostics=dict(empirical=True,same_physical_state_for_output=False),
            chemical_training_range_cm3=support,warnings=warnings,
            scope=dict(grade='retrospective empiricalcomparison;explicitfiniteRTdesign',energetic_labels=0,source_dimensions=len(coef),
                shared_weights='frozenphase20',transport='legacyempiricalCT,not adefectstate',conductivity='derived',
                uncertainty='no calibratedinterval',rareearth='unsupported'))
