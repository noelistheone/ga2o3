"""Saved finite-state source inference and same-state Hall property decisions."""
from pathlib import Path
import json,hashlib
import numpy as np
from .functional_predictor import DopedPropertyPredictor as Contract,row_key,sample_key
from .relative_predictor_v2 import SavedModel,DopedPropertyPredictor as Rows
from .source_ensemble import SourceEnsemble,source_grid,weighted_index
from .coherent_fermi import CoherentMaterialProfile
from .trainable_predictor_v2 import structure_fingerprint,supplied_energy_fingerprint


class FiniteEnsemblePredictor:
    def __init__(self,bundle,db=None):
        self.bundle=Path(bundle).resolve();self.manifest=json.loads((self.bundle/'manifest.json').read_text());self.db=db
        if self.manifest['version']!=9:raise ValueError('Version9 finite-state model required')
        for f in self.manifest['files']:
            p=(self.bundle/f['path']).resolve()
            if self.bundle not in p.parents or hashlib.sha256(p.read_bytes()).hexdigest()!=f['sha256']:raise ValueError('Saved model checksum/path mismatch')
        self.profile=CoherentMaterialProfile.from_dict(self.manifest['profile']);self.models={};self.sources={};self.query_cache={}

    def model(self,key,pressure):
        signature=(key,float(pressure))
        if signature not in self.models:
            record=json.loads((self.bundle/'models'/f'{key}.json').read_text())
            if record['method'] in ('inventory','output'):
                if self.db is None:
                    from .kroger_db import load
                    self.db=load()
                if structure_fingerprint(self.db)!=self.manifest['structure_fingerprint'] or supplied_energy_fingerprint(self.db)!=self.manifest['full_energy_fingerprint']:
                    raise ValueError('Supplied defect catalogue differs from trained model')
            self.models[signature]=SavedModel(record,profile=self.profile,db=self.db,pO2_atm=pressure)
        return self.models[signature]

    def predict(self,dopant,incorporated_cm3,*,method,observation_T_K=300.,H_known=False,H_cm3=0.,
                carbon_known=False,carbon_cm3=0.,growth_T_C=None,anchors=(),source_inputs=(),pO2_atm=1e-5):
        if method not in self.manifest['variants']:raise ValueError('Explicit saved ensemble method required')
        if not np.isfinite(pO2_atm) or pO2_atm<=0:raise ValueError('Positive finite oxygen scenario required')
        row=Rows.row(dopant,float(incorporated_cm3),float(observation_T_K),H_known,float(H_cm3),carbon_known,float(carbon_cm3),growth_T_C)
        source_inputs=list(source_inputs);anchors=list(anchors)
        if len(anchors)>2:raise ValueError('Only0/1/2 distinct-dose RT specimen anchors supported')
        parsed=[Contract.parse(dopant,r) for r in source_inputs]
        if any(r.Tobs_K!=300. for r in parsed):raise ValueError('Declared source nodes must be at300K')
        unique={row_key(r):r for r in parsed};nodes=[unique[k] for k in sorted(unique)];nodeindex={row_key(r):i for i,r in enumerate(nodes)}
        ai=[];observed=[]
        for a in anchors:
            i=a.get('input_index');y=float(a['measured_nH_cm3'])
            if not isinstance(i,(int,np.integer)) or isinstance(i,(bool,np.bool_)) or not 0<=i<len(parsed):raise ValueError('Anchor index must identify supplied source input')
            if not np.isfinite(y) or y<1e13:raise ValueError('Finite uncensored Hall anchor above declared floor required')
            ai.append(nodeindex[row_key(parsed[i])]);observed.append(np.log10(y))
        if len({nodes[i].incorporated_cm3 for i in ai})!=len(ai):raise ValueError('Distinct anchor doses required')
        if ai:
            if sample_key(row) not in {sample_key(r) for r in nodes}:raise ValueError('Query chemistry/metadata must be in declared source design')
            order=np.argsort(ai);ai=[ai[i] for i in order];observed=np.asarray(observed)[order]
        spec=self.manifest['variants'][method];model=self.model(spec['model_key'],pO2_atm);kind=spec['source_kind'];choice=spec['source_selection'][str(len(ai))]
        record=model.record
        if record['method']=='ridge' and (growth_T_C is None or any(x.get('growth_T_C') is None for x in source_inputs)):
            raise ValueError('Ridge requires query/source growth metadata')
        cachehit=False;sourcecalls=0
        if not ai:
            result=model.point(row,source_kind=kind,shift=0.);point=np.array([result.values[0],*result.transport_values]);u=0.;uncertainty={};querycalls=1
        else:
            signature=(spec['model_key'],float(pO2_atm),kind,tuple(sorted(choice.items())),tuple(map(row_key,nodes)),tuple(ai),tuple(observed))
            cachehit=signature in self.sources
            if not cachehit:
                ugrid,bw=source_grid(self.manifest['grid_size']);zero=len(ugrid)//2;values=np.full((len(ugrid),len(nodes),3),np.nan);valid=np.zeros((len(ugrid),len(nodes)),bool);failures=[]
                if len(ugrid)*len(nodes)>32768:raise ValueError('Declared design exceeds source point budget')
                for j,r in enumerate(nodes):
                    v=model.point(r,source_kind=kind,shift=0.,require_observation=False);sourcecalls+=1
                    values[zero,j]=[v.values[0],*v.transport_values];valid[zero,j]=v.diagnostics['observation_domain_valid']
                for i,z in enumerate(ugrid):
                    if i==zero:continue
                    if kind=='offset' and model.physical:
                        values[i]=values[zero]+[z,0.,z];valid[i]=valid[zero]&(values[i,:,0]>=13.);continue
                    for j,r in enumerate(nodes):
                        sourcecalls+=1
                        try:
                            v=model.point(r,source_kind=kind,shift=float(z),require_observation=False);values[i,j]=[v.values[0],*v.transport_values];valid[i,j]=v.diagnostics['observation_domain_valid']
                        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:failures.append(dict(coordinate=float(z),node=j,error=repr(exc)))
                ensemble=SourceEnsemble(ugrid,bw,values[:,:,0],np.all(valid,axis=1));posterior=ensemble.posterior(ai,observed,**choice)
                self.sources[signature]=(ugrid,values,posterior,sourcecalls,failures,int(ensemble.valid.sum()))
            ugrid,values,posterior,_,failures,valid_states=self.sources[signature];weights=posterior.weights;positive=np.flatnonzero(weights>0);querycalls=0
            if row.Tobs_K==300.:
                j=nodeindex[row_key(row)];qvalues=values[:,j];qvalid=np.isfinite(qvalues).all(axis=1)
            else:
                qvalues=np.full((len(ugrid),3),np.nan);qvalid=np.zeros(len(ugrid),bool)
                for i in positive:
                    k=(spec['model_key'],float(pO2_atm),kind,float(ugrid[i]),row_key(row))
                    if k not in self.query_cache:
                        querycalls+=1
                        try:
                            v=model.point(row,source_kind=kind,shift=float(ugrid[i]),require_observation=False)
                            self.query_cache[k]=(np.array([v.values[0],*v.transport_values]),bool(v.diagnostics['observation_domain_valid']))
                        except (ValueError,RuntimeError,np.linalg.LinAlgError):self.query_cache[k]=(np.full(3,np.nan),False)
                    qvalues[i],qvalid[i]=self.query_cache[k]
                invalid_mass=float(weights[~qvalid].sum())
                if invalid_mass>0:raise ValueError(f'Positive posterior support outside query-temperature observation domain; invalid_mass={invalid_mass:.17g}; no renormalization')
            selected=positive[weighted_index(qvalues[positive,0],weights[positive])];point=qvalues[selected];u=float(ugrid[selected])
            latent=[float(qvalues[positive[weighted_index(qvalues[positive,0],weights[positive],q)],0]) for q in (.025,.975)]
            uncertainty=dict(latent_log_nH_95_interval=latent,posterior_spike_probability=posterior.spike_posterior_probability,
                posterior_zero_state_probability=float(weights[len(ugrid)//2]),effective_states=float(1/(weights@weights)),
                valid_finite_states=valid_states,positive_weight_states=len(positive),working_sigma_dex=choice['sigma'],
                calibrated_material_interval=False,source_point_failures=failures)
        nh,mu,sigma=map(float,point)
        if abs(sigma-(np.log10(1.602176634e-19)+nh+mu))>1e-8:raise ValueError('Hall property identity violated')
        support=self.manifest['chemical_support_cm3'][dopant];warnings=[]
        if not support[0]<=row.incorporated_cm3<=support[1]:warnings.append('outside training chemical range')
        if row.Tobs_K!=300.:warnings.append('no newtemperature/mobility supervision; all positive posterior states must satisfy query domain')
        if model.physical and (carbon_known or any(r.carbon_known for r in nodes)):warnings.append('carbon absent from defect catalogue; metadata only')
        if not np.isclose(np.log10(pO2_atm),-5.,rtol=0.,atol=1e-12):warnings.append('uncalibrated oxygen sensitivity scenario')
        return dict(version=9,method=method,dopant=dopant,incorporated_cm3=row.incorporated_cm3,observation_T_K=row.Tobs_K,
            log10_nH_cm3=nh,log10_muH_cm2Vs=mu,log10_sigma_S_cm=sigma,nH_cm3=10**nh,muH_cm2Vs=10**mu,sigma_S_cm=10**sigma,
            anchor_count=len(ai),source_design_size=len(nodes),raw_source_coordinate=u,source_kind=kind,source_prior=choice,
            source_calibration=uncertainty,calibration_cache_hit=cachehit,source_point_calls=sourcecalls,query_point_calls=querycalls,
            warnings=warnings,scope=dict(prediction='known-input-set conditional Si/Sn retrospective finite model',
                shared_learning='phase20 weights and globalgamma frozen; training-family CV of source distribution only',
                point_action='latentlognH weighted median; mu/sigma from that state; not noisy-observation Bayes median',
                posterior='declared finite measure; not continuum or microscopic posterior; no calibrated material intervals',
                source='same posterior,query-specific state summary; not one common point parameter',
                thermal='effectiveTf,not measured process history',conductivity='derived q*nH*muH',rareearth='unsupported'))
