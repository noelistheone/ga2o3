"""Reusable, saved ML+Sandbox property prediction on an explicit input design.

No training/experimental table is read here. Source updates are conditional on
the supplied input set and at most two explicit Hall anchor labels.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
import numpy as np
from .context_saved_model import SavedContextModel,INPUT_NAMES
from .pool_share_saved_model import SavedPoolModel
from .context_property_predictor_v2 import json_native
from .context_source import solve_context_source,solve_affine_context
from .coherent_fermi import published_coherent_profile
from .kroger_db import NPZ


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_input(value):
    if set(value)!=set(INPUT_NAMES):
        raise ValueError('Explicit input fields required; missing='+str(sorted(set(INPUT_NAMES)-set(value)))+
            '; unexpected='+str(sorted(set(value)-set(INPUT_NAMES))))
    row=dict(value)
    if row['element'] not in ('Si','Sn'):raise ValueError('Current trained domain is Si/Sn')
    for name in ('H_known','carbon_known'):
        if not isinstance(row[name],(bool,np.bool_)):raise ValueError(name+' must be a boolean')
        row[name]=bool(row[name])
    for name in set(INPUT_NAMES)-{'element','H_known','carbon_known','oxygen_input_kind'}:
        row[name]=float(row[name])
        if not np.isfinite(row[name]):raise ValueError('Finite '+name+' required')
    if row['incorporated_cm3']<=0 or row['scenario_pO2_atm']<=0 or row['Tobs_K']!=300.:
        raise ValueError('Positive dose/declared oxygen scenario and the 300K comparison task required')
    for flag,amount in (('H_known','H_cm3'),('carbon_known','carbon_cm3')):
        if row[amount]<0 or row[flag]!=(row[amount]>0):raise ValueError('Explicit known-positive or unknown-zero '+amount+' required')
    if not isinstance(row['oxygen_input_kind'],str) or not row['oxygen_input_kind'].strip():
        raise ValueError('Explicit oxygen interpretation required;O3/plasma pressure is not equilibrium O2')
    return row


class _PoolPropertyPredictor:
    def __init__(self,bundle):
        self.bundle=Path(bundle).resolve()
        self.manifest=json.loads((self.bundle/'manifest.json').read_text())
        if self.manifest['version']!=1:raise ValueError('Unknown context predictor bundle version')
        if self.manifest['profile_sha256']!=published_coherent_profile().fingerprint:
            raise ValueError('Material/Fermi reference differs from training')
        if sha(NPZ)!=self.manifest['physical_reference_sha256']:
            raise ValueError('Frozen physical defect reference checksum mismatch')
        self.paths={};self.fits={};self.models={};self.source_cache={}
        for f in self.manifest['files']:
            p=(self.bundle/f['path']).resolve()
            if self.bundle not in p.parents or sha(p)!=f['sha256']:raise ValueError('Saved-model path/checksum mismatch')
            self.paths[f['path']]=p

    def model(self,preset):
        if preset not in self.manifest['presets']:raise ValueError('Unknown saved preset')
        spec=self.manifest['presets'][preset];key=spec['model_key']
        if key not in self.models:
            name='models/'+key+'.json'
            if name not in self.paths:raise ValueError('Model is not in the verified bundle ledger')
            self.fits[key]=json.loads(self.paths[name].read_text())
            factory=SavedPoolModel if spec.get('model_type')=='pool_share' else SavedContextModel
            self.models[key]=factory(self.fits[key])
        return self.models[key],spec

    def predict(self,*,preset,inputs,anchors=()):
        if not inputs:raise ValueError('At least one declared prediction input required')
        if len(anchors)>2:raise ValueError('Saved source policies support0/1/2Hall anchors')
        canonical=[checked_input(v) for v in inputs]
        if len({r['element'] for r in canonical})!=1:raise ValueError('One dopant species per source adaptation required')
        # Match the experiment's unique input design. Repeated measurement IDs
        # do not silently increase the response-regularization weight.
        unique={json.dumps(r,sort_keys=True):r for r in canonical}
        ordered=sorted(unique,key=lambda key:(unique[key]['incorporated_cm3'],key))
        nodes=[SimpleNamespace(**unique[k]) for k in ordered];lookup={k:i for i,k in enumerate(ordered)}
        mapping=[lookup[json.dumps(r,sort_keys=True)] for r in canonical]
        ai=[];y=[]
        for a in anchors:
            if set(a)!={'input_index','measured_nH_cm3'}:raise ValueError('Anchor needs only input_index and measured_nH_cm3')
            index=a['input_index']
            if not isinstance(index,int) or isinstance(index,bool) or not 0<=index<len(inputs):raise ValueError('Anchor input_index out of declared design')
            value=float(a['measured_nH_cm3'])
            if not np.isfinite(value) or value<1e13:raise ValueError('Positive Hall anchor within the declared detection domain required')
            ai.append(mapping[index]);y.append(np.log10(value))
        if len(set(ai))!=len(ai):raise ValueError('Two identical input tuples cannot be counted as distinct anchors')
        model,spec=self.model(preset);budget=len(ai);lam=spec['source_lambdas'][str(budget)];kind=spec['source_kind']
        cache_key=json.dumps(dict(preset=preset,inputs=[unique[k] for k in ordered],anchor_indices=ai,labels=y),sort_keys=True)
        if cache_key in self.source_cache:
            points,solution,point_errors=self.source_cache[cache_key]
        else:
            cache={};point_errors={}
            def safe_point(index,z,callback):
                try:return callback()
                except (ValueError,RuntimeError,ArithmeticError,np.linalg.LinAlgError) as exc:
                    point_errors.setdefault(str(float(z)),{})[index]=repr(exc)
                    return None
            def points_at(z):
                z=float(z)
                if z not in cache:
                    if kind=='offset' and z!=0:
                        cache[z]=[None if b is None else safe_point(i,z,lambda:model.offset_point(r,b,z,require_observation=False))
                                  for i,(r,b) in enumerate(zip(nodes,points_at(0.)))]
                    else:
                        actual='none' if kind=='affine' or not budget else kind
                        cache[z]=[safe_point(i,z,lambda:model.point(r,source_kind=actual,shift=z,require_observation=False))
                                  for i,r in enumerate(nodes)]
                return cache[z]
            base=points_at(0.)
            if not budget or lam is None:
                points=base;solution=dict(raw_shift=0.,source_parameters=[0.,0.] if kind=='affine' else None,
                    valid=all(p is not None and p.valid for p in base),status='no_anchor' if not budget else 'trained_no_update_policy',lam=lam)
            elif kind=='affine':
                if any(p is None for p in base):raise ValueError('Affine reference has a failed numerical state')
                result=solve_affine_context([p.log_nH for p in base],np.log10([r.incorporated_cm3 for r in nodes]),ai,y,[lam])
                solution=result['solutions'][0]
                if not solution['finite_response_bound_valid']:raise ValueError('Affine candidate outside declared finite response bound')
                points=[model.offset_point(r,b,z,require_observation=False) for r,b,z in zip(nodes,base,solution['response_delta'])]
                solution.update(raw_shift=None,valid=all(p.valid for p in points))
            else:
                def forward(z):
                    ps=points_at(z)
                    return np.array([p.log_nH if p is not None else np.nan for p in ps]),np.array([p.source_derivative if p is not None else np.nan for p in ps]),np.array([p is not None and p.valid for p in ps])
                result=solve_context_source(forward,ai,y,[lam],linear=kind=='offset')
                solution=result['solutions'][0]
                if not solution['valid']:raise ValueError('Saved source policy did not yield a valid candidate')
                points=points_at(solution['raw_shift'])
            self.source_cache[cache_key]=(points,solution,point_errors)
        result=[]
        for index,(r,ni) in enumerate(zip(canonical,mapping)):
            p=points[ni];warnings=[];support=spec['training_support'].get(r['element'])
            if support is None: warnings.append('element absent from this fitted training subset')
            else:
                if not support['dose_cm3'][0]<=r['incorporated_cm3']<=support['dose_cm3'][1]:warnings.append('chemical dose outside this model training support')
                if not support['growth_T_C'][0]<=r['growth_T_C']<=support['growth_T_C'][1]:warnings.append('growth setting outside this model training support')
            if p is None:
                failure=point_errors.get(str(float(solution.get('raw_shift') or 0.)),{}).get(ni,'Numerical state unavailable')
                result.append(dict(input_index=index,anchor=index in [a['input_index'] for a in anchors],
                    numeric=False,valid=False,error=failure,nH_cm3=None,muH_cm2Vs=None,sigma_S_cm=None,
                    log10_nH_cm3=None,log10_muH_cm2Vs=None,log10_sigma_S_cm=None,
                    log10_mu_CT=None,log10_sigma_CT=None,continuation_log10_nH=None,
                    physical_state_available=False,same_physical_state_for_output=False,warnings=warnings))
                continue
            physical=p.diagnostics.get('physical_state_available',False)
            result.append(dict(input_index=index,anchor=index in [a['input_index'] for a in anchors],numeric=True,valid=p.valid,
                log10_nH_cm3=p.log_nH if p.valid else None,log10_muH_cm2Vs=p.log_muH if p.valid else None,
                log10_sigma_S_cm=p.log_sigma if p.valid else None,
                nH_cm3=10**p.log_nH if p.valid else None,muH_cm2Vs=10**p.log_muH if p.valid else None,
                sigma_S_cm=10**p.log_sigma if p.valid else None,
                log10_mu_CT=p.log_mu_CT if p.valid else None,log10_sigma_CT=p.log_sigma_CT if p.valid else None,
                continuation_log10_nH=None if p.valid else p.log_nH,
                physical_state_available=physical,same_physical_state_for_output=p.diagnostics.get('same_physical_state_for_output',False),
                context_field=p.diagnostics.get('context_field_value'),warnings=warnings))
        return dict(preset=preset,model_key=spec['model_key'],source_kind=kind,Hall_anchor_budget=budget,
            source_lambda=lam,source_solution=solution,full_valid_coverage=all(r['valid'] for r in result),predictions=result,
            information='saved shared weights,explicit input design and permitted Hall anchor labels only;no query labels',
            grade='retrospective conditional Si/Sn research predictor at300K;code-domain flags are not materialaccuracy guarantees',
            transport='physical presets:declared coherent state;empirical presets:CT(predictednH);uniform CT readout also returned',
            conductivity='derived from predictednH/mu;not an independent validation label',
            uncertainty='no calibrated predictive confidence interval supplied')

    def zero_anchor_gradient(self,*,preset,input):
        row=SimpleNamespace(**checked_input(input));model,spec=self.model(preset)
        if model.engine is None:raise ValueError('This IFT interface requires an actual physical Sandbox preset')
        p=model.engine.point(row,model.fit['parameters'],require_observation=True)
        J=np.r_[p.jacobian[[0]],p.transport_jacobian]
        X=np.r_[p.input_jacobian[[0]],p.transport_input_jacobian]
        return dict(preset=preset,model_key=spec['model_key'],properties=['log10_nH','log10_muH','log10_sigma'],
            parameter_names=model.engine.parameter_names,parameter_jacobian=J.tolist(),input_names=p.input_names,input_jacobian=X.tolist(),
            includes_learned_input_chain=True,
            learning_coordinate='absolute atomshare logit or explicit reference' if spec.get('model_type')=='pool_share' else 'legacy conditional inventory/output relation',
            actual_evaluation_method=model.fit.get('evaluation_method',model.fit['method']),
            source_and_hyperparameter_refit_derivatives=False,
            temperature_derivative_scope='local mathematical sensitivity at300K;not new validated temperature-dependent material accuracy',
            assumption='fixed learned parameters,oxygen scenario and explicit effective endpoint closure')



class PoolPropertyPredictor(_PoolPropertyPredictor):
    def predict(self,**kwargs):return json_native(super().predict(**kwargs))
    def zero_anchor_gradient(self,**kwargs):return json_native(super().zero_anchor_gradient(**kwargs))
