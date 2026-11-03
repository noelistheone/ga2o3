"""Saved ML+Sandbox inference conditional on explicit, unlabeled source inputs."""
from pathlib import Path
import json,hashlib,copy
import numpy as np
from .relative_predictor_v2 import SavedModel,DopedPropertyPredictor as ParentPredictor
from .coherent_fermi import CoherentMaterialProfile
from .functional_source import ResponseDesign,solve_response
from .trainable_predictor_v2 import structure_fingerprint,supplied_energy_fingerprint


def row_key(r):return (r.element,r.incorporated_cm3,r.Tobs_K,r.H_known,r.H_cm3,r.carbon_known,r.carbon_cm3,r.growth_T_C)
def sample_key(r):return row_key(r)[:2]+row_key(r)[3:]


class DopedPropertyPredictor:
    def __init__(self,db,bundle):
        self.db=db;self.bundle=Path(bundle).resolve();self.manifest=json.loads((self.bundle/'manifest.json').read_text())
        if self.manifest['version']!=8 or structure_fingerprint(db)!=self.manifest['structure_fingerprint']:raise ValueError('Version8 catalogue contract mismatch')
        for f in self.manifest['files']:
            p=(self.bundle/f['path']).resolve()
            if self.bundle not in p.parents or hashlib.sha256(p.read_bytes()).hexdigest()!=f['sha256']:raise ValueError('Model checksum/path mismatch')
        self.profile=CoherentMaterialProfile.from_dict(self.manifest['profile']);self.models={};self.calibrations={};self.parent=None

    @staticmethod
    def parse(dopant,entry):
        if entry.get('dopant',dopant)!=dopant:raise ValueError('One dopant per declared source design')
        return ParentPredictor.row(dopant,float(entry['incorporated_cm3']),float(entry.get('observation_T_K',300.)),
            entry.get('H_known',False),float(entry.get('H_cm3',0.)),entry.get('carbon_known',False),float(entry.get('carbon_cm3',0.)),entry.get('growth_T_C'))

    def predict(self,dopant,incorporated_cm3,*,method='FieldCurveSafe',observation_T_K=300.,
                H_known=False,H_cm3=0.,carbon_known=False,carbon_cm3=0.,growth_T_C=None,
                anchors=(),source_inputs=(),pO2_atm=1e-5):
        if not np.isfinite(pO2_atm) or pO2_atm<=0:raise ValueError('Positive finite oxygen scenario required')
        row=ParentPredictor.row(dopant,float(incorporated_cm3),float(observation_T_K),H_known,float(H_cm3),carbon_known,float(carbon_cm3),growth_T_C)
        anchors=list(anchors);source_inputs=list(source_inputs)
        if len(anchors)>2:raise ValueError('Only0/1/2 RT specimen anchors supported')
        nodes=[];seen={};mapping=[]
        for entry in source_inputs:
            r=self.parse(dopant,entry)
            if r.Tobs_K!=300.:raise ValueError('Source design and anchors were trained at300K only')
            k=row_key(r)
            if k not in seen:seen[k]=len(nodes);nodes.append(r)
            mapping.append(seen[k])
        ai=[];observed=[]
        for a in anchors:
            idx=a.get('input_index')
            if not isinstance(idx,(int,np.integer)) or isinstance(idx,(bool,np.bool_)) or not 0<=idx<len(mapping):raise ValueError('Anchor input_index must identify a supplied source input')
            measured=float(a['measured_nH_cm3'])
            if not np.isfinite(measured) or measured<1e13:raise ValueError('Finite uncensored RT Hall anchor atdeclared floor required')
            ai.append(mapping[idx]);observed.append(float(np.log10(measured)))
        if len({nodes[i].incorporated_cm3 for i in ai})!=len(ai):raise ValueError('Anchor doses must be distinct')
        if ai:
            if sample_key(row) not in {sample_key(r) for r in nodes}:raise ValueError('Query chemical/metadata inputs must be in the declared source design')
            order=sorted(range(len(ai)),key=lambda i:row_key(nodes[ai[i]]));ai=[ai[i] for i in order];observed=np.asarray(observed)[order]
        if method in self.manifest.get('energy_methods',[]):
            if self.parent is None:
                parent=(self.bundle/self.manifest['parent_bundle_relative']).resolve()
                if hashlib.sha256((parent/'manifest.json').read_bytes()).hexdigest()!=self.manifest['parent_manifest_sha256']:raise ValueError('Inherited model manifest changed')
                self.parent=ParentPredictor(self.db,parent)
            converted=[dict(incorporated_cm3=nodes[i].incorporated_cm3,measured_nH_cm3=10**y,H_known=nodes[i].H_known,H_cm3=nodes[i].H_cm3) for i,y in zip(ai,observed)]
            p=self.parent.predict(dopant,row.incorporated_cm3,method=method,observation_T_K=row.Tobs_K,H_known=H_known,H_cm3=H_cm3,
                carbon_known=carbon_known,carbon_cm3=carbon_cm3,growth_T_C=growth_T_C,anchors=converted,pO2_atm=pO2_atm)
            p.update(version=8,branch='inherited_energy_completion',source_design_ignored_by_legacy_energy_preset=True);return p
        if method not in self.manifest['variants']:raise ValueError('Unknown saved prediction method')
        if supplied_energy_fingerprint(self.db)!=self.manifest['full_energy_fingerprint']:raise ValueError('Supplied energetic library differs')
        spec=self.manifest['variants'][method];key=(spec['model_key'],float(pO2_atm))
        record=json.loads((self.bundle/'models'/f"{spec['model_key']}.json").read_text())
        if record['method']=='ridge' and (growth_T_C is None or any(s.get('growth_T_C') is None for s in source_inputs)):raise ValueError('Ridge requires query and source growth metadata')
        if key not in self.models:self.models[key]=SavedModel(record,profile=self.profile,db=self.db,pO2_atm=pO2_atm)
        model=self.models[key];choice=spec['source_selection'][str(len(ai))];lam=choice['lam'];u=0.;metadata={};cache_hit=False
        if ai and lam is not None:
            signature=(key,spec['source_kind'],spec['mode'],choice['prior'],float(lam),tuple(map(row_key,nodes)),tuple(ai),tuple(observed))
            cache_hit=signature in self.calibrations
            if not cache_hit:
                def forward(z):
                    a=[model.point(r,source_kind=spec['source_kind'],shift=z,require_observation=False) for r in nodes]
                    return np.array([x.values[0] for x in a]),np.array([x.source_jacobian[0] for x in a]),np.array([x.diagnostics['observation_domain_valid'] for x in a])
                penalty=spec['penalty'];pi=[ai[0]] if penalty=='first' else ai if penalty=='anchors' else list(range(len(nodes)))
                vi=list(range(len(nodes))) if spec['safe'] else ai
                design=ResponseDesign(forward,ai,pi,vi,linear=spec['source_kind']=='offset',point_budget=32768)
                solved=solve_response(design,observed,[float(lam)],kind=choice['prior'])['solutions'][0]
                if not solved['valid']:raise ValueError('Source inference did not produce a complete valid solution')
                self.calibrations[signature]=(float(solved['raw_shift']),dict(solution=solved,point_calls=design.points,vector_calls=design.calls))
            u,metadata=self.calibrations[signature]
        result=model.point(row,source_kind=spec['source_kind'],shift=u);n=float(result.values[0]);mu=float(result.transport_values[0]);sigma=float(result.transport_values[1])
        support=self.manifest['chemical_support_cm3'][dopant];warnings=[]
        if not support[0]<=row.incorporated_cm3<=support[1]:warnings.append('outside training chemical range')
        if row.Tobs_K!=300.:warnings.append('query temperature not in RT source-design validity checks;no newtemperature/mobility training')
        if not np.isclose(np.log10(pO2_atm),-5.,atol=1e-12,rtol=0.):warnings.append('uncalibrated oxygen sensitivity scenario')
        if model.physical and (carbon_known or any(r.carbon_known for r in nodes)):warnings.append('carbon inputabsentfrom energeticcatalogue andnot physicallymodeled')
        return dict(version=8,method=method,dopant=dopant,incorporated_cm3=row.incorporated_cm3,observation_T_K=row.Tobs_K,
            nH_cm3=10**n,muH_cm2Vs=10**mu,sigma_S_cm=10**sigma,log10_nH_cm3=n,log10_muH_cm2Vs=mu,log10_sigma_S_cm=sigma,
            anchor_count=len(ai),source_design_size=len(nodes),source_kind=spec['source_kind'],source_mode=spec['mode'],raw_source_coordinate=u,
            source_prior=choice,source_calibration=metadata,calibration_cache_hit=cache_hit,raw_Sandbox_diagnostics=result.diagnostics,
            chemical_training_range_cm3=support,oxygen_scenario_atm=float(pO2_atm),warnings=warnings,
            scope=dict(prediction='known-input-setconditional retrospectiveSi/Snresearch',state_constraint='finite declaredRTnodes,not acontinuumortemperaturecertificate',
                shared_learning='phase20weights/gammafrozen;newsourcehyperparameterselectiononly',thermal='effectiveTf,not measuredhistory',
                growth='metadata onlyinphysicalstate;Ridgefeature',
                transport='coherentFD/BornRTAwith205empiricalextra' if model.physical else 'legacyempiricalCT',
                conductivity='derivedq*nH*muH',rareearth='unsupported',uncertainty='no calibratedpredictioninterval'))
