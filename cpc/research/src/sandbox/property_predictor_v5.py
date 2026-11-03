"""Unified, explicit-budget Si/Sn research prediction.

Uses saved models only. No property tables or query measurement labels are read.
The energy-completion and property-supervised branches keep different budgets.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from scipy.special import expit
from .trainable_predictor_v4 import DopedPropertyPredictor as EnergyPredictor
from .trainable_predictor import file_sha256
from .trainable_predictor_v2 import structure_fingerprint, supplied_energy_fingerprint
from .material_profile import MaterialProfile
from .donor_mixture_model import DonorMixtureModel, L0, L1
from .coherent_hall_model import CoherentHallModel
from .coherent_fermi import CoherentMaterialProfile
from .thermal_coherent_model import ThermalCoherentHallModel
from .parent_allocation import ParentAllocationLayer
from .scalar_profile_adaptation import ScalarProfile
from . import transport


def make_model(record, db):
    kind=record['kind']
    profile=(CoherentMaterialProfile if kind in ('coherent','thermal') else MaterialProfile).from_dict(record['profile'])
    cls={'legacy':DonorMixtureModel,'coherent':CoherentHallModel,
         'thermal':ThermalCoherentHallModel}.get(kind)
    if cls is not None:
        return cls(record['definition'],profile=profile,pO2_atm=1e-5,db=db)
    if kind=='stock':
        return ConstantStock(record,db)
    raise ValueError('Unknown internal model kind')


class ConstantStock:
    """Same bounded constant parent-allocation relation as phase12, with IFT."""
    def __init__(self,record,db):
        self.record=record;self.db=db;self.layers={}
        self.profile=MaterialProfile.from_dict(record['profile'])

    def point(self,row,parameters,*,source_kind='none',source_shift=0.):
        w=np.asarray(parameters,float); z=float(source_shift)
        if source_kind not in ('none','offset','stock'):
            raise ValueError('Unsupported constant-stock source coordinate')
        if source_kind=='stock' and abs(z)>6+1e-12:
            raise ValueError('Internal coordinate outside [-6,6]')
        elements=(row.element,'H') if row.H_known else (row.element,)
        if elements not in self.layers:
            self.layers[elements]=ParentAllocationLayer(self.db,elements,profile=self.profile,extension=False)
        v=w[2]+(np.log(10.)*z if source_kind=='stock' else 0.)
        tanh=np.tanh(v/24.); bias=24.*tanh; dbias=1.-tanh*tanh
        st=expit(w[0]); Tf=10.**(L0+(L1-L0)*st)
        doses=(row.incorporated_cm3,row.H_cm3) if row.H_known else (row.incorporated_cm3,)
        pred,J,_=self.layers[elements].point(Tf_K=Tf,background_cm3=10.**w[1],
            doses=doses,Tobs_K=row.Tobs_K,E0_eV=.0366,s=0.,bias=bias,pO2_atm=1e-5)
        values=pred.values.copy();jac=np.c_[J[:,0]*(L1-L0)*st*(1-st),J[:,1],J[:,4]*dbias]
        source_jac=np.zeros(3)
        if source_kind=='stock':source_jac=J[:,4]*dbias*np.log(10.)
        if source_kind=='offset':values[0]-=z;source_jac[0]=-1.
        return SimpleNamespace(values=values,jacobian=jac,source_jacobian=source_jac,
            physical_values=pred.values.copy(),diagnostics={**pred.diagnostics,
                'process_bias':float(bias),'Tf_K':float(Tf),
                'same_physical_state_for_output':not(source_kind=='offset' and z!=0)})


class DopedPropertyPredictor:
    def __init__(self,db,bundle):
        self.db=db;self.bundle=Path(bundle).resolve()
        self.manifest=json.loads((self.bundle/'manifest.json').read_text())
        if self.manifest['version']!=5 or structure_fingerprint(db)!=self.manifest['structure_fingerprint']:
            raise ValueError('Version5 or catalogue contract mismatch')
        for entry in self.manifest['files']:
            p=(self.bundle/entry['path']).resolve()
            if self.bundle not in p.parents or file_sha256(p)!=entry['sha256']:
                raise ValueError('Bundle checksum/path mismatch')
        parent=(self.bundle/self.manifest['energy_bundle_relative']).resolve()
        if file_sha256(parent/'manifest.json')!=self.manifest['energy_manifest_sha256']:
            raise ValueError('Inherited energy model manifest changed')
        self.energy=EnergyPredictor(db,parent);self.models={}

    @staticmethod
    def _row(dopant,C,T,H_known,H,growth_T_C):
        if dopant not in ('Si','Sn'):raise ValueError('Current trained scope is Si/Sn')
        if not isinstance(H_known,(bool,np.bool_)):raise ValueError('H_known must be an explicit boolean')
        if not np.isfinite([C,T,H]).all() or C<=0 or T<=0 or H<0:
            raise ValueError('Positive finite dose/temperature and nonnegative H required')
        if (H_known and H<=0) or (not H_known and H!=0):
            raise ValueError('H supply must match its declared known/unknown status')
        if growth_T_C is not None and not np.isfinite(growth_T_C):raise ValueError('Invalid growth metadata')
        # This input is inactive in ALL current presets. Zero is an internal
        # placeholder for an unused library field, never an inferred growth T.
        return SimpleNamespace(element=dopant,incorporated_cm3=C,Tobs_K=T,H_known=bool(H_known),
            H_cm3=H,growth_T_C=0. if growth_T_C is None else float(growth_T_C))

    def predict(self,dopant,incorporated_cm3,*,method='ridge_energy_tuned',
                observation_T_K=300.,H_known=False,H_cm3=0.,growth_T_C=None,
                anchors=(),pO2_atm=1e-5):
        C=float(incorporated_cm3);T=float(observation_T_K);H=float(H_cm3)
        row=self._row(dopant,C,T,H_known,H,growth_T_C)
        if method not in self.manifest['methods']:raise ValueError('Unknown method')
        if not np.isfinite(pO2_atm) or not np.isclose(np.log10(pO2_atm),-5.,atol=1e-12,rtol=0.):
            raise ValueError('This fitted collection uses the declared 1e-5 atm scenario only')
        anchors=list(anchors)
        if len(anchors)>2:raise ValueError('Only 0/1/2 distinct-dose same-condition anchors supported')
        ar=[];y=[]
        for a in anchors:
            ar.append(self._row(dopant,float(a['incorporated_cm3']),T,H_known,H,growth_T_C))
            n=float(a['measured_nH_cm3'])
            if not np.isfinite(n) or n<=0:raise ValueError('Invalid anchor Hall density')
            y.append(np.log10(n))
        if len({a.incorporated_cm3 for a in ar})!=len(ar):raise ValueError('Anchor doses must be distinct')
        spec=self.manifest['methods'][method]
        if spec['branch']=='energy':
            if T!=300. or H_known or H!=0:raise ValueError('Inherited energy presets require 300K and the zero-H scenario')
            result=self.energy.predict(dopant,C,method=method,anchors=anchors,pO2_atm=pO2_atm)
            result['collection_version']=5;result['active_inputs']=['element','actual_incorporated_dose','explicit_energy_budget','anchor_nH']
            result['growth_T_C_metadata_only']=growth_T_C
            return result
        b=len(ar);choice=spec['anchor_penalty'][str(b)];lam=np.inf if choice is None else float(choice)
        z=0.;status='no_update';model=None
        if spec['branch']=='empirical':
            if T!=300.:raise ValueError('Dose-only control is a 300K predictor')
            def raw(r):return np.array([np.log10(r.incorporated_cm3)+spec['activation_offset_dex']])
        else:
            if supplied_energy_fingerprint(self.db)!=self.manifest['full_energy_fingerprint']:
                raise ValueError('Internal property-learning models require the full supplied energy library')
            key=spec['model_key'];record=json.loads((self.bundle/'models'/f'{key}.json').read_text())
            if key not in self.models:self.models[key]=make_model(record,self.db)
            model=self.models[key];w=record['parameters']
            def raw(r):return model.point(r,w).values
        kind=spec.get('source_kind','offset')
        if ar and np.isfinite(lam):
            if kind=='stock':
                def forward(v):
                    out=[model.point(a,w,source_kind='stock',source_shift=v) for a in ar]
                    return np.array([p.values[0] for p in out]),np.array([p.source_jacobian[0] for p in out])
                fit=ScalarProfile(forward).fit(np.asarray(y),lam);z=float(fit.shift);status=fit.status
            else:
                z=float(np.sum(np.array([raw(a)[0] for a in ar])-np.asarray(y))/(b+lam));status='closed_output_offset'
        point=None
        if model is not None:
            point=model.point(row,w,source_kind=kind,source_shift=z)
            logn=float(point.values[0]);dg=point.diagnostics
            if not dg.get('single_electron_Hall_valid',True):raise ValueError('Single-electron observation domain failed')
            if dg.get('free_n_cm3',np.inf)<1e13:raise ValueError('Carrier detection floor failed')
            logmu=float(point.transport_values[0]) if hasattr(point,'transport_values') else None
        else:logn=float(raw(row)[0]-z);dg=None;logmu=None
        if logn<13.:raise ValueError('Predicted Hall density below the declared 1e13 cm^-3 floor')
        if logmu is None and T==300.:
            logmu=float(np.log10(transport.mu_caughey_thomas(10.**logn,300.))+self.manifest['transport_offset_dex'])
        logsigma=None if logmu is None else float(np.log10(transport.E)+logn+logmu)
        support=self.manifest['domain']['chemical_cm3'][dopant]
        warnings=[]
        if not support[0]<=C<=support[1]:warnings.append('outside chemical training range')
        if T!=300.:warnings.append('temperature evidence uses old Si curves; Sn temperature transfer is unvalidated' if dopant=='Sn' else 'temperature evidence uses seven old Si specimens')
        if dg and 'coherent_transport' in dg:warnings.append('Born/RTA/single-band transport assumptions; numerical state is not material validation')
        if C in {a.incorporated_cm3 for a in ar}:warnings.append('prediction is at an anchor dose, not a held-out query')
        return dict(collection_version=5,dopant=dopant,method=method,incorporated_cm3=C,observation_T_K=T,
            nH_cm3=10.**logn,muH_cm2Vs=None if logmu is None else 10.**logmu,
            sigma_S_cm=None if logsigma is None else 10.**logsigma,
            log10_nH_cm3=logn,log10_muH_cm2Vs=logmu,log10_sigma_S_cm=logsigma,
            anchor_count=b,source_coordinate=z,source_kind=kind,source_status=status,shrinkage_lambda=choice,
            raw_Sandbox_diagnostics=dg,physical_state_shared_with_readout=None if model is None else kind=='stock' or z==0.,
            energy_budget=spec['energy_budget'],active_inputs=spec['active_inputs'],
            growth_T_C_metadata_only=growth_T_C,oxygen_scenario_atm=1e-5,
            unknown_H_zero_scenario=not H_known,chemical_training_range_cm3=support,warnings=warnings,
            scope=dict(grade='conditional retrospective research prediction; full-five-source fit is not a held-out test',
                learning=spec['learning'],transport=spec['transport'],conductivity='derived q*nH*muH',
                process='effective shared closure; no measured thermal/oxygen history inferred',
                uncertainty='no calibrated prediction interval; no rare-earth validation'))
