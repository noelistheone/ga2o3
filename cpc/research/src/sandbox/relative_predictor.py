"""Saved ML + relative-inventory Sandbox property predictor, no table reads."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib
import numpy as np
from .relative_inventory_model import RelativeInventoryModel
from .coherent_fermi import CoherentMaterialProfile
from .observation_adaptation import ObservationCoordinate,solve_effect
from .trainable_predictor_v2 import structure_fingerprint,supplied_energy_fingerprint
from .trainable_predictor_v4 import DopedPropertyPredictor as EnergyPredictor
from . import transport


def features(rows):
    c=np.array([np.log10(r.incorporated_cm3)-18. for r in rows]);t=np.array([r.Tobs_K for r in rows])
    g=np.array([(r.growth_T_C+273.15-1100)/300. for r in rows]);sn=np.array([r.element=='Sn' for r in rows],float)
    hk=np.array([r.H_known for r in rows],float);ck=np.array([r.carbon_known for r in rows],float)
    h=np.log10(np.maximum([r.H_cm3 for r in rows],1.))-17.;carbon=np.log10(np.maximum([r.carbon_cm3 for r in rows],1.))-17.
    return np.c_[c,c*c,g,sn,np.log(t/300.),300./t-1,c*(300./t-1),hk,h*hk,ck,carbon*ck]


class SavedModel:
    def __init__(self,record,*,profile,db,pO2_atm=1e-5):
        self.record=record;self.kind=record['method'];self.physical=self.kind in ('inventory','output')
        self.model=RelativeInventoryModel(dict(field=self.kind),profile=profile,pO2_atm=pO2_atm,db=db) if self.physical else None

    def point(self,row,*,source_kind='offset',shift=0.,require_observation=True):
        r=self.record
        if self.physical:return self.model.point(row,r['parameters'],source_kind=source_kind,source_shift=shift,require_observation=require_observation)
        if self.kind=='ridge':
            x=(features([row])-np.array(r['center']))/np.array(r['scale']);n=float(r['intercept']+(x@np.array(r['coefficients']))[0])
        else:
            x=np.log10(row.incorporated_cm3+(row.H_cm3 if self.kind=='unitH' else 0.))
            n=float(r['coefficients'][0]+r['coefficients'][1]*(x-18.)) if self.kind=='line' else float(x+r['offset_dex'])
        n+=float(shift)
        if require_observation and n<13.:raise ValueError('Empirical readout below declared detection floor')
        mu=float(np.log10(transport.mu_caughey_thomas(10.**n,row.Tobs_K)))
        return SimpleNamespace(values=np.array([n]),source_jacobian=np.ones(1),
            transport_values=np.array([mu,np.log10(transport.E)+n+mu]),
            diagnostics=dict(observation_domain_valid=n>=13.,empirical=True,
                same_physical_state_for_output=False,mobility_scope='legacy empiricalCT(nHpred,T),not a microscopic state'))

    def anchors(self,rows,kind,z):
        p=[self.point(r,source_kind=kind,shift=z,require_observation=False) for r in rows]
        return np.array([v.values[0] for v in p]),np.array([v.source_jacobian[0] for v in p]),np.array([v.diagnostics['observation_domain_valid'] for v in p])


class DopedPropertyPredictor:
    def __init__(self,db,bundle):
        self.db=db;self.bundle=Path(bundle).resolve();self.manifest=json.loads((self.bundle/'manifest.json').read_text())
        if self.manifest['version']!=6 or structure_fingerprint(db)!=self.manifest['structure_fingerprint']:
            raise ValueError('Version6 catalogue contract mismatch')
        for f in self.manifest['files']:
            p=(self.bundle/f['path']).resolve()
            if self.bundle not in p.parents or hashlib.sha256(p.read_bytes()).hexdigest()!=f['sha256']:raise ValueError('Model bundle checksum/path mismatch')
        self.profile=CoherentMaterialProfile.from_dict(self.manifest['profile']);self.models={};self.energy_model=None;self.source_cache={}

    @staticmethod
    def row(dopant,C,T,H_known,H,carbon_known,carbon,growth):
        if dopant not in ('Si','Sn'):raise ValueError('Trained prediction scope is Si/Sn')
        if not isinstance(H_known,(bool,np.bool_)) or not isinstance(carbon_known,(bool,np.bool_)):
            raise ValueError('Explicit impurity known/unknown boolean flags required')
        if not np.isfinite([C,T,H,carbon]).all() or C<=0 or T<=0 or min(H,carbon)<0:raise ValueError('Positive finite chemical/measurement inputs required')
        if bool(H_known)!=(H>0) or bool(carbon_known)!=(carbon>0):raise ValueError('Impurity supply/status mismatch')
        if growth is not None and not np.isfinite(growth):raise ValueError('Invalid growth metadata')
        return SimpleNamespace(element=dopant,incorporated_cm3=float(C),Tobs_K=float(T),H_known=bool(H_known),H_cm3=float(H),
            carbon_known=bool(carbon_known),carbon_cm3=float(carbon),growth_T_C=0. if growth is None else float(growth))

    def predict(self,dopant,incorporated_cm3,*,method='RelativeInventory',observation_T_K=300.,
                H_known=False,H_cm3=0.,carbon_known=False,carbon_cm3=0.,growth_T_C=None,anchors=(),pO2_atm=1e-5):
        legacy=method in self.manifest.get('energy_methods',[])
        if method not in self.manifest['variants'] and not legacy:raise ValueError('Unknown saved prediction method')
        if not np.isfinite(pO2_atm) or pO2_atm<=0:raise ValueError('Explicit positive oxygen scenario required')
        row=self.row(dopant,float(incorporated_cm3),float(observation_T_K),H_known,float(H_cm3),carbon_known,float(carbon_cm3),growth_T_C)
        anchors=list(anchors)
        if legacy:
            if row.Tobs_K!=300. or H_known or H_cm3!=0:raise ValueError('Legacy energetic presets require300K andzeroHscenario')
            for a in anchors:
                if a.get('observation_T_K',300.)!=300. or a.get('H_known',False) or a.get('H_cm3',0.)!=0.:
                    raise ValueError('Legacy energetic anchorinputs require300K/zeroH')
            if self.energy_model is None:
                parent=(self.bundle/self.manifest['energy_bundle_relative']).resolve()
                if hashlib.sha256((parent/'manifest.json').read_bytes()).hexdigest()!=self.manifest['energy_manifest_sha256']:
                    raise ValueError('Inherited energetic model manifest changed')
                self.energy_model=EnergyPredictor(self.db,parent)
            result=self.energy_model.predict(dopant,float(incorporated_cm3),method=method,anchors=anchors,pO2_atm=pO2_atm)
            result.update(version=6,branch='inherited_energy_completion',growth_T_C_metadata_only=growth_T_C)
            return result
        if supplied_energy_fingerprint(self.db)!=self.manifest['full_energy_fingerprint']:raise ValueError('Full supplied energetic library differs')
        spec=self.manifest['variants'][method];key=(spec['model_key'],float(pO2_atm))
        record=json.loads((self.bundle/'models'/f"{spec['model_key']}.json").read_text())
        if record['method']=='ridge' and growth_T_C is None:raise ValueError('Growth metadata is a required Ridge feature')
        if key not in self.models:self.models[key]=SavedModel(record,profile=self.profile,db=self.db,pO2_atm=pO2_atm)
        model=self.models[key];ar=[];observed=[]
        if len(anchors)>2:raise ValueError('Only0/1/2 distinct-dose RT sample anchors supported')
        for a in anchors:
            T=float(a.get('observation_T_K',300.))
            if T!=300.:raise ValueError('Current source priors were trained on RT specimen anchors only')
            r=self.row(dopant,float(a['incorporated_cm3']),T,a.get('H_known',H_known),float(a.get('H_cm3',H_cm3)),
                a.get('carbon_known',carbon_known),float(a.get('carbon_cm3',carbon_cm3)),a.get('growth_T_C',growth_T_C))
            if record['method']=='ridge' and a.get('growth_T_C',growth_T_C) is None:raise ValueError('Anchor growth metadata required for Ridge')
            n=float(a['measured_nH_cm3'])
            if not np.isfinite(n) or n<1e13:raise ValueError('Finite uncensored anchor Hall density above declared floor required')
            ar.append(r);observed.append(np.log10(n))
        if len({r.incorporated_cm3 for r in ar})!=len(ar):raise ValueError('Anchor doses must be distinct')
        if len(ar):
            order=np.argsort([r.incorporated_cm3 for r in ar]);ar=[ar[i] for i in order];observed=np.asarray(observed)[order]
        choice=spec['source_selection'][str(len(ar))];lam=choice['lam'];effect=raw=0.;calls=0
        cache_hit=False;calibration_calls=0
        if len(ar) and lam is not None:
            signature=(key,spec['source_kind'],choice['prior'],float(lam),tuple(
                (r.element,r.incorporated_cm3,r.Tobs_K,r.H_known,r.H_cm3,r.carbon_known,r.carbon_cm3,r.growth_T_C,float(y)) for r,y in zip(ar,observed)))
            cache_hit=signature in self.source_cache
            if not cache_hit:
                coordinate=ObservationCoordinate(lambda z:model.anchors(ar,spec['source_kind'],z),linear_shift=spec['source_kind']=='offset')
                solved=solve_effect(coordinate,np.asarray(observed),[float(lam)],kind=choice['prior'])['solutions'][0]
                if not solved['valid']:raise ValueError('No valid anchor-response solution')
                self.source_cache[signature]=(float(solved['effect']),float(solved['raw_shift']),coordinate.calls)
            effect,raw,calibration_calls=self.source_cache[signature];calls=0 if cache_hit else calibration_calls
        result=model.point(row,source_kind=spec['source_kind'],shift=raw)
        n=float(result.values[0]);mu=float(result.transport_values[0]);sigma=float(result.transport_values[1]);support=self.manifest['chemical_support_cm3'][dopant]
        warnings=[]
        if not support[0]<=row.incorporated_cm3<=support[1]:warnings.append('outside training chemical range')
        if not np.isclose(np.log10(pO2_atm),-5.,atol=1e-12,rtol=0.):warnings.append('oxygen scenario differs from fitted1e-5atm;uncalibratedprocess sensitivity')
        if carbon_known and model.physical:warnings.append('carbon suppliedasmetadata butabsentfrom energeticcatalogue;not modeled')
        if row.Tobs_K!=300.:warnings.append('temperature endpointwas notusedin newsharedtraining;Sioldcurves supplementary,Snunvalidated')
        learning=('optimizer-budget reference with named sourceadaptation' if record.get('reference_search_control') else
            'frozen original sharedclosure with named sourceadaptation' if record.get('exact_reference_fallback') else
            'learned shared relative complex/isolated inventory odds' if record['method']=='inventory' else
            'learned shared output correction onphysicalstate' if record['method']=='output' else 'direct empirical predictor;no internal defect state')
        return dict(version=6,dopant=dopant,method=method,incorporated_cm3=row.incorporated_cm3,observation_T_K=row.Tobs_K,
            nH_cm3=10**n,muH_cm2Vs=10**mu,sigma_S_cm=10**sigma,
            log10_nH_cm3=n,log10_muH_cm2Vs=mu,log10_sigma_S_cm=sigma,
            anchor_count=len(ar),first_anchor_dose_cm3=None if not ar else ar[0].incorporated_cm3,
            source_effect_dex=effect,raw_source_coordinate=raw,source_kind=spec['source_kind'],source_prior=choice,
            anchor_vector_calls=calls,calibration_anchor_vector_calls=calibration_calls,calibration_cache_hit=cache_hit,raw_Sandbox_diagnostics=result.diagnostics,
            chemical_training_range_cm3=support,oxygen_scenario_atm=float(pO2_atm),warnings=warnings,
            scope=dict(grade='conditional retrospective research prediction;not independent validation or calibrated interval',
                learned_relation=learning,
                thermal='effectivefreeze closure,not measured filmhistory',
                transport='coherentFD/Born/RTAwith205 empiricalextra' if model.physical else 'legacy empiricalCT',
                source='RTspecimenanchors only;designconditional responsepenalty;queryTmaydiffer;not microscopictruth',
                energy='all873publishedenergetic inputs supplied,nonewDFT',conductivity='derived q*nH*muH',
                rareearth='no validation;unsupported'))
