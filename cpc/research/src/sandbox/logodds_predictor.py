"""Saved ML + Sandbox research predictor without experimental-table access."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
from scipy.special import expit
from .logodds_inventory_model import LogOddsInventoryModel
from .coherent_fermi import CoherentMaterialProfile,TABLE
from .kroger_db import NPZ
from .bounded_source_learning import learn_bounded_source
from . import transport


class LogOddsResearchPredictor:
    """Reuse shared weights; calibrate with 0/1/2 supplied carrier anchors.

    Scope: Si/Sn conditional inputs, historical research validation. Calibrated
    coordinates are effective quantities, not measured microscopic inventories.
    """
    def __init__(self,bundle):
        self.bundle=dict(bundle);self.definition=bundle['definition'];self.parameters=np.asarray(bundle['parameters'],float)
        for path,key in ((NPZ,'energetics_sha256'),(TABLE,'fd_reference_sha256')):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=bundle[key]:raise ValueError('Physical reference hash mismatch')
        self.physical='empirical' not in self.definition
        self.model=LogOddsInventoryModel(self.definition,profile=CoherentMaterialProfile.from_dict(bundle['profile']),pO2_atm=bundle['pO2_atm']) if self.physical else None

    def shared_log_n(self,inputs,logodds=0.):
        if self.physical:return np.array([self.model.point(SimpleNamespace(**r),self.parameters,
            source_kind='logodds' if logodds!=0 else 'none',source_shift=logodds).values[0] for r in inputs])
        C=np.array([r['incorporated_cm3'] for r in inputs]);H=np.array([r['H_cm3'] for r in inputs]);w=self.parameters
        g=w[-2]+w[-1]*(np.log10(C)-18.)+np.log(10.)*logodds;lf=-np.logaddexp(0.,-g)
        if self.definition['empirical']=='activation_product':return np.log10(C+H+10**w[0])+lf/np.log(10.)
        return np.logaddexp(w[0]*np.log(10.),np.log(C+H)+lf)/np.log(10.)

    def calibrate(self,anchors,kind,lam):
        if kind not in ('offset','logodds'):raise ValueError('Explicit supported source adaptation required')
        if len(anchors)>2:raise ValueError('Research protocol supports up to two specimen anchors')
        if self.physical and kind=='logodds' and self.definition['closure']!='inventory':raise ValueError('No internal log-odds coordinate in output baseline')
        if not anchors or np.isinf(lam):return dict(kind=kind,shift=0.,anchor_count=len(anchors),search=None)
        inputs=[a['input'] for a in anchors];y=np.array([a['observed_log_nH'] for a in anchors],float)
        if not np.isfinite(y).all() or lam<0:raise ValueError('Finite supplied carrier labels/nonnegative penalty required')
        if kind=='offset':return dict(kind=kind,shift=float(np.sum(y-self.shared_log_n(inputs))/(len(y)+lam)),anchor_count=len(y),search=None)
        fit=learn_bounded_source(lambda z:self.shared_log_n(inputs,z),y,[lam]);return dict(kind=kind,shift=fit['solutions'][0]['z'],anchor_count=len(y),search=fit)

    def predict(self,inputs,calibration):
        kind=calibration['kind'];z=float(calibration['shift']);row=SimpleNamespace(**inputs)
        if self.physical:
            a=self.model.point(row,self.parameters,source_kind='logodds' if kind=='logodds' else 'offset',source_shift=z if kind=='logodds' else -z)
            g=a.diagnostics;values=a.values.tolist();tv=a.transport_values.tolist()
            dg={k:g[k] for k in ('same_physical_state_for_output','free_n_cm3','Tf_K','hall_factor','foreign_atom_relative_error',
                'complex_reallocation_fraction','host_Ga_O_inventory_change_cm3','allocation_logit','thermal_cap_K')}
            layer,solver=self.model.layer_for(row);p=solver.last_prediction;stock=solver.last_inventory.totals
            mq=np.bincount(layer.parent,weights=p.probabilities*layer.q,minlength=layer.db.nD)
            chi=np.bincount(layer.parent,weights=p.probabilities*(layer.q-mq[layer.parent])**2,minlength=layer.db.nD)
            ell=g['localized_fraction']
            for plus,neutral in layer.pairs:
                f=p.probabilities[neutral]/ell if ell>0 else 0.;chi[layer.parent[plus]]=ell*f*(1-f)
            screen=g['coherent_transport']['n_screen_cm3']
            dg.update(bound_equilibrium_response_over_free_screen=float(stock@chi/screen),
                max_native_parent_fraction=float(max(stock[solver.native]/layer.db.prefactor[layer.first[solver.native]])),
                material_Born_validity_established=False)
        else:
            logn=float(self.shared_log_n([inputs],z if kind=='logodds' else 0.)[0]+(z if kind=='offset' else 0.))
            T=float(row.Tobs_K);ceiling=transport.CT_MU_MAX*float(transport.mu_pop_hall(T)/transport.mu_pop_hall(300.))
            mu=transport.CT_MU_MIN+(ceiling-transport.CT_MU_MIN)*float(expit(-transport.CT_ALPHA*np.log(10.)*(logn-np.log10(transport.CT_N_REF))))
            logmu=float(np.log10(mu));values=[logn,None,None];tv=[logmu,float(np.log10(transport.E)+logn+logmu)]
            dg=dict(same_physical_state_for_output=False,empirical_CT_uses_calibrated_predicted_nH=True,
                outside_CT_declared_temperature_calibration=not 300<=T<=500)
        if not np.isfinite([values[0],*tv]).all():raise ValueError('Nonfinite property output')
        return dict(values=values,transport_values=tv,diagnostics=dg)
