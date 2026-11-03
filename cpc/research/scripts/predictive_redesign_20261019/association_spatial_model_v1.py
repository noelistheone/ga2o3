"""Hypothesis: frozen triple-complex fraction gates spatial charge texture.

The relation is a predictive constitutive hypothesis, not a theorem that a
neutral point complex proves phase separation. All properties are recomputed
from the conserved atomic state, vector-background Poisson and Hall operator.
"""
from dataclasses import dataclass
import numpy as np
from sandbox.kroger_db import N_SITE
from sandbox.thermo import KB_EV
from neutral_association_model_v1 import NeutralAssociationModel
from spatial_inventory_model_v1 import SpatialPoint
from spatial_parent_profile_v1 import parent_profile
from spatial_local_charge_v2 import FrozenChargeKernel
from spatial_poisson_v2 import solve_poisson
from spatial_transport_v1 import local_transport,laminate_readout

LN10=float(np.log(10.))


@dataclass
class CoupledPoint(SpatialPoint):
    unassociated_values:np.ndarray
    unassociated_jacobian:np.ndarray


class AssociationSpatialModel:
    def __init__(self,*,profile,db,period_nm=1000.,coupling='triple_fraction',background='co_profiled',grid_size=128):
        if coupling not in ('triple_fraction','constant_amplitude') or background not in ('co_profiled','uniform'):
            raise ValueError('Explicit texture and background hypotheses required')
        if not np.isfinite(period_nm) or period_nm<=0 or grid_size<8 or grid_size%2:raise ValueError('Positive period and even grid>=8 required')
        self.prefix=NeutralAssociationModel(association_basis='constant',profile=profile,db=db)
        self.profile=profile;self.period_nm=float(period_nm);self.coupling=coupling;self.background=background;self.grid_size=int(grid_size)
        self.parameter_names=tuple(self.prefix.parameter_names)+('shared_maximum_texture_amplitude',)
        self.spatial_calls=0

    def point(self,row,parameters,*,source_kind='none',source_shift=0.,require_observation=True):
        w=np.asarray(parameters,float);z=float(source_shift)
        if w.shape!=(8,) or not np.isfinite(w).all() or not 0<=w[7]<=30:raise ValueError('Eight shared finite parameters;texture amplitude[0,30] required')
        if row.Tobs_K!=300.:raise ValueError('This spatial version serves fixed300K;no uncomputed temperature derivative is returned')
        prefix=self.prefix.point(row,w[:7],source_kind=source_kind,source_shift=z,require_observation=False)
        layer,solver=self.prefix.engine_for(row).layer_for(row);inventory=solver.last_inventory;cold=solver.last_prediction
        ref=inventory.totals;oldJ=inventory.jacobian
        old_inputs=[8]+([10] if row.H_known else [])+[oldJ.shape[1]-1]
        names=('log10_dose_cm3',)+(('log10_H_cm3',) if row.H_known else ())+('growth_T_C',)
        nv=9+len(names);input_col=9;source_col=8
        meanJ=np.zeros((len(ref),nv));meanJ[:,:7]=oldJ[:,:7];meanJ[:,source_col]=oldJ[:,7];meanJ[:,input_col:]=oldJ[:,old_inputs]
        dest=int(inventory.diagnostics['neutral_association_product_parent'])
        fraction=3*ref[dest]/row.incorporated_cm3
        fractionJ=3*meanJ[dest]/row.incorporated_cm3;fractionJ[input_col]-=fraction*LN10
        if fraction<0 or fraction>1+2e-10:raise ValueError('Triple-complex dopant fraction violates chemical inventory')
        if self.coupling=='triple_fraction':
            a=w[7]*fraction;aJ=w[7]*fractionJ;aJ[7]+=fraction
        else:a=w[7];aJ=np.zeros(nv);aJ[7]=1.
        foreign=np.any(solver.parent_atoms>0,axis=1)
        caps=layer.db.prefactor[layer.first]*np.where(foreign,.1*(1-1e-6),1.)
        spatial=parent_profile(ref,caps,a,grid_size=self.grid_size)
        stocks=spatial.concentrations
        stocksJ=spatial.reference_partial[:,:,None]*meanJ[:,None,:]+spatial.amplitude_partial[:,:,None]*aJ[None,None,:]
        Nd=10**w[1];background_amplitude=a if self.background=='co_profiled' else 0.
        bg=parent_profile(np.array([Nd]),np.array([N_SITE]),background_amplitude,grid_size=self.grid_size)
        NdJ=np.zeros((self.grid_size,nv));NdJ[:,1]=bg.reference_partial[0]*LN10*Nd
        if self.background=='co_profiled':NdJ+=bg.amplitude_partial[0,:,None]*aJ[None,:]
        pars=layer.parameters(T=prefix.diagnostics['Tf_K'],Nd=Nd,weights=[0.],
            doses=(row.incorporated_cm3,row.H_cm3) if row.H_known else (row.incorporated_cm3,),
            pO2=row.scenario_pO2_atm,Tobs=300.,binding_eV=.0366,screen_s=0.)
        kernel=FrozenChargeKernel(layer,pars)
        field=solve_poisson(kernel,stocks,background_cm3=bg.concentrations[0],period_nm=self.period_nm,
            initial_eV=cold.ef_eV,stock_jacobian=stocksJ,background_jacobian=NdJ,log_period_jacobian=np.zeros(nv))
        _,Ec,*_=self.profile.bands(300.)
        local=local_transport((field.fermi_eV-Ec)/(KB_EV*300.),np.log10(field.local.carriers_cm3[:,0]),
            np.log10(field.local.Nq2_cm3),300.,Nc_ref_cm3=self.profile.Nc_ref,T_ref_K=self.profile.T_ref,
            mass_ratio=self.profile.electron_mass_ratio,extra_mobility_cm2Vs=205.)
        localJ=np.einsum('nij,njk->nik',local.jacobian,field.log_state_jacobian)
        effective=laminate_readout(local.values,localJ);v=effective.values.copy();J=effective.jacobian.copy();physical=v.copy()
        if source_kind=='offset':v[[0,2]]+=z;J[[0,2],source_col]+=1.
        valid=bool(np.all(field.local.carriers_cm3[:,0]>field.local.carriers_cm3[:,1]) and v[0]>=13.)
        if require_observation and not valid:raise ValueError('Outside local-electron/Hall detection domain')
        uniform=np.r_[prefix.values[0],prefix.transport_values].copy()
        if source_kind=='offset':uniform[[0,2]]-=z
        uniformJ=np.zeros((3,8));uniformJ[:,:7]=np.r_[prefix.jacobian[[0]],prefix.transport_jacobian]
        unassociatedJ=np.zeros((3,8));unassociatedJ[:,:7]=prefix.unassociated_jacobian
        logn=np.log(field.local.carriers_cm3[:,0]);grad=(np.roll(logn,-1)-np.roll(logn,1))/(2*self.period_nm/self.grid_size)
        bad=np.asarray(local.diagnostics['local_moment1_fraction_kell_below1'])
        ell=np.asarray(local.diagnostics['moment1_mean_free_path_nm'])
        self.spatial_calls+=1
        dg=dict(prefix.diagnostics,coupling=self.coupling,background_spatial_hypothesis=self.background,
            triple_complex_dopant_fraction=float(fraction),actual_texture_amplitude=float(a),maximum_texture_amplitude=float(w[7]),
            fixed_period_nm=self.period_nm,period_measured=False,grid_size=self.grid_size,
            spatial_inventory=spatial.diagnostics,background_inventory=bg.diagnostics,spatial_poisson=field.diagnostics,
            local_transport=local.diagnostics,effective_Hall=effective.diagnostics,
            local_free_n_cm3=field.local.carriers_cm3[:,0].tolist(),local_Nq2_cm3=field.local.Nq2_cm3.tolist(),
            local_fermi_eV=field.fermi_eV.tolist(),local_transport_values=local.values.tolist(),
            maximum_local_hole_electron_ratio=float(max(field.local.carriers_cm3[:,1]/field.local.carriers_cm3[:,0])),
            maximum_moment1_ell_grad_logn=float(max(ell*abs(grad))),
            conductivity_weighted_kell_bad=float(np.asarray(effective.diagnostics['conductivity_weights'])@bad),
            resistivity_weighted_kell_bad=float(np.asarray(effective.diagnostics['resistivity_weights'])@bad),
            physical_spatial_values=physical.tolist(),observation_domain_valid=valid,
            single_electron_Hall_valid=bool(np.all(field.local.carriers_cm3[:,0]>field.local.carriers_cm3[:,1])),
            same_physical_state_for_output=bool(source_kind!='offset' or z==0.),
            observation_offset_dex=z if source_kind=='offset' else 0.,numerical_observation_continuation=not valid,
            learned_texture_relation_is_hypothesis_not_proof_of_phase_separation=True,
            conditional_literature_overlay=True,physical_regime_established=False,
            fixed_temperature_and_material_derivatives_only=True)
        return CoupledPoint(v[:1],J[:1,:8],J[:1,source_col],v[1:],J[1:,:8],J[1:,source_col],
            names,J[:1,input_col:],J[1:,input_col:],uniform,uniformJ,dg,prefix.unassociated_values,unassociatedJ)

    def counters(self):return dict(self.prefix.counters(),spatial_model_calls=self.spatial_calls)
