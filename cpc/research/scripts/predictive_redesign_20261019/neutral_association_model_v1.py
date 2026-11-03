"""Shared ML association closure inside the atomic and cold-charge Sandbox."""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit
from sandbox.charge_feasible_model_v2 import ChargeFeasibleModel
from sandbox.charge_feasible_share import allocate_feasible_share
from sandbox.context_inventory_model import field_features
from sandbox.relative_inventory_model import RelativeInventoryPrediction,L0,L1
from sandbox.coherent_transport import fermi_transport
from neutral_association_inventory_v1 import associate_neutral

LN10=float(np.log(10.))


@dataclass
class AssociationPoint(RelativeInventoryPrediction):
    unassociated_values:np.ndarray
    unassociated_jacobian:np.ndarray


class NeutralAssociationModel(ChargeFeasibleModel):
    def __init__(self,*,association_basis,profile,db):
        if association_basis not in ('constant','dose'):raise ValueError('Explicit association basis required')
        super().__init__(basis='process',profile=profile,db=db)
        self.association_basis=association_basis
        self.parameter_names=tuple(self.parameter_names)+('neutral_association_intercept',)+(('neutral_association_logdose_slope',) if association_basis=='dose' else ())

    def point(self,row,parameters,*,source_kind='none',source_shift=0.,require_observation=True):
        w=np.asarray(parameters,float);z=float(source_shift);n=len(self.parameter_names)
        if w.shape!=(n,) or not np.isfinite(w).all():raise ValueError('Matching finite shared association parameters required')
        if row.element not in ('Si','Sn') or not np.isfinite([row.incorporated_cm3,row.Tobs_K,row.H_cm3,row.growth_T_C]).all():raise ValueError('Finite declared Si/Sn chemistry/process required')
        if row.incorporated_cm3<=0 or row.Tobs_K<=0 or row.H_cm3<0 or bool(row.H_known)!=(row.H_cm3>0):raise ValueError('Positive dose/temperature and explicit H scenario required')
        if source_kind not in ('none','share','offset') or not np.isfinite(z) or abs(z)>6 or (source_kind=='none' and z!=0):raise ValueError('Explicit supported source coordinate required')
        layer,solver=self.engine_for(row).layer_for(row)
        ni=3 if row.H_known else 2;nv=n+1+ni+1;source_col=n;input_col=n+1;growth_col=nv-1
        chain=np.zeros((layer.np,nv));s=float(expit(w[0]));Tf=10**(L0+(L1-L0)*s)
        chain[0,0]=(L1-L0)*s*(1-s);chain[1,1]=1.;chain[layer.idose,input_col]=1.;chain[layer.iobs,input_col+1]=1.
        if row.H_known:chain[2+layer.b+1,input_col+2]=1.
        phi=field_features(row,'process');psi=float(phi@w[2:6]);dpsi=np.zeros(nv);dpsi[2:6]=phi
        dpsi[input_col]=w[3]/2.;dpsi[growth_col]=w[4]/300.
        if source_kind=='share':psi+=LN10*z;dpsi[source_col]=LN10
        pars=layer.parameters(T=Tf,Nd=10**w[1],weights=[0.],doses=(row.incorporated_cm3,row.H_cm3) if row.H_known else (row.incorporated_cm3,),
            pO2=row.scenario_pO2_atm,Tobs=row.Tobs_K,binding_eV=.0366,screen_s=0.)
        original=allocate_feasible_share(solver,pars,chain,psi,dpsi)
        def solve(inventory):
            cold=solver.solve(pars,inventory,chain)
            tr=fermi_transport(cold.values[0],cold.values[2],row.Tobs_K,Nc_ref_cm3=self.profile.Nc_ref,
                T_ref_K=self.profile.T_ref,mass_ratio=self.profile.electron_mass_ratio,extra_mobility_cm2Vs=205.)
            J=tr.jacobian[:,:2]@cold.jacobian[[0,2]];J[:,input_col+1]+=tr.jacobian[:,2]
            return cold,tr,J
        old,oldtr,oldJ=solve(original)
        association_phi=np.array([1.]) if self.association_basis=='constant' else np.array([1.,(np.log10(row.incorporated_cm3)-18.)/2.])
        association_psi=float(association_phi@w[6:]);association_J=np.zeros(nv);association_J[6:n]=association_phi
        if self.association_basis=='dose':association_J[input_col]=w[7]/2.
        associated=associate_neutral(solver,original,association_psi,association_J)
        cold,tr,J=solve(associated);values=cold.values.copy();vJ=cold.jacobian.copy()
        values[0]=tr.values[0];vJ[0]=J[0];physical=values.copy();tv=tr.values[1:].copy();TJ=J[1:].copy()
        if source_kind=='offset':values[0]+=z;tv[1]+=z;vJ[0,source_col]+=1.;TJ[1,source_col]+=1.
        valid=bool(cold.diagnostics['single_electron_Hall_valid'] and values[0]>=13.)
        if require_observation and not valid:raise ValueError('Outside single-electron/Hall detection domain')
        dg=dict(cold.diagnostics,Tf_K=Tf,temperature_mode='effective',source_kind=source_kind,source_shift=z,
            share_coefficients=w[2:6].tolist(),association_basis=self.association_basis,association_coefficients=w[6:].tolist(),
            same_physical_state_for_output=bool(source_kind!='offset' or z==0.),
            observation_domain_valid=valid,observation_offset_dex=z if source_kind=='offset' else 0.,numerical_observation_continuation=not valid,
            unassociated_state_values=oldtr.values.tolist(),coherent_transport=tr.diagnostics,
            conditional_literature_overlay=True,no_new_charge_transition_levels_computed=True,
            association_identity_not_uniquely_inferred_from_Hall=True,
            learning_scope='shared process-to-stock plus bounded stoichiometric neutral-association extent;all properties from re-equilibrated same state')
        names=('log10_dose_cm3','log10_Tobs_K')+(('log10_H_cm3',) if row.H_known else ())+('growth_T_C',)
        return AssociationPoint(values,vJ[:,:n],vJ[:,source_col],physical,dg,tv,TJ[:,:n],TJ[:,source_col],
            vJ[:,input_col:],TJ[:,input_col:],names,oldtr.values,oldJ[:,:n])
