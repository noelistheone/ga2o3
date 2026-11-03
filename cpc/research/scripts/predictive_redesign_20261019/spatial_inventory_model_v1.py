"""Shared process-to-inventory ML followed by periodic Poisson and Hall.

Eight shared parameters, no sample/family identifiers or query labels. The
temperature and material are fixed at the declared 300 K readout. Derivatives
of a different observation temperature are deliberately not offered.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit
from sandbox.charge_feasible_model_v2 import ChargeFeasibleModel
from sandbox.charge_feasible_share import allocate_feasible_share
from sandbox.context_inventory_model import field_features
from sandbox.relative_inventory_model import L0,L1
from sandbox.coherent_transport import fermi_transport
from sandbox.thermo import KB_EV
from spatial_parent_profile_v1 import parent_profile
from spatial_local_charge_v1 import FrozenChargeKernel
from spatial_poisson_v1 import solve_poisson
from spatial_transport_v1 import local_transport,laminate_readout

LN10=float(np.log(10.))


@dataclass
class SpatialPoint:
    values:np.ndarray
    jacobian:np.ndarray
    source_jacobian:np.ndarray
    transport_values:np.ndarray
    transport_jacobian:np.ndarray
    transport_source_jacobian:np.ndarray
    input_names:tuple
    input_jacobian:np.ndarray
    transport_input_jacobian:np.ndarray
    uniform_values:np.ndarray
    uniform_jacobian:np.ndarray
    diagnostics:dict


class SpatialInventoryModel(ChargeFeasibleModel):
    def __init__(self,*,mode,profile,grid_size=32,db=None):
        if mode not in ('common','structural_contrast'):raise ValueError('Declared structural spatial mode required')
        if grid_size<8 or grid_size%2:raise ValueError('Even grid>=8 required')
        super().__init__(basis='process',profile=profile,db=db)
        self.mode=mode;self.grid_size=grid_size
        self.parameter_names=tuple(self.parameter_names)+('shared_spatial_amplitude','shared_log10_period_nm')
        self.spatial_calls=0

    def point(self,row,parameters,*,source_kind='none',source_shift=0.,require_observation=True):
        w=np.asarray(parameters,float);z=float(source_shift);n=8
        if w.shape!=(n,) or not np.isfinite(w).all():raise ValueError('Eight finite shared spatial parameters required')
        if row.element not in ('Si','Sn') or not np.isfinite([row.incorporated_cm3,row.Tobs_K,row.H_cm3,row.growth_T_C]).all():
            raise ValueError('Explicit finite Si/Sn chemistry and process inputs required')
        if row.incorporated_cm3<=0 or row.Tobs_K!=300. or row.H_cm3<0 or bool(row.H_known)!=(row.H_cm3>0):
            raise ValueError('Positive dose, fixed300K and explicit Hknown/unknown-zero scenario required')
        if source_kind not in ('none','share','offset') or not np.isfinite(z) or abs(z)>6 or (source_kind=='none' and z!=0):
            raise ValueError('Supported source kind and explicit coordinate in[-6,6] required')
        if not 0<=w[6]<=40 or not -1<=w[7]<=5:raise ValueError('Outside prototype geometry numerical domain')
        layer,solver=self.engine_for(row).layer_for(row)
        names=('log10_dose_cm3',)+(('log10_H_cm3',) if row.H_known else ())+('growth_T_C',)
        nv=n+1+len(names);source_col=n;input_col=n+1;growth_col=nv-1
        chain=np.zeros((layer.np,nv));s=float(expit(w[0]))
        Tf=10**(L0+(L1-L0)*s);Nd=10**w[1]
        chain[0,0]=(L1-L0)*s*(1-s);chain[1,1]=1.;chain[layer.idose,input_col]=1.
        if row.H_known:chain[2+layer.b+1,input_col+1]=1.
        phi=field_features(row,'process');psi=float(phi@w[2:6]);dpsi=np.zeros(nv)
        dpsi[2:6]=phi;dpsi[input_col]=w[3]/2.;dpsi[growth_col]=w[4]/300.
        if source_kind=='share':psi+=LN10*z;dpsi[source_col]=LN10
        pars=layer.parameters(T=Tf,Nd=Nd,weights=[0.],
            doses=(row.incorporated_cm3,row.H_cm3) if row.H_known else (row.incorporated_cm3,),
            pO2=row.scenario_pO2_atm,Tobs=300.,binding_eV=.0366,screen_s=0.)
        inventory=allocate_feasible_share(solver,pars,chain,psi,dpsi)
        cold=solver.solve(pars,inventory,chain)
        uniform=fermi_transport(cold.values[0],cold.values[2],300.,Nc_ref_cm3=self.profile.Nc_ref,
            T_ref_K=self.profile.T_ref,mass_ratio=self.profile.electron_mass_ratio,extra_mobility_cm2Vs=205.)
        uniformJ=uniform.jacobian[:,:2]@cold.jacobian[[0,2]]
        ref=inventory.totals;refJ=inventory.jacobian
        dm=layer.db.dm[layer.first];foreign=np.any(solver.parent_atoms>0,axis=1)
        caps=layer.db.prefactor[layer.first]*np.where(foreign,.1*(1-1e-6),1.)
        roles=np.ones(len(ref)) if self.mode=='common' else np.zeros(len(ref))
        if self.mode=='structural_contrast':
            roles[solver.receivers]=1.;roles[solver.complex]=-1.;roles[solver.native&(dm[:,0]<0)]=-1.
        N=self.grid_size;stock=np.empty((len(ref),N));dc=np.empty((len(ref),N,nv));profile_diags=[]
        for role in sorted(set(roles)):
            mask=roles==role;profile=parent_profile(ref[mask],caps[mask],w[6]*role,grid_size=N)
            stock[mask]=profile.concentrations
            dc[mask]=profile.reference_partial[:,:,None]*refJ[mask,None,:]
            dc[mask,:,6]+=role*profile.amplitude_partial
            profile_diags.append(dict(role=role,**profile.diagnostics))
        kernel=FrozenChargeKernel(layer,pars);dNd=np.zeros(nv);dNd[1]=LN10*Nd
        dlogL=np.zeros(nv);dlogL[7]=1.
        field=solve_poisson(kernel,stock,background_cm3=Nd,period_nm=10**w[7],initial_eV=cold.ef_eV,
            stock_jacobian=dc,background_jacobian=dNd,log_period_jacobian=dlogL)
        _,Ec,*_=self.profile.bands(300.)
        local=local_transport((field.fermi_eV-Ec)/(KB_EV*300.),np.log10(field.local.carriers_cm3[:,0]),
            np.log10(field.local.Nq2_cm3),300.,Nc_ref_cm3=self.profile.Nc_ref,T_ref_K=self.profile.T_ref,
            mass_ratio=self.profile.electron_mass_ratio,extra_mobility_cm2Vs=205.)
        localJ=np.einsum('nij,njk->nik',local.jacobian,field.log_state_jacobian)
        out=laminate_readout(local.values,localJ);v=out.values.copy();J=out.jacobian.copy()
        physical=v.copy()
        if source_kind=='offset':v[[0,2]]+=z;J[[0,2],source_col]+=1.
        local_electron=bool(np.all(field.local.carriers_cm3[:,0]>field.local.carriers_cm3[:,1]))
        valid=bool(local_electron and v[0]>=13.)
        if require_observation and not valid:raise ValueError('Outside declared local-electron/effective-Hall detection domain')
        self.spatial_calls+=1
        # Locality diagnostics use centered periodic gradients of log n and
        # the first/second transport-moment mean free paths. No validity theorem
        # is inferred merely from a small fitted effective mobility.
        logn=np.log(field.local.carriers_cm3[:,0]);dx=10**w[7]/N
        grad=(np.roll(logn,-1)-np.roll(logn,1))/(2*dx)
        ell1=np.asarray(local.diagnostics['moment1_mean_free_path_nm']);ell2=np.asarray(local.diagnostics['moment2_mean_free_path_nm'])
        dg=dict(cold.diagnostics,source_kind=source_kind,source_shift=z,Tf_K=Tf,
            pO2_atm=float(row.scenario_pO2_atm),share_coefficients=w[2:6].tolist(),spatial_mode=self.mode,
            spatial_amplitude=float(w[6]),period_nm=float(10**w[7]),grid_size=N,
            fixed_temperature_and_material_derivatives_only=True,unsupported_temperature_derivative_not_returned=True,
            spatial_profile_diagnostics=profile_diags,spatial_poisson=field.diagnostics,
            local_transport=local.diagnostics,effective_Hall=out.diagnostics,
            local_fermi_eV=field.fermi_eV.tolist(),local_free_n_cm3=field.local.carriers_cm3[:,0].tolist(),
            local_Nq2_cm3=field.local.Nq2_cm3.tolist(),local_transport_values=local.values.tolist(),
            maximum_local_hole_electron_ratio=float(max(field.local.carriers_cm3[:,1]/field.local.carriers_cm3[:,0])),
            maximum_moment1_ell_grad_logn=float(max(ell1*abs(grad))),maximum_moment2_ell_grad_logn=float(max(ell2*abs(grad))),
            maximum_cell_kell_bad_moment1_fraction=max(local.diagnostics['local_moment1_fraction_kell_below1']),
            conductivity_weighted_kell_bad_moment1_fraction=float(np.asarray(out.diagnostics['conductivity_weights'])@np.asarray(local.diagnostics['local_moment1_fraction_kell_below1'])),
            resistivity_weighted_kell_bad_moment1_fraction=float(np.asarray(out.diagnostics['resistivity_weights'])@np.asarray(local.diagnostics['local_moment1_fraction_kell_below1'])),
            physical_spatial_values=physical.tolist(),uniform_same_inventory_values=uniform.values.tolist(),
            same_physical_state_for_output=bool(source_kind!='offset' or z==0.),
            observation_domain_valid=valid,above_carrier_detection_floor=bool(v[0]>=13.),
            single_electron_Hall_valid=local_electron,n_gt_p_does_not_establish_single_band=True,
            observation_offset_dex=z if source_kind=='offset' else 0.,numerical_observation_continuation=not valid,
            spatial_profile_measured=False,spatial_length_measured=False,material_Born_validity_established=False,
            learning_scope='shared process-to-feasible-mean-inventory relation plus shared spatial amplitude/length;physical Poisson state and same-state local/effective Hall readout',
            scope='fixed300K,fixedcoldenergies,periodiclaminateinplane,localBornRTA;not general3Dmicrostructure')
        return SpatialPoint(v[:1],J[:1,:n],J[:1,source_col],v[1:],J[1:,:n],J[1:,source_col],
            names,J[:1,input_col:],J[1:,input_col:],uniform.values,uniformJ[:,:n],dg)

    def counters(self):
        return dict(super().counters(),spatial_model_calls=self.spatial_calls)
