"""New joint-model registry, normalized source/scenario weights and exact loss."""
from pathlib import Path
from types import SimpleNamespace
from itertools import combinations
import sys,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.pool_share_controls import PoolReferenceModel,DirectActivationModel
from sandbox.charge_feasible_model_v2 import ChargeFeasibleModel
from sandbox.joint_inventory_scattering import JointInventoryScatteringModel,BASE_LOG_MU0
from sandbox.reference_scattering import ReferenceScatteringModel
from sandbox.activation_transport_control import ActivationTransportControl,ct_log_readout
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.context_saved_model import feature_vector,INPUT_NAMES
from sandbox.ionized_transport import E_CHARGE
import phase29_data as data

OUT=data.OUT
GAMMA=(0.,.1,1.)
ALPHA=(0.,.1,1.)
RIDGE_ALPHA=(0.,.01,.1,1.,10.,100.,1e3,1e4,1e5)
SPECS={'Reference205':dict(kind='reference',basis='none',learn_scale=False),
       'ReferenceTau':dict(kind='reference',basis='none',learn_scale=True)}
for basis in ('constant','dose','process'):
    SPECS['Share'+basis.title()+'205']=dict(kind='share',basis=basis,learn_scale=False)
    SPECS['Joint'+basis.title()]=dict(kind='share',basis=basis,learn_scale=True)
SPECS['ActivationProcessPair']=dict(kind='activation',basis='process',learn_scale=True)
NONLINEAR=tuple(SPECS)
EMPIRICAL=('UnitDose','MedianActivation','DoseLine')
METHODS=NONLINEAR+('DirectMultiRidge',)+EMPIRICAL
MODELS={}


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()[:24]
def units():return tuple(sorted(data.load_conditions().related_cluster.unique()))
def contexts():return [tuple(c) for n in (3,4,5) for c in combinations(units(),n)]
def configs(method):
    if method=='Reference205':return [dict(gamma=g,alpha=0.) for g in GAMMA]
    if method in NONLINEAR:return [dict(gamma=g,alpha=a) for g in GAMMA for a in ALPHA]
    if method=='DirectMultiRidge':return [dict(ridge_alpha=a) for a in RIDGE_ALPHA]
    return [{}]


def model(method):
    if method not in MODELS:
        s=SPECS[method];p=published_coherent_profile()
        if s['kind']=='reference':
            obj=ReferenceScatteringModel(profile=p) if s['learn_scale'] else PoolReferenceModel(temperature_mode='effective',profile=p)
        elif s['kind']=='share':
            obj=JointInventoryScatteringModel(basis=s['basis'],profile=p) if s['learn_scale'] else ChargeFeasibleModel(basis=s['basis'],profile=p)
        else:obj=ActivationTransportControl(basis=s['basis'])
        MODELS[method]=obj
    return MODELS[method]


def observed_pair(table):
    paired=table.mobility_eligible.eq(True).to_numpy(bool)
    rho=np.full(len(table),np.nan)
    rho[paired]=-np.log10(E_CHARGE)-table.target.to_numpy(float)[paired]-np.log10(table.mu_cm2Vs.to_numpy(float)[paired])
    assert np.isfinite(rho[paired]).all()
    return paired,rho


def weights(table,*,paired_only=False):
    mask=table.mobility_eligible.eq(True) if paired_only else np.ones(len(table),bool)
    sub=table.loc[mask]
    result=np.zeros(len(table))
    if sub.empty:return result
    nunit=sub.related_cluster.nunique()
    families=sub.groupby('related_cluster').family.transform('nunique').to_numpy(float)
    samples=sub.groupby('family').sample_id.transform('nunique').to_numpy(float)
    values=sub.scenario_weight.to_numpy(float)/(nunit*families*samples)
    result[np.asarray(mask,bool)]=values
    assert abs(result.sum()-1.)<1e-12
    return result


def point_tuple(p):
    return np.r_[p.values[0],p.transport_values],np.r_[p.jacobian[[0]],p.transport_jacobian]


def input_key(row,basis):
    return (row.element,float(row.incorporated_cm3),bool(row.H_known),float(row.H_cm3),
        float(row.Tobs_K),float(row.scenario_pO2_atm),float(row.growth_T_C) if basis=='process' else 0.)


def forward(method,w,table,*,penalty=False):
    w=np.asarray(w,float);spec=SPECS[method];mapper=model(method)
    values=[];jac=[];delta=[];delta_jac=[];valid=[];cache={};physical_calls=0
    for row in table.itertuples():
        key=input_key(row,spec['basis'])
        if key not in cache:
            p=mapper.point(row,w,require_observation=False);v,J=point_tuple(p)
            physical_calls+=int(spec['kind']!='activation')
            change=0.;dchange=np.zeros(len(w))
            if penalty and spec['kind']=='share':
                refmethod='ReferenceTau' if spec['learn_scale'] else 'Reference205'
                rw=np.r_[w[:2],w[-1]] if spec['learn_scale'] else w[:2]
                q=model(refmethod).point(row,rw,require_observation=False);physical_calls+=1
                change=v[0]-q.values[0];dchange=J[0].copy()
                dchange[:2]-=q.jacobian[0,:2]
                if spec['learn_scale']:dchange[-1]-=q.jacobian[0,-1]
            elif penalty and spec['kind']=='activation':
                ref=DirectActivationModel(basis='none').point(row,w[:2],require_observation=False)
                change=v[0]-ref.values[0];dchange=J[0].copy();dchange[:2]-=ref.jacobian[0]
            cache[key]=(v,J,change,dchange,bool(p.diagnostics['observation_domain_valid']))
        v,J,change,dchange,ok=cache[key]
        values.append(v);jac.append(J);delta.append(change);delta_jac.append(dchange);valid.append(ok)
    return dict(values=np.asarray(values),jacobian=np.asarray(jac),delta=np.asarray(delta),
        delta_jacobian=np.asarray(delta_jac),valid=np.asarray(valid,bool),physical_point_calls=physical_calls,
        unique_forward_inputs=len(cache))


def loss_arrays(method,w,table,config,*,calculated=None):
    gamma=float(config['gamma']);alpha=float(config['alpha'])
    q=forward(method,w,table,penalty=alpha>0) if calculated is None else calculated
    values=q['values'];J=q['jacobian'];wn=weights(table);wp=weights(table,paired_only=True)
    paired,rho=observed_pair(table);target=table.target.to_numpy(float)
    residual=[np.sqrt(wn)*(values[:,0]-target)]
    derivative=[np.sqrt(wn)[:,None]*J[:,0]]
    if gamma>0 and paired.any():
        # The first nH direction is already counted exactly once above.
        # This second direction is the primitive log-rho residual, not an
        # additional independent nH and not a third conductivity label.
        scale=np.sqrt(gamma*wp[paired])
        residual.append(scale*(-values[paired,2]-rho[paired]))
        derivative.append(-scale[:,None]*J[paired,2])
    if alpha>0:
        residual.append(np.sqrt(alpha*wn)*q['delta'])
        derivative.append(np.sqrt(alpha*wn)[:,None]*q['delta_jacobian'])
        if SPECS[method]['learn_scale']:
            residual.append(np.array([np.sqrt(alpha)*w[-1]]))
            tail=np.zeros((1,len(w)));tail[0,-1]=np.sqrt(alpha);derivative.append(tail)
    return np.concatenate(residual),np.concatenate(derivative,axis=0),q


def bounds(method):
    s=SPECS[method];n=len(model(method).parameter_names)
    if s['kind']=='activation':
        return np.r_[12.,-12.,np.repeat(-12.,n-3),-6.],np.r_[21.,12.,np.repeat(12.,n-3),6.]
    lo=np.r_[-6.,14.,np.repeat(-12.,n-2)]
    hi=np.r_[6.,18.,np.repeat(12.,n-2)]
    if s['learn_scale']:lo[-1]=-6.-BASE_LOG_MU0;hi[-1]=6.-BASE_LOG_MU0
    return lo,hi


def weighted_median(values,weight):
    order=np.argsort(values,kind='stable');v=np.asarray(values)[order];w=np.asarray(weight)[order]
    return float(v[np.searchsorted(np.cumsum(w),w.sum()/2.,side='left')])


def direct_features(table):return np.array([feature_vector(r) for r in table.itertuples()])


def fit_direct(table,alpha):
    X=direct_features(table);wn=weights(table);wp=weights(table,paired_only=True)
    mask,rho=observed_pair(table);center=np.average(X,axis=0,weights=wn)
    scale=np.sqrt(np.average((X-center)**2,axis=0,weights=wn));scale[scale<1e-12]=1.
    A=np.c_[np.ones(len(table)),(X-center)/scale]
    reg=np.diag(np.r_[0.,np.repeat(np.sqrt(alpha),X.shape[1])])
    def solve(y,weight,nconditions):
        sw=np.sqrt(weight*nconditions)
        return np.linalg.lstsq(np.r_[A*sw[:,None],reg],np.r_[y*sw,np.zeros(A.shape[1])],rcond=None)[0]
    a=solve(table.target.to_numpy(float),wn,table.sample_id.nunique())
    if mask.any():
        target=np.where(mask,rho,0.)
        b=solve(target,wp,table.loc[mask,'sample_id'].nunique())
    else:b=None
    return dict(center=center.tolist(),scale=scale.tolist(),nH_coefficients=a.tolist(),
        rho_coefficients=b.tolist() if b is not None else None,new_linear_solves=2 if b is not None else 1,
        two_heads_use_two_primitive_observables=True,not_independent_sigma_supervision=True)


def direct_values(fit,table):
    X=np.c_[np.ones(len(table)),(direct_features(table)-np.asarray(fit['center']))/np.asarray(fit['scale'])]
    n=X@np.asarray(fit['nH_coefficients'])
    if fit['rho_coefficients'] is None:
        mu=np.array([ct_log_readout(g,300.)[0] for g in n]);sigma=np.log10(E_CHARGE)+n+mu
    else:
        sigma=-(X@np.asarray(fit['rho_coefficients']));mu=sigma-n-np.log10(E_CHARGE)
    return np.c_[n,mu,sigma]


def fit_empirical(method,table):
    x=np.log10(table.incorporated_cm3.to_numpy(float));y=table.target.to_numpy(float);wn=weights(table)
    coefficient=None;offset=0.
    if method=='MedianActivation':offset=weighted_median(y-x,wn)
    elif method=='DoseLine':
        A=np.c_[np.ones(len(x)),x-18.];coefficient=np.linalg.lstsq(A*np.sqrt(wn)[:,None],y*np.sqrt(wn),rcond=None)[0].tolist()
    fit=dict(coefficients=coefficient,offset_dex=offset)
    base=empirical_values(method,fit,table)
    wp=weights(table,paired_only=True);mask,rho=observed_pair(table)
    if mask.any():
        mu_error=np.log10(table.mu_cm2Vs.to_numpy(float)[mask])-base[mask,1]
        sigma_error=-rho[mask]-base[mask,2]
        weight=wp[mask]
        fit.update(mu_mean_offset=float(weight@mu_error),mu_median_offset=weighted_median(mu_error,weight),
            rho_mean_offset=float(weight@sigma_error),paired_scale_labels=table.loc[mask,'sample_id'].nunique())
    else:fit.update(mu_mean_offset=0.,mu_median_offset=0.,rho_mean_offset=0.,paired_scale_labels=0)
    return fit


def empirical_values(method,fit,table):
    x=np.log10(table.incorporated_cm3.to_numpy(float))
    n=fit['coefficients'][0]+fit['coefficients'][1]*(x-18.) if method=='DoseLine' else x+fit['offset_dex']
    mu=np.array([ct_log_readout(g,300.)[0] for g in n])
    return np.c_[n,mu,np.log10(E_CHARGE)+n+mu]
