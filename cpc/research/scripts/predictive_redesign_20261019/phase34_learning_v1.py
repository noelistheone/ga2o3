"""Matched same-overlay controls for frozen neutral-association learning."""
from pathlib import Path
from dataclasses import replace
import sys,numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import phase31_learning as parent
from si_multidonor_overlay_v1 import make_overlay
from neutral_association_model_v1 import NeutralAssociationModel
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.charge_feasible_model_v2 import ChargeFeasibleModel
from sandbox.relative_inventory_model import RelativeInventoryModel

OUT=c.BASE/'phase34_neutral_association';GRID_SIZE=None
METHODS={'OverlayReference':'fixedreference','OverlayShareProcess':'noassociation',
    'ConstantAssociation':'constant','DoseAssociation':'dose'}
MODELS={};REFERENCE=None;DB=None
configs=parent.configs
subset=parent.subset


def database():
    global DB
    if DB is None:DB=make_overlay()[0]
    return DB


class OverlayReference:
    def __init__(self):
        self.engines={};self.parameter_names=('logit_Tf','log10_background')
    def point(self,row,parameters,*,source_kind='none',source_shift=0.,require_observation=True):
        if len(parameters)!=2:raise ValueError('Two reference coordinates required')
        pO2=float(row.scenario_pO2_atm)
        if pO2 not in self.engines:self.engines[pO2]=RelativeInventoryModel(dict(field='inventory'),
            profile=published_coherent_profile(),pO2_atm=pO2,db=database())
        p=self.engines[pO2].point(row,np.r_[parameters,0.],source_kind='field' if source_kind=='share' else source_kind,
            source_shift=source_shift,require_observation=require_observation)
        return replace(p,jacobian=p.jacobian[:,:2],transport_jacobian=p.transport_jacobian[:,:2])
    def counters(self):
        values=[e.counters() for e in self.engines.values()]
        return {k:sum(v[k] for v in values) for k in ['hot_solve_calls','hot_cache_hits','cold_solve_calls']}


def model(method):
    if method not in MODELS:
        kind=METHODS[method]
        if kind=='fixedreference':MODELS[method]=OverlayReference()
        elif kind=='noassociation':MODELS[method]=ChargeFeasibleModel(basis='process',profile=published_coherent_profile(),db=database())
        else:MODELS[method]=NeutralAssociationModel(association_basis=kind,profile=published_coherent_profile(),db=database())
    return MODELS[method]


def reference_model():
    global REFERENCE
    if REFERENCE is None:REFERENCE=OverlayReference()
    return REFERENCE


def bounds(method):
    n=len(model(method).parameter_names)
    lo=np.array([-6.,14.,-12.,-12.,-12.,-12.,-12.,-12.]);hi=np.array([6.,18.,12.,12.,12.,12.,12.,12.])
    return lo[:n],hi[:n]


def initializer(train,method,config):
    oldmethod='Reference205' if method=='OverlayReference' else 'ShareProcess205'
    warmconfig=dict(config,alpha=0.) if oldmethod=='Reference205' else config
    fitted,path=c.fit(train,oldmethod,warmconfig);assert fitted['training_valid'] and fitted['selected_success']
    w=np.array(fitted['parameters']);kind=METHODS[method]
    if kind=='constant':initial=[np.r_[w,-4.],np.r_[w,4.]]
    elif kind=='dose':initial=[np.r_[w,-4.,0.],np.r_[w,4.,1.]]
    elif kind=='fixedreference':initial=[w,np.array([0.,14.5])]
    else:initial=[w,w+np.array([-.2,0.,.1,0.,0.,0.])]
    return initial,dict(path=str(path.relative_to(c.ROOT)),sha256=c.e.sha(path),fit_key=fitted['key'],
        actual_initializer_config=warmconfig,same_training_groups=True,inherited_optimization_cost_not_claimed_zero=True)


def forward(method,w,rows,*,penalty):
    values=[];jac=[];aa=[];aJ=[];mm=[];mJ=[];valid=[];calls=0;refs=0;cache={}
    for row in rows.itertuples():
        key=c.learning.input_key(row,'process')
        if key not in cache:
            p=model(method).point(row,w,require_observation=False);calls+=1
            v=np.r_[p.values[0],p.transport_values];J=np.r_[p.jacobian[[0]],p.transport_jacobian]
            da=0.;dJa=np.zeros(len(w));dm=0.;dJm=np.zeros(len(w))
            if penalty:
                ref=reference_model().point(row,np.array(w)[:2],require_observation=False);refs+=1
                da=v[0]-ref.values[0];dJa=J[0].copy();dJa[:2]-=ref.jacobian[0]
                if hasattr(p,'unassociated_values'):dm=v[1]-p.unassociated_values[1];dJm=J[1]-p.unassociated_jacobian[1]
            cache[key]=(v,J,da,dJa,dm,dJm,p.diagnostics['observation_domain_valid'])
        v,J,da,dJa,dm,dJm,ok=cache[key]
        values.append(v);jac.append(J);aa.append(da);aJ.append(dJa);mm.append(dm);mJ.append(dJm);valid.append(ok)
    return dict(values=np.asarray(values),jacobian=np.asarray(jac),prior_nH=np.asarray(aa),prior_nH_jacobian=np.asarray(aJ),
        prior_mu=np.asarray(mm),prior_mu_jacobian=np.asarray(mJ),valid=np.asarray(valid,bool),
        actual_spatial_model_calls=calls,actual_reference_model_calls=refs,
        count_key_is_generic_model_calls_not_spatial_PDE=True,unique_inputs=len(cache))


def loss_arrays(method,w,rows,config,*,calculated=None):
    q=forward(method,w,rows,penalty=config['alpha']>0) if calculated is None else calculated
    return parent.loss_arrays(method,w,rows,config,calculated=q)
