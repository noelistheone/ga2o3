"""Same-capacity controls for association-coupled spatial constitutive learning."""
from pathlib import Path
import sys,numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import phase31_learning as parent
import phase34_learning_v1 as overlay
from association_spatial_model_v1 import AssociationSpatialModel
from sandbox.coherent_fermi import published_coherent_profile

OUT=c.BASE/'phase35_association_spatial';GRID_SIZE=128
METHODS={
    'ClusterCoupled':dict(coupling='triple_fraction',background='co_profiled'),
    'ConstantTexture':dict(coupling='constant_amplitude',background='co_profiled'),
    'UniformBackground':dict(coupling='triple_fraction',background='uniform')}
MODELS={};configs=parent.configs;subset=parent.subset;reference_model=overlay.reference_model


def model(method):
    if method not in MODELS:MODELS[method]=AssociationSpatialModel(profile=published_coherent_profile(),
        db=overlay.database(),grid_size=GRID_SIZE,period_nm=1000.,**METHODS[method])
    return MODELS[method]


def bounds():
    return (np.array([-6.,14.,-12.,-12.,-12.,-12.,-12.,0.]),np.array([6.,18.,12.,12.,12.,12.,12.,30.]))


def initializer(train,method,config):
    f,path=c.fit(train,'ShareProcess205',config);assert f['training_valid'] and f['selected_success'];w=np.array(f['parameters'])
    return [np.r_[w,-4.,8.],np.r_[w,4.,24.]],dict(path=str(path.relative_to(c.ROOT)),sha256=c.e.sha(path),fit_key=f['key'],
        same_training_groups=True,inherited_optimization_cost_not_claimed_zero=True,geometry_and_initializers_not_selected_by_query_scores=True)


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
                # Total learned transport response vs the same upstream
                # unassociated uniform inventory, not a measured noise prior.
                dm=v[1]-p.unassociated_values[1];dJm=J[1]-p.unassociated_jacobian[1]
            cache[key]=(v,J,da,dJa,dm,dJm,p.diagnostics['observation_domain_valid'])
        v,J,da,dJa,dm,dJm,ok=cache[key]
        values.append(v);jac.append(J);aa.append(da);aJ.append(dJa);mm.append(dm);mJ.append(dJm);valid.append(ok)
    return dict(values=np.asarray(values),jacobian=np.asarray(jac),prior_nH=np.asarray(aa),prior_nH_jacobian=np.asarray(aJ),
        prior_mu=np.asarray(mm),prior_mu_jacobian=np.asarray(mJ),valid=np.asarray(valid,bool),
        actual_spatial_model_calls=calls,actual_reference_model_calls=refs,unique_inputs=len(cache))


def loss_arrays(method,w,rows,config,*,calculated=None):
    q=forward(method,w,rows,penalty=config['alpha']>0) if calculated is None else calculated
    return parent.loss_arrays(method,w,rows,config,calculated=q)
