"""Trainable spatial candidate registry; original data/weights remain unchanged."""
from pathlib import Path
import sys,numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import phase31_learning as parent
from spatial_inventory_model_v1 import SpatialInventoryModel
from sandbox.coherent_fermi import published_coherent_profile

OUT=c.BASE/'phase33_spatial_predictive'
METHODS={'CommonSpatial':'common','ContrastSpatial':'structural_contrast'}
MODELS={};GRID_SIZE=128
configs=parent.configs
subset=parent.subset
reference_model=parent.reference_model


def model(method):
    if method not in MODELS:MODELS[method]=SpatialInventoryModel(mode=METHODS[method],profile=published_coherent_profile(),grid_size=GRID_SIZE)
    return MODELS[method]


def bounds():
    return (np.array([-6.,14.,-12.,-12.,-12.,-12.,0.,1.]),
        np.array([6.,18.,12.,12.,12.,12.,8.,3.]))


def forward(method,w,rows,*,penalty):
    values=[];jac=[];prior_a=[];prior_aJ=[];prior_mu=[];prior_muJ=[];valid=[]
    cache={};calls=0;reference_calls=0;mapper=model(method)
    for row in rows.itertuples():
        key=c.learning.input_key(row,'process')
        if key not in cache:
            p=mapper.point(row,w,require_observation=False);calls+=1
            v=np.r_[p.values[0],p.transport_values];J=np.r_[p.jacobian[[0]],p.transport_jacobian]
            da=0.;dJa=np.zeros(len(w));dm=0.;dJm=np.zeros(len(w))
            if penalty:
                ref=reference_model().point(row,np.asarray(w)[:2],require_observation=False)
                reference_calls+=1;da=v[0]-ref.values[0];dJa=J[0].copy();dJa[:2]-=ref.jacobian[0]
                dm=v[1]-p.uniform_values[1];dJm=J[1]-p.uniform_jacobian[1]
            cache[key]=(v,J,da,dJa,dm,dJm,bool(p.diagnostics['observation_domain_valid']))
        v,J,da,dJa,dm,dJm,ok=cache[key]
        values.append(v);jac.append(J);prior_a.append(da);prior_aJ.append(dJa)
        prior_mu.append(dm);prior_muJ.append(dJm);valid.append(ok)
    return dict(values=np.asarray(values),jacobian=np.asarray(jac),prior_nH=np.asarray(prior_a),
        prior_nH_jacobian=np.asarray(prior_aJ),prior_mu=np.asarray(prior_mu),prior_mu_jacobian=np.asarray(prior_muJ),
        valid=np.asarray(valid,bool),actual_spatial_model_calls=calls,actual_reference_model_calls=reference_calls,unique_inputs=len(cache))


def loss_arrays(method,w,rows,config,*,calculated=None):
    q=forward(method,w,rows,penalty=config['alpha']>0) if calculated is None else calculated
    return parent.loss_arrays(method,w,rows,config,calculated=q)


def initializer(train,method,config):
    fitted,path=c.fit(train,'ShareProcess205',config)
    assert fitted['training_valid'] and fitted['selected_success']
    stock=np.asarray(fitted['parameters'],float)
    geometry=([.5,np.log10(20.)],[2.,np.log10(200.)]) if METHODS[method]=='common' else ([.02,np.log10(20.)],[.2,np.log10(200.)])
    return [np.r_[stock,v] for v in geometry],dict(path=str(path.relative_to(c.ROOT)),sha256=c.e.sha(path),
        fit_key=fitted['key'],same_training_groups=True,inherited_optimization_cost_not_claimed_zero=True,
        initial_geometry_from_unlabeled_charge_domain_audit_not_query_scores=True)
