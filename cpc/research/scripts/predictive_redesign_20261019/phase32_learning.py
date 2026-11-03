"""Matched response-prior controls for the phase31 spectral hypothesis."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import phase31_learning as parent
from sandbox.state_rate_inventory import StateRateInventoryModel
from sandbox.state_rate_saved import FEATURES
from sandbox.coherent_fermi import published_coherent_profile

OUT=c.BASE/'phase32_rate_controls'
METHODS=FEATURES.copy();MODELS={}
configs=parent.configs
subset=parent.subset
reference_model=parent.reference_model


def model(method):
    if method not in MODELS:
        MODELS[method]=StateRateInventoryModel(feature=METHODS[method],profile=published_coherent_profile())
    return MODELS[method]


def bounds(method):
    lo=np.array([-6.,14.,-12.,-12.,-12.,-12.,-8.])
    hi=np.array([6.,18.,12.,12.,12.,12.,6.])
    return (lo,hi) if METHODS[method]=='global' else (np.r_[lo,-2.],np.r_[hi,2.])


def forward(method,w,rows,*,penalty):
    values=[];jac=[];prior_a=[];prior_aJ=[];prior_mu=[];prior_muJ=[];valid=[]
    cache={};calls=0;reference_calls=0;mapper=model(method)
    for row in rows.itertuples():
        key=c.learning.input_key(row,'process')
        if key not in cache:
            p=mapper.point(row,w,require_observation=False);calls+=1
            v=np.r_[p.values[0],p.transport_values]
            J=np.r_[p.jacobian[[0]],p.transport_jacobian]
            da=0.;dJa=np.zeros(len(w));dm=0.;dJm=np.zeros(len(w))
            if penalty:
                ref=reference_model().point(row,np.asarray(w)[:2],require_observation=False)
                reference_calls+=1;da=v[0]-ref.values[0];dJa=J[0].copy();dJa[:2]-=ref.jacobian[0]
                dm=v[1]-p.uncorrelated_values[1];dJm=J[1]-p.uncorrelated_jacobian[1]
            cache[key]=(v,J,da,dJa,dm,dJm,bool(p.diagnostics['observation_domain_valid']))
        v,J,da,dJa,dm,dJm,ok=cache[key]
        values.append(v);jac.append(J);prior_a.append(da);prior_aJ.append(dJa)
        prior_mu.append(dm);prior_muJ.append(dJm);valid.append(ok)
    return dict(values=np.asarray(values),jacobian=np.asarray(jac),
        prior_nH=np.asarray(prior_a),prior_nH_jacobian=np.asarray(prior_aJ),
        prior_mu=np.asarray(prior_mu),prior_mu_jacobian=np.asarray(prior_muJ),
        valid=np.asarray(valid,bool),actual_state_rate_model_calls=calls,
        actual_reference_model_calls=reference_calls,unique_inputs=len(cache))


def loss_arrays(method,w,rows,config,*,calculated=None):
    q=forward(method,w,rows,penalty=config['alpha']>0) if calculated is None else calculated
    # Exactly the same observable residuals, weights and response priors.
    return parent.loss_arrays(method,w,rows,config,calculated=q)


def fit_path(train,method,config):
    key=c.digest(dict(protocol=c.e.sha(OUT/'training_protocol.json'),
                      train=sorted(train),method=method,config=config))
    return key,OUT/'fits'/(key+'.json')


def initializer(train,method,config):
    fitted,path=c.fit(train,'ShareProcess205',config)
    assert fitted['training_valid'] and fitted['selected_success']
    stock=np.asarray(fitted['parameters'],float)
    suffixes=([-3.],[0.]) if METHODS[method]=='global' else ([-3.,0.],[0.,1.])
    return [np.r_[stock,v] for v in suffixes],dict(path=str(path.relative_to(c.ROOT)),
        sha256=c.e.sha(path),fit_key=fitted['key'],same_training_groups=True,
        reused_matched_baseline_optimization_cost_not_claimed_zero=True)
