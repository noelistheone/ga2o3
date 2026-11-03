"""Training registry and exact loss for shared charge-disorder relations."""
from pathlib import Path
import sys,hashlib,json
import numpy as np

sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from sandbox.correlated_inventory_model import CorrelatedInventoryModel
from sandbox.pool_share_controls import PoolReferenceModel
from sandbox.coherent_fermi import published_coherent_profile

OUT=c.BASE/'phase31_rate_redesign'
METHODS={'ChargeRatioSpectrum':'charge_ratio','ConstantSpectrum':'constant',
    'DoseSpectrum':'dose','ComplexSpectrum':'complex'}
GAMMA=(0.,.1,1.);ALPHA=(0.,.1,1.)
MODELS={};REFERENCE=None


def model(method):
    if method not in MODELS:
        MODELS[method]=CorrelatedInventoryModel(feature=METHODS[method],
            profile=published_coherent_profile())
    return MODELS[method]


def reference_model():
    global REFERENCE
    if REFERENCE is None:REFERENCE=PoolReferenceModel(
        temperature_mode='effective',profile=published_coherent_profile())
    return REFERENCE


def configs():return [dict(gamma=g,alpha=a) for g in GAMMA for a in ALPHA]
def table():return c.cases()
def subset(train):return table()[table().related_cluster.isin(train)].copy()


def bounds():
    return (np.array([-6.,14.,-12.,-12.,-12.,-12.,-8.,np.log10(.3)]),
        np.array([6.,18.,12.,12.,12.,12.,6.,1.]))


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
                reference_calls+=1;da=v[0]-ref.values[0];dJa=J[0].copy()
                dJa[:2]-=ref.jacobian[0]
                dm=v[1]-p.uncorrelated_values[1]
                dJm=J[1]-p.uncorrelated_jacobian[1]
            cache[key]=(v,J,da,dJa,dm,dJm,bool(p.diagnostics['observation_domain_valid']))
        v,J,da,dJa,dm,dJm,ok=cache[key]
        values.append(v);jac.append(J);prior_a.append(da);prior_aJ.append(dJa)
        prior_mu.append(dm);prior_muJ.append(dJm);valid.append(ok)
    return dict(values=np.asarray(values),jacobian=np.asarray(jac),
        prior_nH=np.asarray(prior_a),prior_nH_jacobian=np.asarray(prior_aJ),
        prior_mu=np.asarray(prior_mu),prior_mu_jacobian=np.asarray(prior_muJ),
        valid=np.asarray(valid,bool),actual_correlated_model_calls=calls,
        actual_reference_model_calls=reference_calls,unique_inputs=len(cache))


def loss_arrays(method,w,rows,config,*,calculated=None):
    gamma=float(config['gamma']);alpha=float(config['alpha'])
    q=forward(method,w,rows,penalty=alpha>0) if calculated is None else calculated
    wn=c.learning.weights(rows);wp=c.learning.weights(rows,paired_only=True)
    paired,rho=c.learning.observed_pair(rows);v=q['values'];J=q['jacobian']
    residual=[np.sqrt(wn)*(v[:,0]-rows.target.to_numpy(float))]
    derivative=[np.sqrt(wn)[:,None]*J[:,0]]
    if gamma>0 and paired.any():
        sw=np.sqrt(gamma*wp[paired])
        residual.append(sw*(-v[paired,2]-rho[paired]))
        derivative.append(-sw[:,None]*J[paired,2])
    if alpha>0:
        sw=np.sqrt(alpha*wn)
        residual.extend([sw*q['prior_nH'],sw*q['prior_mu']])
        derivative.extend([sw[:,None]*q['prior_nH_jacobian'],
            sw[:,None]*q['prior_mu_jacobian']])
    return np.concatenate(residual),np.concatenate(derivative,axis=0),q


def fit_path(train,method,config):
    key=c.digest(dict(protocol=c.e.sha(OUT/'training_protocol.json'),
        train=sorted(train),method=method,config=config))
    return key,OUT/'fits'/(key+'.json')


def initializer(train,config):
    fitted,path=c.fit(train,'ShareProcess205',config)
    assert fitted['training_valid'] and fitted['selected_success']
    stock=np.asarray(fitted['parameters'],float)
    assert stock.shape==(6,)
    return [np.r_[stock,-3.,0.],np.r_[stock,0.,.5]],dict(
        path=str(path.relative_to(c.ROOT)),sha256=c.e.sha(path),
        fit_key=fitted['key'],same_training_groups=True,
        reused_matched_baseline_optimization_cost_not_claimed_zero=True)
