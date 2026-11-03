"""Current-cohort nonlinear direct controls with the same allowed feature map."""
from pathlib import Path
from itertools import product
import sys,time,numpy as np
from sklearn.ensemble import ExtraTreesRegressor
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from strong_tree_saved import METHODS,raw_values
from sandbox.context_saved_model import feature_vector
from sandbox.activation_transport_control import ct_log_readout
from sandbox.ionized_transport import E_CHARGE

OUT=c.BASE/'strong_tree_controls_v1'
N_ESTIMATORS=64
SEED=1729


def configs():
    return [dict(max_depth=d,min_samples_leaf=m) for d,m in product((2,4,None),(1,2,4))]


def fit_path(train,method,config):
    key=c.digest(dict(train=sorted(train),method=method,config=config,study='strong_tree_controls_v1'))
    return key,OUT/'fits'/f'{key}.json'


def minimal_fit(fit):
    base=c.design.minimal_fit(fit)
    if fit['method'] in METHODS:
        base.update({k:fit[k] for k in ['center','scale','nH_forest','rho_forest']})
    return base


def serialize(model):
    return [dict(left=t.tree_.children_left.tolist(),right=t.tree_.children_right.tolist(),
        feature=t.tree_.feature.tolist(),threshold=t.tree_.threshold.tolist(),
        value=t.tree_.value[:,0,0].tolist()) for t in model.estimators_]


def fit(task):
    train,method,config=task;key,path=fit_path(train,method,config)
    assert not path.exists(),'Preserve original fit'
    table=c.cases();table=table[table.related_cluster.isin(train)]
    start=time.perf_counter();X=np.array([feature_vector(r) for r in table.itertuples()])
    wn=c.learning.weights(table);mask,rho=c.learning.observed_pair(table)
    center=np.average(X,axis=0,weights=wn)
    scale=np.sqrt(np.average((X-center)**2,axis=0,weights=wn));scale[scale<1e-12]=1.
    X=(X-center)/scale
    y=table.target.to_numpy(float);base=np.log10(table.incorporated_cm3.to_numpy(float))
    residual=method=='DoseResidualExtraTrees'
    nmodel=ExtraTreesRegressor(n_estimators=N_ESTIMATORS,random_state=SEED,n_jobs=1,**config)
    nmodel.fit(X,y-base if residual else y,sample_weight=wn)
    nH=nmodel.predict(X)+(base if residual else 0.)
    rmodel=None
    if mask.any():
        rmodel=ExtraTreesRegressor(n_estimators=N_ESTIMATORS,random_state=SEED+1,n_jobs=1,**config)
        rbase=np.array([-(np.log10(E_CHARGE)+v+ct_log_readout(v,300.)[0]) for v in base])
        w=c.learning.weights(table,paired_only=True)[mask]
        rmodel.fit(X[mask],rho[mask]-rbase[mask] if residual else rho[mask],sample_weight=w)
        rho_pred=rmodel.predict(X)+(rbase if residual else 0.)
        sigma=-rho_pred;mu=sigma-nH-np.log10(E_CHARGE)
    else:
        mu=np.array([ct_log_readout(v,300.)[0] for v in nH]);sigma=np.log10(E_CHARGE)+nH+mu
    values=np.c_[nH,mu,sigma]
    record=dict(key=key,method=method,config=config,train_units=sorted(train),
        train_families=sorted(table.family.unique()),training_condition_ids=sorted(table.sample_id.unique()),
        training_case_ids=table.case_id.tolist(),reported_conditions=table.sample_id.nunique(),
        scenario_rows=len(table),paired_conditions=table.loc[mask,'sample_id'].nunique(),
        center=center.tolist(),scale=scale.tolist(),nH_forest=serialize(nmodel),
        rho_forest=serialize(rmodel) if rmodel is not None else None,
        training_nH_weights=wn.tolist(),
        training_rho_weights=c.learning.weights(table,paired_only=True).tolist(),
        selected_success=True,training_valid=bool(np.isfinite(values).all() and np.all(nH>=13.)),
        selected_training_values=values.tolist(),transport_head='native',transport_head_offset_dex=0.,
        scenario_rows_not_independent_samples=True,loss_observables=['log10_nH','primitive_log10_rho'],
        growth_rate_not_available_in_current_input_contract=True,n_estimators=N_ESTIMATORS,
        random_seed=SEED,rho_seed=SEED+1,sklearn_internal_float32_prediction_inputs=True,
        no_physical_inventory_or_new_method_claim=True,goal_complete=False)
    replay=np.array([raw_values(record,r) for r in table.itertuples()])
    error=float(np.max(abs(replay-values)))
    record.update(serialized_training_replay_max_error=error,seconds=time.perf_counter()-start,
                  actual_forest_fits=1+int(rmodel is not None),actual_trees=N_ESTIMATORS*(1+int(rmodel is not None)))
    c.write(path,record)
    assert error<1e-10,'Preserve serialization failure'
    print('treefit',key,method,'groups',len(train),'sec',round(record['seconds'],3),flush=True)
    return dict(key=key,training_valid=record['training_valid'],seconds=record['seconds'],
                serialization_error=error,forests=record['actual_forest_fits'])
