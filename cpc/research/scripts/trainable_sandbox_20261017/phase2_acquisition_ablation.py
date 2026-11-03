"""Separate feature-dependent energy learning from physical acquisition.

This follow-up is declared after inspecting the initial replay. It is an
explicit development ablation, not a newly blind confirmation.
"""
from pathlib import Path
from itertools import product
import sys,os,json,pickle
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from sandbox.goal_acquisition import condition_coordinate,prediction_variance_reduction
from sandbox.energy_learning import ChargeEnergyLearner
from csv_io import read_result_csv
import phase2_energy as energy
import phase2_acquisition as acquisition

OUT=energy.OUT/"acquisition_ablation"
METHODS=("global_mean","charge_pair_mean","ridge","rbf","extra_trees")
COVS=("diagonal_A","diagonal_xi")
def run():
    energy.frozen();acquisition.freeze()
    proto=OUT/"protocol.json"
    payload=dict(version=1,created_utc=energy.now(),code_sha256=energy.sha(Path(__file__)),
        after_initial_results=True,validation="retrospective method-development ablation",
        task="determine whether acquisition gains require feature-dependent learned mean/covariance",
        means=METHODS,covariances=COVS,
        diagonal_A="independent equal variance in original unknown intercepts; C_xi=diag(1/delta_q^2)",
        diagonal_xi="independent equal variance in crossing energies; C_xi=I",
        policy="same Sandbox gradient, goal utility and exact label budget as initial replay",
        full_covariance_reference="original RBF/ExtraTrees learned covariance replay retained separately",
        design=acquisition.DESIGN,evaluation=acquisition.EVAL,budgets=list(range(9)),
        caveat="same-budget comparisons separate learned-energy contribution and physical selection; no material accuracy claim")
    if proto.exists():
        assert json.loads(proto.read_text())["code_sha256"]==payload["code_sha256"]
    else:energy.write(proto,payload)
    db,rr,_=energy.records();allrows=[]
    for e,method,ckind in product(("Si","Sn"),METHODS,COVS):
        dest=OUT/"replays"/f"{e}__{method}__{ckind}.csv"
        if dest.exists():
            allrows.extend(read_result_csv(dest).to_dict("records"));continue
        tr=[r for r in rr if r["element"]!=e];targets=[r for r in rr if r["element"]==e]
        with (energy.OUT/"models"/f"{e}__{method}.pkl").open("rb") as f:model=pickle.load(f)
        mean=model.predict(targets);truth=np.array([r["crossing_eV"] for r in targets])
        dq=np.array([r["delta_q"] for r in targets],float)
        cov=np.diag(1/dq**2) if ckind=="diagonal_A" else np.eye(len(targets))
        oracle,_,_=acquisition.state_predictions(db,targets,truth,acquisition.EVAL)
        acquired=[];means=[];states=[];rows=[]
        for budget in range(9):
            means.append(mean.copy());states.append("|".join(targets[i]["label_id"] for i in acquired))
            pred,_,valid=acquisition.state_predictions(db,targets,mean,acquisition.EVAL)
            for i,(T,dose) in enumerate(acquisition.EVAL):
                rows.append(dict(element=e,model=method,covariance=ckind,budget=budget,Tf_K=T,
                    dose_cm3=dose,predicted_log_nH=float(pred[i]),oracle_log_nH=float(oracle[i]),
                    physical_valid=bool(valid[i]),acquired_labels=states[-1]))
            if budget==8:break
            _,J,_=acquisition.state_predictions(db,targets,mean,acquisition.DESIGN,True)
            utility=prediction_variance_reduction(J,cov);utility[acquired]=-np.inf
            j=int(np.argmax(utility))
            mean,cov=condition_coordinate(mean,cov,j,truth[j]);acquired.append(j)
        energy.csvwrite(dest,rows);allrows.extend(rows)
        q=OUT/"states"/f"{e}__{method}__{ckind}.npz";q.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(q,means=np.array(means),labels=np.array(states))
        print("ablation",e,method,ckind,"budget4",
            np.mean([abs(r["predicted_log_nH"]-r["oracle_log_nH"]) for r in rows if r["budget"]==4]),flush=True)
    energy.csvwrite(OUT/"predictions.csv",allrows)
    d=pd.DataFrame(allrows)
    d["error"]=abs(d.predicted_log_nH-d.oracle_log_nH)
    summary=d.groupby(["element","model","covariance","budget"]).agg(
        MAE=("error","mean"),max_error=("error","max"),physical=("physical_valid","sum")).reset_index()
    summary.to_csv(OUT/"summary.csv",index=False)
    print(summary[summary.budget.isin([0,2,4,8])].to_string(index=False),flush=True)
if __name__=="__main__":run()
