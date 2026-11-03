"""Target sensitivities for choosing the next internal learning object.

Basis classes use database stoichiometry and parent charge support, not target
labels. Outer errors are diagnostic development evidence, not a fresh test.
"""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from sandbox.kroger_db import load
from sandbox.implicit_defects import ImplicitDefectLayer,energy_basis
from scipy.special import expit
OUT=ROOT/"results/trainable_sandbox_20261017"

def bases(db):
    count=db.dm[:,2:].sum(axis=1)
    iscomplex=(count>0)&(np.abs(db.dm[:,:2]).sum(axis=1)>count)
    ids=np.unique(db.cs_ID[iscomplex])
    acceptor=[i for i in ids if db.charge[db.cs_ID==i].min()<0]
    nonacceptor=[i for i in ids if db.charge[db.cs_ID==i].min()>=0]
    B=np.c_[energy_basis(db)[:,0],np.isin(db.cs_ID,acceptor),np.isin(db.cs_ID,nonacceptor)].astype(float)
    return B,["native_Ga_vacancy","compensating_dopant_complex","nonacceptor_dopant_complex"]

def main():
    db=load();B,names=bases(db)
    layers={e:ImplicitDefectLayer(db,(e,),B) for e in ["Si","Sn"]}
    d=pd.read_csv(OUT/"dataset_v1.csv")
    p=pd.read_csv(OUT/"predictions_v1.csv")
    records=[];ranks=[];checks=[]
    for family in sorted(d.family.unique()):
        keys=p[(p.family==family)&p.budget.eq(0)&p.method.eq("physics_global_process_offset")].fit_key.unique()
        assert len(keys)==1
        fit=json.loads((OUT/"fits"/(keys[0]+".json")).read_text())
        w=fit["parameters"];logT=np.log10(600.)+(np.log10(1950.)-np.log10(600.))*expit(w[0])
        block=[]
        for row in d.itertuples():
            layer=layers[row.element]
            pars=layer.parameters(T=10.**logT,Nd=10.**w[1],weights=[0.,0.,0.],doses=(row.incorporated_cm3,))
            r=layer.predict(pars)
            obs=dict(outer_family=family,sample_id=row.sample_id,family=row.family,
                use="outer_diagnostic" if row.family==family else "training",
                log10_Tf=logT,predicted_log_nH=r.values[0],target_log_nH=row.observed_log_nH,
                d_nH_d_logT=r.jacobian[0,0],d_nH_d_logNd=r.jacobian[0,1])
            for j,name in enumerate(names):obs["d_nH_d_"+name]=r.jacobian[0,2+j]
            records.append(obs);block.append(obs)
        tr=pd.DataFrame(block);tr=tr[tr.use.eq("training")]
        weights=1/tr.family.map(tr.family.value_counts()).to_numpy(float)
        J=tr[["d_nH_d_"+name for name in names]].to_numpy()
        A=tr[["d_nH_d_logT","d_nH_d_logNd"]].to_numpy()
        JW=J*np.sqrt(weights[:,None]);AW=A*np.sqrt(weights[:,None])
        residual=JW-AW@np.linalg.pinv(AW)@JW
        ranks.append(dict(outer_family=family,train_families=fit["train_families"],
            energy_singular_values=np.linalg.svd(JW,compute_uv=False).tolist(),
            energy_column_norms=dict(zip(names,np.linalg.norm(JW,axis=0).tolist())),
            residual_to_process_norms=dict(zip(names,np.linalg.norm(residual,axis=0).tolist())),
            scope="local, row/parameter scaling explicit; not a global identifiability theorem"))
    # Multiple energy directions require a fresh derivative check beyond V1's
    # one-basis gate. These are two virtual numerical cases.
    for element,dose in [("Si",1e19),("Sn",1e19)]:
        layer=layers[element];pars=layer.parameters(T=1325.,Nd=1e16,weights=[.05,-.05,.03],doses=(dose,))
        r=layer.predict(pars)
        for j in range(3):
            a=pars.copy();b=pars.copy();a[2+j]+=1e-5;b[2+j]-=1e-5
            fd=(layer.predict(a).values-layer.predict(b).values)/2e-5
            error=float(np.max(abs(fd-r.jacobian[:,2+j])/(1+abs(r.jacobian[:,2+j]))))
            checks.append(dict(element=element,basis=names[j],scaled_error=error))
    pd.DataFrame(records).to_csv(OUT/"energy_basis_sensitivities_v1.csv",index=False)
    (OUT/"energy_basis_training_ranks_v1.json").write_text(json.dumps(ranks,ensure_ascii=False,indent=2)+"\n")
    (OUT/"multi_basis_gradient_gate.json").write_text(json.dumps(dict(cases=checks,passed=max(x["scaled_error"] for x in checks)<3e-5),indent=2)+"\n")
    summary=dict(names=names,parent_counts={name:int(len(np.unique(db.cs_ID[B[:,j]>0]))) for j,name in enumerate(names)},
        mean_training_sensitivity_norm={name:float(np.mean([x["energy_column_norms"][name] for x in ranks])) for name in names},
        mean_residual_to_process_norm={name:float(np.mean([x["residual_to_process_norms"][name] for x in ranks])) for name in names},
        multi_basis_gradient_passed=max(x["scaled_error"] for x in checks)<3e-5,
        next_hypothesis="Test a regularized parent-consistent complex-energy closure, separate from process learning, if complex directions carry incremental target sensitivity.")
    (OUT/"energy_basis_diagnosis_v1.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    assert summary["multi_basis_gradient_passed"]

if __name__=="__main__":main()
