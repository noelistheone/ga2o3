"""Replay frozen energy selections on two additional virtual temperatures.

Requires the published-input energetic training assets; no labels are selected
from these query conditions, and the historical archive is read only.
"""
from pathlib import Path
from itertools import product
import sys,json,hashlib
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1]
ROOT=BASE/'research' if (BASE/'research').is_dir() else BASE.parents[3]
sys.path.insert(0,str(ROOT/'scripts/trainable_sandbox_20261017'))
import phase2_acquisition as a
from sandbox.goal_acquisition import condition_coordinate,prediction_variance_reduction
from sandbox.energy_learning import reconstruct
from sandbox.implicit_defects import ImplicitDefectLayer
D=BASE/'source_data';V=BASE/'verification'
QUERY=list(product((925.,1600.),(3e17,3e18,3e19,3e20)))

def run():
    protocol=dict(query=QUERY,budgets=[0,1,2,3,4],policies=['goal','energy_variance'],
        selection='Replay original selected labels; no re-ranking using new queries',
        target='Full-catalogue virtual response, not newly measured materials',
        protocol_correction='Initial900K is a thermochemical piece boundary rejected before any query scores; moved to925K, retaining the initial failure record',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dest=V/'acquisition_transfer_protocol.json'
    if dest.exists():assert json.loads(dest.read_text())==protocol
    else:dest.write_text(json.dumps(protocol,indent=2))
    out=[];labels=[];original_checks=[]
    for element in ('Si','Sn'):
        db,rr,mean,cov=a.prior(element,'extra_trees')
        truth=np.array([r['crossing_eV'] for r in rr]);ids={r['label_id']:j for j,r in enumerate(rr)}
        oracle=a.state_predictions(db,rr,truth,QUERY)[0]
        for policy in ('goal','energy_variance'):
            old=pd.read_csv(a.OUT/'replays'/f'{element}__extra_trees__{policy}__0.csv')
            m=mean.copy();C=cov.copy();acquired=[]
            for k in range(5):
                if k:
                    seq=old[old.budget.eq(k)].acquired_labels.iloc[0].split('|')
                    label=seq[-1];j=ids[label]
                    if policy=='goal' and k<=2:
                        before,J,valid=a.state_predictions(db,rr,m,a.DESIGN,True)
                        util=prediction_variance_reduction(J,C);rec=rr[j]
                        labels.append(dict(element=element,step=k,label_id=label,parent=rec['parent'],q=rec['q'],q_reference=rec['q_reference'],
                            delta_Ga=rec['dm_Ga'],delta_O=rec['dm_O'],dopant_atoms=1,selected_utility_rank=int(1+(util>util[j]+1e-12).sum()),
                            mean_abs_sensitivity_dex_per_eV=float(abs(J[:,j]).mean()),max_abs_sensitivity_dex_per_eV=float(abs(J[:,j]).max()),
                            local_design_variance_share=float(util[j]/max(np.trace(J@C@J.T)/len(J),1e-30))))
                    m,C=condition_coordinate(m,C,j,float(truth[j]));acquired.append(label)
                y,_,valid=a.state_predictions(db,rr,m,QUERY)
                for (T,dose),p,o,ok in zip(QUERY,y,oracle,valid):
                    out.append(dict(element=element,policy=policy,budget=k,Tf_K=T,dose_cm3=dose,predicted=p,full_catalogue=o,
                        discrepancy_dex=abs(p-o),physical_valid=bool(ok),acquired_labels='|'.join(acquired)))
                if k in (0,2,4):
                    original=a.state_predictions(db,rr,m,a.EVAL)[0]
                    ref=old[old.budget.eq(k)].sort_values(['Tf_K','dose_cm3']).predicted_log_nH.to_numpy(float)
                    original_checks.append(float(abs(original-ref).max()))
        print('Frozen acquisition transfer',element,flush=True)
    assert max(original_checks)<1e-8
    pd.DataFrame(out).to_csv(D/'acquisition_transfer_predictions.csv',index=False)
    pd.DataFrame(out).groupby(['element','policy','budget']).discrepancy_dex.agg(['mean','max']).reset_index().to_csv(D/'acquisition_transfer_scores.csv',index=False)
    pd.DataFrame(labels).to_csv(D/'acquisition_selected_states.csv',index=False)
    (V/'acquisition_transfer_checks.json').write_text(json.dumps(dict(original_replay_max_error=max(original_checks),original_replay_cases=len(original_checks),
        prediction_rows=len(out),all_valid=all(r['physical_valid'] for r in out),selected_labels=labels),indent=2))

if __name__=='__main__':run()
