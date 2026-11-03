"""Synthetic serialization/source-isolation gates before current-cohort fits."""
from pathlib import Path
import subprocess,sys,numpy as np
from sklearn.ensemble import ExtraTreesRegressor
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import strong_tree_learning as l
from strong_tree_saved import raw_values,StrongTreeSavedModel
from sandbox.context_saved_model import feature_vector
from sandbox.ionized_transport import E_CHARGE
from sandbox.activation_transport_control import ct_log_readout
from sandbox.kroger_db import NPZ
from types import SimpleNamespace

OUT=l.OUT/'gates_v1'


def main():
    assert not (OUT/'protocol_before.json').exists()
    nodes=[c.design.inputs(r) for r in c.cases().drop_duplicates('sample_id').head(3).itertuples()]
    rows=[SimpleNamespace(**v) for v in nodes]
    code=[Path(__file__),Path(l.__file__),Path(__file__).with_name('strong_tree_saved.py'),
          Path(__file__).with_name('strong_tree_source_runtime.py'),c.OLD_SCRIPTS/'phase29_source_runtime.py']
    c.write(OUT/'protocol_before.json',dict(code_hashes=c.file_hashes(code),
        input_hashes=c.file_hashes([c.OLD/'data/scenarios49.csv']),labels='synthetic only',
        actual_current_cohort_training_not_started=True,goal_complete=False))
    X=np.array([feature_vector(r) for r in rows]);n=np.array([17.4,18.2,18.8]);rho=np.array([0.,.3,1.1])
    dose=np.array([np.log10(r.incorporated_cm3) for r in rows])
    rb=np.array([-(np.log10(E_CHARGE)+v+ct_log_readout(v,300.)[0]) for v in dose])
    tasks=[];checks=[];max_error=0.
    for method in l.METHODS:
        for paired in [False,True]:
            residual=method=='DoseResidualExtraTrees'
            a=ExtraTreesRegressor(n_estimators=64,random_state=1729,n_jobs=1).fit(X,n-dose if residual else n)
            b=(ExtraTreesRegressor(n_estimators=64,random_state=1730,n_jobs=1).fit(X,rho-rb if residual else rho)) if paired else None
            fit=dict(key=f'synthetic_{method}_{paired}',method=method,center=[0.]*10,scale=[1.]*10,
                nH_forest=l.serialize(a),rho_forest=l.serialize(b) if paired else None,
                selected_success=True,training_valid=True,transport_head='native',transport_head_offset_dex=0.)
            mapper=StrongTreeSavedModel(fit)
            expected=a.predict(X)+(dose if residual else 0.)
            for i,row in enumerate(rows):
                value=raw_values(fit,row);err=abs(value[0]-expected[i]);max_error=max(max_error,err)
                checks.append(err<1e-12)
                if paired:
                    er=abs(-value[2]-(b.predict(X)[i]+(rb[i] if residual else 0.)))
                    max_error=max(max_error,er);checks.append(er<1e-12)
                base=mapper.point(row,source_kind='offset',shift=0.)
                for shift in [-.2,0.,.3]:
                    point=mapper.offset_point(row,base,shift)
                    direct=mapper.point(row,source_kind='offset',shift=shift)
                    checks.append(abs(point.log_muH-direct.log_muH)<1e-12)
                    checks.append(abs(point.log_sigma-(np.log10(E_CHARGE)+point.log_nH+point.log_muH))<1e-12)
                    h=1e-5
                    plus=mapper.offset_point(row,base,shift+h);minus=mapper.offset_point(row,base,shift-h)
                    fd=np.array([plus.log_muH-minus.log_muH,plus.log_sigma-minus.log_sigma])/(2*h)
                    er=max(abs(fd-point.source_transport_derivative));max_error=max(max_error,float(er));checks.append(er<1e-8)
            episodes=[dict(budget=k,anchor_condition_ids=[f'synthetic{i}' for i in range(k)],
                anchor_indices=list(range(k)),anchor_labels=(n[:k]+np.array([.3,-.1])[:k]).tolist()) for k in [0,1,2]]
            tasks.append(dict(id=fit['key'],fit=fit,kind='offset',nodes=nodes,episodes=episodes))
    packet=OUT/'packet.json';target=OUT/'runtime.json'
    c.write(packet,dict(tasks=tasks,lambdas=[None,0.,.1,1.],physical_reference_sha256=c.e.sha(NPZ)))
    with (OUT/'runtime.log').open('x') as stream:
        subprocess.run([sys.executable,str(Path(__file__).with_name('strong_tree_source_runtime.py')),str(packet),str(target)],
            cwd=c.ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    runtime=c.document(target);checks.append(runtime['denied_reads']==[])
    for record in runtime['records']:
        for ep in record['episodes']:
            k=ep['budget']
            for sol in ep['solutions']:
                expected=0. if sol['lam'] is None or k==0 else float(np.mean(np.array([.3,-.1])[:k])/(1+sol['lam']))
                err=abs(sol['raw_shift']-expected);max_error=max(max_error,err)
                checks.extend([sol['valid'],err<1e-9])
                for point in sol['records']:
                    checks.append(abs(point['predicted_log_sigma']-(np.log10(E_CHARGE)+point['predicted']+point['predicted_log_mu']))<1e-12)
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    c.write(OUT/'gates.json',dict(passed=all(checks),checks=len(checks),maximum_error=max_error,
        isolated_runtime_tasks=4,denied_reads=runtime['denied_reads'],synthetic_label_only=True,
        no_prediction_gain_or_material_validity_claim=True,goal_complete=False))
    print('Strong tree synthetic gates',len(checks),'failed',sum(not x for x in checks),'max',max_error,flush=True)
    assert all(checks),'Preserve first gate failure'


if __name__=='__main__':main()
