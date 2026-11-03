"""Numerical gates and fresh-process, table-free checks of delivered models."""
from pathlib import Path
import sys,json,argparse,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))


def gates():
    import numpy as np
    import phase19_delivery as s
    from sandbox.property_predictor_v5 import make_model
    from sandbox.kroger_db import load
    from sandbox.source_stock_model import SourceStockModel
    from sandbox.material_profile import MaterialProfile
    from types import SimpleNamespace
    rows=[];db=load()
    rt=s.inputs.PRED[s.inputs.PRED.kind.eq('RT')]
    for method in s.SPECS:
        m=make_model(s.record(method),db);w=np.array(s.spec(method)[2][0])
        for r in (rt.iloc[0],rt[rt.element.eq('Sn')].iloc[0]):
            p=m.point(r,w)
            for k in range(len(w)):
                h=2e-4;wp=w.copy();wm=w.copy();wp[k]+=h;wm[k]-=h
                fd=(m.point(r,wp).values-m.point(r,wm).values)/(2*h)
                err=float(np.max(np.abs(fd-p.jacobian[:,k])/(1+np.abs(fd))))
                rows.append(dict(test='parameter_IFT',method=method,element=r.element,column=k,error=err,passed=err<2e-5))
    index=json.loads((s.BASE/'phase12_stock_adaptation/research_runtime/index.json').read_text())
    tasks=json.loads((s.BASE/'phase12_stock_adaptation/research_runtime/tasks.json').read_text())
    for item in index:
        if item['model']!='constant_stock' or item['budget']!=0:continue
        bundle=json.loads((ROOT/item['bundle']).read_text())
        old=SourceStockModel(bundle['parameters'],bundle['definition'],bundle['feature_metadata'],
            profile=MaterialProfile.from_dict(bundle['material_profile']),pO2_atm=1e-5,db=db)
        new=make_model(s.record('stock_const_noext_RT'),db)
        task=next(t for t in tasks if t['task_id']==item['task_id'])
        for p in task['points']:
            row=SimpleNamespace(**p['input'])
            for z in (0.,.3):
                a=old.predict(row,source_kind='stock',source_shift=z);b=new.point(row,bundle['parameters'],source_kind='stock',source_shift=z)
                err=float(max(np.max(np.abs(a.values-b.values)),np.max(np.abs(a.source_jacobian-b.source_jacobian))))
                rows.append(dict(test='old_stock_replay',obs_id=p['obs_id'],source_shift=z,error=err,passed=err<1e-7))
    s.write(s.OUT/'numerical_gates.json',dict(passed=all(r['passed'] for r in rows),checks=len(rows),rows=rows,
        maximum_error=max(r['error'] for r in rows),code_sha256=s.energy.sha(ROOT/'src/sandbox/property_predictor_v5.py')))
    print('gates',len(rows),'passed',all(r['passed'] for r in rows),'max',max(r['error'] for r in rows),flush=True)
    assert all(r['passed'] for r in rows)


def worker(task_path,result_path):
    forbidden=('dataset_v1.csv','episodes_v1.csv','temperature_observations.csv','paired_specimens.csv',
        'mobility_bounds_and_inputs.csv','/phase19_delivery/fits/','/phase19_delivery/training_targets/',
        '/primary_predictions.csv','Leedy2018_methods.txt')
    blocked=[];opens=[]
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            path=str(args[0]);opens.append(path)
            if any(s in path for s in forbidden):blocked.append(path);raise RuntimeError('Property-table access denied')
    sys.addaudithook(guard)
    from sandbox.property_predictor_v5 import DopedPropertyPredictor
    from sandbox.kroger_db import load
    task=json.loads(Path(task_path).read_text());model=DopedPropertyPredictor(load(),task['bundle']);rows=[]
    for q in task['queries']:
        try:result=model.predict(**q['input']);rows.append(dict(id=q['id'],result=result,error=None))
        except (ValueError,RuntimeError) as exc:rows.append(dict(id=q['id'],result=None,error=repr(exc)))
    def clean(v):
        import numpy as np
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
        if isinstance(v,np.generic):return v.item()
        return v
    Path(result_path).write_text(json.dumps(clean(dict(rows=rows,blocked=blocked,opened_paths=len(opens),property_tables_read=False)),ensure_ascii=False,indent=2)+'\n')


def verify():
    import numpy as np
    import phase19_delivery as s
    from sandbox.property_predictor_v5 import DopedPropertyPredictor
    from sandbox.kroger_db import load
    model=DopedPropertyPredictor(load(),s.BUNDLE);queries=[];expected={}
    # All old RT/curve points, with 0/1/2 anchors drawn explicitly from the old
    # RT specimen list. No RT/150K per-curve anchors are silently introduced.
    rt=s.inputs.PRED[s.inputs.PRED.kind.eq('RT')]
    for method,definition in model.manifest['methods'].items():
        supported=s.inputs.PRED if definition['branch']=='internal' else rt[~rt.H_known]
        for row in supported.itertuples():
            for b in (0,1,2):
                candidates=rt[(rt.family.eq(row.family))&rt.H_known.eq(row.H_known)&rt.H_cm3.eq(row.H_cm3)].sort_values('incorporated_cm3')
                if len(candidates)<b:continue
                # Explicit same-temperature RT anchors only. Do not transfer
                # RT measurements as if they were low-temperature labels.
                if row.Tobs_K!=300. and b:continue
                anchors=[dict(incorporated_cm3=float(a.incorporated_cm3),measured_nH_cm3=float(10**a.target)) for a in candidates.iloc[:b].itertuples()]
                args=dict(dopant=row.element,incorporated_cm3=float(row.incorporated_cm3),method=method,
                    observation_T_K=float(row.Tobs_K),H_known=bool(row.H_known),H_cm3=float(row.H_cm3),
                    growth_T_C=float(row.growth_T_C),anchors=anchors)
                key=f'q{len(queries)}';queries.append(dict(id=key,input=args))
                try:expected[key]=model.predict(**args)
                except (ValueError,RuntimeError) as exc:expected[key]=dict(error=repr(exc))
    domain_tests=[dict(dopant='Eu',incorporated_cm3=1e18),dict(dopant='Si',incorporated_cm3=-1.),
        dict(dopant='Si',incorporated_cm3=float('nan')),dict(dopant='Si',incorporated_cm3=1e18,pO2_atm=1e-3),
        dict(dopant='Si',incorporated_cm3=1e18,H_known=True,H_cm3=0.),
        dict(dopant='Si',incorporated_cm3=1e18,observation_T_K=200.),
        dict(dopant='Si',incorporated_cm3=1e18,anchors=[{'incorporated_cm3':1e18,'measured_nH_cm3':1e18}]*2),
        dict(dopant='Si',incorporated_cm3=1e18,H_known='False')]
    domain=[]
    for q in domain_tests:
        try:model.predict(**q);domain.append(False)
        except ValueError:domain.append(True)
    s.write(s.OUT/'runtime_tasks.json',dict(bundle=str(s.BUNDLE.resolve()),queries=queries))
    out=s.OUT/'runtime_outputs.json';run=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(s.OUT/'runtime_tasks.json'),str(out)],cwd=ROOT,capture_output=True,text=True)
    (s.OUT/'runtime_worker.log').write_text(run.stdout+'\n'+run.stderr);run.check_returncode()
    actual=json.loads(out.read_text());errors=[];failed=[]
    for row in actual['rows']:
        e=expected[row['id']]
        if 'error' in e:
            failed.append(dict(id=row['id'],expected=e['error'],actual=row['error'],matched=row['error']==e['error']));continue
        assert row['error'] is None,row
        for c in ('log10_nH_cm3','log10_muH_cm2Vs','log10_sigma_S_cm'):
            if e.get(c) is not None:errors.append(abs(e[c]-row['result'][c]))
    record=dict(passed=max(errors,default=0.)<2e-6 and all(domain) and all(r['matched'] for r in failed) and not actual['blocked'],
        inference_queries=len(queries),numeric_components=len(errors),maximum_replay_error=max(errors,default=0.),
        domain_checks=len(domain),domain_checks_passed=sum(domain),refused_predictions=failed,
        property_tables_read=False,blocked_paths=actual['blocked'],
        grade='saved-model independent-process integration check, not new material accuracy')
    s.write(s.OUT/'runtime_verification.json',record);print(json.dumps(record,ensure_ascii=False),flush=True);assert record['passed']


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--gates',action='store_true');ap.add_argument('--worker',nargs=2);a=ap.parse_args()
    if a.worker:worker(*a.worker)
    elif a.gates:gates()
    else:verify()
