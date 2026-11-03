"""Fresh-process inference using explicit source designs and anchor labels only."""
from pathlib import Path
import sys,json,argparse,copy
ROOT=Path(__file__).resolve().parents[2]


def worker(task_path,result_path):
    forbidden=('dataset_v1.csv','episodes_v1.csv','temperature_observations.csv','paired_specimens.csv',
        'mobility_bounds_and_inputs.csv','/fits/','/episodes/','/variants/','/parent_reference.json','/samples.csv','/request_plans.json')
    blocked=[];opens=[]
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            p=str(args[0]);opens.append(p)
            if any(s in p for s in forbidden):blocked.append(p);raise RuntimeError('Experimental/fit/selection table read forbidden')
    sys.addaudithook(guard);sys.path.insert(0,str(ROOT/'src'))
    import numpy as np
    from sandbox.functional_predictor import DopedPropertyPredictor
    from sandbox.affine_predictor import AffinePropertyPredictor
    from sandbox.kroger_db import load
    payload=json.loads(Path(task_path).read_text());models={};rows=[]
    for q in payload['queries']:
        policy=q.get('policy','trained');bundle=q.get('bundle',payload.get('bundle'));engine=q.get('engine',payload.get('engine','functional'));key=(engine,bundle,policy)
        if key not in models:
            model=AffinePropertyPredictor(None,bundle) if engine=='affine' else DopedPropertyPredictor(load(),bundle)
            if policy!='trained':
                assert policy in ('no_update','unpenalized');model.manifest=copy.deepcopy(model.manifest)
                for variant in model.manifest['variants'].values():
                    for budget in ('1','2'):variant['source_selection'][budget]=dict(prior='quadratic',lam=None if policy=='no_update' else 0.)
            models[key]=model
        try:r=models[key].predict(**q['input']);rows.append(dict(id=q['id'],policy=policy,result=r,error=None))
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:rows.append(dict(id=q['id'],policy=policy,result=None,error=repr(exc)))
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(tuple,list,np.ndarray)):return [clean(x) for x in v]
        if isinstance(v,np.generic):return v.item()
        return v
    Path(result_path).write_text(json.dumps(clean(dict(rows=rows,blocked_paths=blocked,opened_paths=len(opens),property_table_open_guard_active=True,
        property_tables_read=False,query_labels_in_worker=False)),ensure_ascii=False,allow_nan=False)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',nargs=2,required=True);a=ap.parse_args();worker(*a.worker)
