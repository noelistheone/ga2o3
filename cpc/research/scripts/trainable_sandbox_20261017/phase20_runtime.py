"""Fresh-process v7 inference; experimental tables are forbidden in workers."""
from pathlib import Path
import sys,json,argparse,copy
ROOT=Path(__file__).resolve().parents[2]


def worker(task_path,result_path):
    blocked=[];opened=[]
    forbidden=('dataset_v1.csv','episodes_v1.csv','temperature_observations.csv','paired_specimens.csv',
        'mobility_bounds_and_inputs.csv','/fits/','/episodes/','/training_targets/',
        '/variants/','/primary_predictions.csv','/samples.csv','Leedy2018_methods.txt')
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            p=str(args[0]);opened.append(p)
            if any(s in p for s in forbidden):blocked.append(p);raise RuntimeError('Property/fit/source table read forbidden')
    sys.addaudithook(guard);sys.path.insert(0,str(ROOT/'src'))
    import numpy as np
    from sandbox.relative_predictor_v2 import DopedPropertyPredictor
    from sandbox.kroger_db import load
    payload=json.loads(Path(task_path).read_text());models={};rows=[]
    for q in payload['queries']:
        policy=q.get('policy','trained');bundle=q.get('bundle',payload.get('bundle'));model_key=(bundle,policy)
        if model_key not in models:
            m=DopedPropertyPredictor(load(),bundle)
            if policy!='trained':
                assert policy in ('no_update','unpenalized')
                m.manifest=copy.deepcopy(m.manifest)
                for spec in m.manifest['variants'].values():
                    for b in ('1','2'):spec['source_selection'][b]=dict(prior='quadratic',lam=None if policy=='no_update' else 0.)
            models[model_key]=m
        try:
            p=models[model_key].predict(**q['input']);rows.append(dict(id=q['id'],policy=policy,result=p,error=None))
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            rows.append(dict(id=q['id'],policy=policy,result=None,error=repr(exc)))
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
        if isinstance(v,np.generic):return v.item()
        return v
    Path(result_path).write_text(json.dumps(clean(dict(rows=rows,blocked_paths=blocked,opened_paths=len(opened),
        property_table_open_guard_active=True,property_tables_read=False,query_labels_in_worker=False)),ensure_ascii=False,allow_nan=False)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',nargs=2,required=True);args=ap.parse_args();worker(*args.worker)
