"""Fresh-process v9 inference, with experimental/property-table reads forbidden."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[2]


def worker(task_path,result_path):
    forbidden=('dataset_v1.csv','episodes_v1.csv','temperature_observations.csv','paired_specimens.csv',
        'mobility_bounds_and_inputs.csv','/fits/','/episodes/','/variants/','/parent_reference.json','/samples.csv','/request_plans.json')
    blocked=[];opens=[]
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            s=str(args[0]);opens.append(s)
            if any(x in s for x in forbidden):blocked.append(s);raise RuntimeError('Experimental/fit/target table read forbidden')
    sys.addaudithook(guard);sys.path.insert(0,str(ROOT/'src'))
    import numpy as np
    from sandbox.ensemble_predictor import FiniteEnsemblePredictor
    payload=json.loads(Path(task_path).read_text());models={};rows=[]
    for q in payload['queries']:
        bundle=q.get('bundle',payload.get('bundle'))
        if bundle not in models:models[bundle]=FiniteEnsemblePredictor(bundle)
        try:r=models[bundle].predict(**q['input']);rows.append(dict(id=q['id'],result=r,error=None))
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:rows.append(dict(id=q['id'],result=None,error=repr(exc)))
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(tuple,list,np.ndarray)):return [clean(x) for x in v]
        if isinstance(v,np.generic):return v.item()
        return v
    cost=dict(source_point_calls=sum(sum(v[3] for v in m.sources.values()) for m in models.values()),
        query_nonreference_point_calls=sum(len(m.query_cache) for m in models.values()),
        reference_request_calls=sum(not q['input'].get('anchors') for q in payload['queries']))
    Path(result_path).write_text(json.dumps(clean(dict(rows=rows,blocked_paths=blocked,opened_paths=len(opens),property_tables_read=False,
        query_labels_in_worker=False,point_cost=cost,query_domain='all positive posterior support required; no invalid-state renormalization')),ensure_ascii=False,allow_nan=False)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',nargs=2,required=True);a=ap.parse_args();worker(*a.worker)
