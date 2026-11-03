"""Frozen predictors on new-source inputs; no property tables or refitting."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[2]


def worker(task,output):
    forbidden=('/process_audit/','published_Hall_table.csv','source_contract.json','dataset_v1.csv','episodes_v1.csv',
        '/parent_reference.json','/fits/','/episodes/','/variants/','/source_inputs.csv')
    blocked=[];opened=[]
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            f=str(args[0]);opened.append(f)
            if any(x in f for x in forbidden):blocked.append(f);raise RuntimeError('Target/experimental/fit read forbidden')
    sys.addaudithook(guard);sys.path.insert(0,str(ROOT/'src'))
    import numpy as np
    from sandbox.functional_predictor import DopedPropertyPredictor
    from sandbox.affine_predictor import AffinePropertyPredictor
    from sandbox.ensemble_predictor import FiniteEnsemblePredictor
    from sandbox.kroger_db import load
    payload=json.loads(Path(task).read_text());models={};rows=[];db=None
    for q in payload['queries']:
        key=(q['engine'],q['bundle'])
        if key not in models:
            if q['engine']=='functional':
                if db is None:db=load()
                model=DopedPropertyPredictor(db,q['bundle'])
            elif q['engine']=='affine':model=AffinePropertyPredictor(None,q['bundle'])
            else:model=FiniteEnsemblePredictor(q['bundle'])
            models[key]=model
        try:r=models[key].predict(**q['input']);rows.append(dict(id=q['id'],result=r,error=None))
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:rows.append(dict(id=q['id'],result=None,error=repr(exc)))
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
        if isinstance(v,np.generic):return v.item()
        return v
    Path(output).write_text(json.dumps(clean(dict(rows=rows,blocked_paths=blocked,opened_paths=len(opened),query_labels_in_worker=False,property_tables_read=False)),ensure_ascii=False,allow_nan=False)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',nargs=2,required=True);a=ap.parse_args();worker(*a.worker)
