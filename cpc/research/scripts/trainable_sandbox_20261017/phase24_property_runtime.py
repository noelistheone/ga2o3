"""Fresh-process coupled calibration/property inference with no experimental/query-table reads."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[2]


def worker(task,output):
    blocked=[];opened=[];forbidden=('/process_audit/','/chemical_learning/','/extraction/','parent_reference.json','source_inputs.csv','dataset_v1.csv','episodes_v1.csv','/fits/','/episodes/','/variants/')
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            f=str(args[0]);opened.append(f)
            if any(s in f for s in forbidden):blocked.append(f);raise RuntimeError('Training/experimental/query read forbidden')
    sys.addaudithook(guard);sys.path.insert(0,str(ROOT/'src'))
    import numpy as np
    from sandbox.process_property_predictor import ProcessPropertyPredictor
    payload=json.loads(Path(task).read_text());models={};rows=[]
    for q in payload['queries']:
        key=(q['engine'],q['property_bundle'],q['chemical_bundle'])
        if key not in models:models[key]=ProcessPropertyPredictor(q['chemical_bundle'],q['property_bundle'],q['engine'])
        try:r=models[key].predict(**q['input']);rows.append(dict(id=q['id'],result=r,error=None))
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as ex:rows.append(dict(id=q['id'],result=None,error=repr(ex)))
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(a) for k,a in v.items()}
        if isinstance(v,(list,tuple,np.ndarray)):return [clean(a) for a in v]
        if isinstance(v,np.generic):return v.item()
        return v
    Path(output).write_text(json.dumps(clean(dict(rows=rows,blocked_paths=blocked,opened_paths=len(opened),property_tables_read=False,chemical_training_tables_read=False)),ensure_ascii=False,allow_nan=False)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',nargs=2,required=True);a=ap.parse_args();worker(*a.worker)
