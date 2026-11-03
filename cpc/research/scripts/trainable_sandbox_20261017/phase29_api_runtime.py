"""Fresh-process saved API requests; experimental and fit-table reads denied."""
from pathlib import Path
import sys,json,builtins,io

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from sandbox.joint_property_predictor import JointPropertyPredictor,clean
from sandbox.kroger_db import NPZ


def main():
    task=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve()
    packet=json.loads(task.read_text());bundle=Path(packet['bundle']).resolve();physical=NPZ.resolve()
    real=builtins.open;real_io=io.open;blocked=[];reads=[]
    def guard(factory):
        def opened(file,mode='r',*args,**kwargs):
            if isinstance(file,(str,bytes,Path)) and ('r' in mode or '+' in mode):
                path=Path(file).resolve();reads.append(str(path))
                if 'results' in path.parts and path not in (task,physical) and bundle not in path.parents:
                    blocked.append(str(path));raise RuntimeError('Experimental/result/fit-table read forbidden in saved API')
            return factory(file,mode,*args,**kwargs)
        return opened
    builtins.open=guard(real);io.open=guard(real_io)
    predictor=JointPropertyPredictor(bundle);rows=[]
    for q in packet['queries']:
        try:
            result=predictor.predict(**q['input'])
            rows.append(dict(id=q['id'],result=result,error=None))
        except (ValueError,RuntimeError,ArithmeticError) as exc:
            rows.append(dict(id=q['id'],result=None,error=repr(exc)))
        print('API29',q['id'],'refused',rows[-1]['error'] is not None,flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(clean(dict(rows=rows,blocked_paths=blocked,opened_count=len(reads),
        query_tables_read=False,training_tables_read=False)),ensure_ascii=False,indent=2,allow_nan=False))
    print('API29 complete',len(rows),'refusals',sum(r['error'] is not None for r in rows),'blocked',len(blocked),flush=True)


if __name__=='__main__':main()
