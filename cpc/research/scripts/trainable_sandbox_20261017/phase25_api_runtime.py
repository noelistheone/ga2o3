"""Fresh saved-bundle requests with experimental/training-table reads forbidden."""
from pathlib import Path
import sys,json,builtins,io

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from sandbox.context_property_predictor import ContextPropertyPredictor
from sandbox.kroger_db import NPZ


def main():
    task=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve()
    payload=json.loads(task.read_text());bundle=Path(payload['bundle']).resolve();physical=NPZ.resolve()
    blocked=[];opened=0;real=builtins.open;real_io=io.open
    def guard(factory):
        def openfile(file,mode='r',*args,**kwargs):
            nonlocal opened
            if isinstance(file,(str,Path)) and ('r' in mode or '+' in mode):
                path=Path(file).resolve();opened+=1
                if 'results' in path.parts and path not in (task,physical) and bundle not in path.parents:
                    blocked.append(str(path));raise RuntimeError('Experimental/result read forbidden in saved API')
            return factory(file,mode,*args,**kwargs)
        return openfile
    builtins.open=guard(real);io.open=guard(real_io)
    predictor=ContextPropertyPredictor(bundle);rows=[]
    for q in payload['queries']:
        try:
            result=predictor.zero_anchor_gradient(**q['input']) if q.get('gradient') else predictor.predict(**q['input'])
            rows.append(dict(id=q['id'],result=result,error=None))
        except (ValueError,RuntimeError,ArithmeticError) as exc:rows.append(dict(id=q['id'],result=None,error=repr(exc)))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(rows=rows,blocked_paths=blocked,opened_count=opened,
        query_tables_read=False,training_tables_read=False),ensure_ascii=False,indent=2,allow_nan=False))
    print('saved context API worker',len(rows),'errors',sum(r['error'] is not None for r in rows),'blocked',len(blocked),flush=True)


if __name__=='__main__':main()
