"""Explicit research entrypoint for finite-state source inference; preserves refusals."""
from pathlib import Path
import argparse,sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sandbox.ensemble_predictor import FiniteEnsemblePredictor


def main():
    base=ROOT/'results/trainable_sandbox_20261017/phase22_predictive_inference/delivery/predictor_bundle_v9'
    ap=argparse.ArgumentParser();ap.add_argument('--list-methods',action='store_true');ap.add_argument('--input-json',type=Path);ap.add_argument('--output',type=Path);a=ap.parse_args()
    if a.list_methods:
        print(json.dumps(list(json.loads((base/'manifest.json').read_text())['variants']),ensure_ascii=False,indent=2));return
    if a.input_json is None or a.output is None:ap.error('--input-json and --output required')
    raw=json.loads(a.input_json.read_text());requests=raw if isinstance(raw,list) else [raw];model=FiniteEnsemblePredictor(base);out=[]
    for i,q in enumerate(requests):
        try:
            if 'method' not in q:raise ValueError('Select an explicit method; these prototypes are not accuracy recommendations')
            result=model.predict(**q);out.append(dict(id=i,result=result,error=None))
        except (ValueError,RuntimeError) as exc:out.append(dict(id=i,result=None,error=repr(exc)))
    def clean(v):
        import numpy as np
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(tuple,list,np.ndarray)):return [clean(x) for x in v]
        if isinstance(v,np.generic):return v.item()
        return v
    a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(clean(out if isinstance(raw,list) else out[0]),ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    if any(x['error'] is not None for x in out):raise SystemExit(1)


if __name__=='__main__':main()
