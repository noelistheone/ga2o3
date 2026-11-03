"""JSON interface to explicit-budget full-source ML+Sandbox predictors."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sandbox.property_predictor_v5 import DopedPropertyPredictor
from sandbox.kroger_db import load


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-json',type=Path,required=True)
    ap.add_argument('--output',type=Path);ap.add_argument('--bundle',type=Path,
        default=ROOT/'results/trainable_sandbox_20261017/phase19_delivery/predictor_bundle_v5')
    args=ap.parse_args();q=json.loads(args.input_json.read_text())
    result=DopedPropertyPredictor(load(),args.bundle).predict(**q)
    def clean(v):
        import numpy as np
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
        if isinstance(v,np.generic):return v.item()
        return v
    text=json.dumps(clean(result),ensure_ascii=False,indent=2)+'\n'
    if args.output:args.output.write_text(text)
    else:print(text,end='')


if __name__=='__main__':main()
