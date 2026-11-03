"""Use saved v7 models with explicit chemical inputs and optional RT anchors."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sandbox.relative_predictor_v2 import DopedPropertyPredictor
from sandbox.kroger_db import load


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-json',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--bundle',type=Path,default=ROOT/'results/trainable_sandbox_20261017/phase20_source_inventory/delivery_v2/predictor_bundle_v7')
    a=ap.parse_args();model=DopedPropertyPredictor(load(),a.bundle);payload=json.loads(a.input_json.read_text())
    requests=payload if isinstance(payload,list) else [payload];rows=[model.predict(**r) for r in requests]
    def clean(v):
        import numpy as np
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
        if isinstance(v,np.generic):return v.item()
        return v
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(clean(rows if isinstance(payload,list) else rows[0]),ensure_ascii=False,allow_nan=False,indent=2)+'\n')


if __name__=='__main__':main()
