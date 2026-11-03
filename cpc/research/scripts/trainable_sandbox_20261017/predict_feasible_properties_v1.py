"""CLI for saved charge-feasible and absolute-share and inherited context-property predictors."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from sandbox.feasible_property_predictor_v2 import FeasiblePropertyPredictorV2 as PoolPropertyPredictor


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--request',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--zero-anchor-gradient',action='store_true');a=p.parse_args()
    model=PoolPropertyPredictor(a.bundle);q=json.loads(a.request.read_text())
    value=model.zero_anchor_gradient(**q) if a.zero_anchor_gradient else model.predict(**q)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('Saved pool/context property response',a.output)


if __name__=='__main__':main()
