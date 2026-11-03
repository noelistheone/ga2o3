"""CLI for saved conditional inventory ML+Sandbox and same-information controls."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from sandbox.context_property_predictor import ContextPropertyPredictor


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--request',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--zero-anchor-gradient',action='store_true')
    a=ap.parse_args();predictor=ContextPropertyPredictor(a.bundle);q=json.loads(a.request.read_text())
    result=predictor.zero_anchor_gradient(**q) if a.zero_anchor_gradient else predictor.predict(**q)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('Saved context-property prediction',a.output)


if __name__=='__main__':main()
