"""Saved process-to-property prediction; zero-anchor gradients use independent tied parameters."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sandbox.process_property_predictor import ProcessPropertyPredictor
from sandbox.process_parameter_gradients import zero_anchor_trained_parameter_gradient


def main():
    p=argparse.ArgumentParser();p.add_argument('--chemical-bundle',type=Path,required=True);p.add_argument('--property-bundle',type=Path,required=True)
    p.add_argument('--engine',choices=['functional','affine','ensemble'],default='functional');p.add_argument('--request',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--zero-anchor-gradient',action='store_true');a=p.parse_args()
    model=ProcessPropertyPredictor(a.chemical_bundle,a.property_bundle,a.engine);q=json.loads(a.request.read_text())
    result=zero_anchor_trained_parameter_gradient(model,**q) if a.zero_anchor_gradient else model.predict(**q)
    def clean(x):
        import numpy as np
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
        if isinstance(x,np.generic):return x.item()
        return x
    a.output.write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n');print('Saved process prediction/independent-parameter gradient',a.output)


if __name__=='__main__':main()
