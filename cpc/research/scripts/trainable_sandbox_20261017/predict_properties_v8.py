"""Explicit-method entrypoint for the21 Sandbox presets and3 affine controls."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sandbox.functional_predictor import DopedPropertyPredictor
from sandbox.affine_predictor import AffinePropertyPredictor
from sandbox.kroger_db import load


def main():
    base=ROOT/'results/trainable_sandbox_20261017/phase21_curve_response'
    ap=argparse.ArgumentParser();ap.add_argument('--input-json',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--list-methods',action='store_true')
    ap.add_argument('--transport-comparisons',action='store_true');a=ap.parse_args()
    fb=base/'delivery/predictor_bundle_v8';ab=base/'affine_controls/predictor_bundle';fm=json.loads((fb/'manifest.json').read_text());am=json.loads((ab/'manifest.json').read_text())
    if a.list_methods:print(json.dumps(dict(Sandbox=list(fm['variants']),energy_completion=fm['energy_methods'],affine_controls=list(am['variants'])),ensure_ascii=False,indent=2));return
    if a.input_json is None or a.output is None:ap.error('--input-json and --output are required for prediction')
    requests=json.loads(a.input_json.read_text());items=requests if isinstance(requests,list) else [requests];models={};outputs=[]
    for q in items:
        if 'method' not in q:raise ValueError('Select an explicit saved method;experimental variants are not recommendations')
        engine='affine' if q['method'] in am['variants'] else 'functional'
        if engine not in models:models[engine]=AffinePropertyPredictor(None,ab) if engine=='affine' else DopedPropertyPredictor(load(),fb)
        p=models[engine].predict(**q)
        if a.transport_comparisons:
            from sandbox.chemical_transport_control import chemical_hall_transport
            from sandbox.coherent_fermi import published_coherent_profile
            profile=published_coherent_profile();p['transport_comparisons']={}
            for charge in (1,3):p['transport_comparisons'][str(charge)]=chemical_hall_transport(p['log10_nH_cm3'],float(q['incorporated_cm3']),
                float(q.get('H_cm3',0.)),float(q.get('observation_T_K',300.)),acceptor_charge=charge,profile=profile)
        outputs.append(p)
    def clean(v):
        import numpy as np
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(tuple,list,np.ndarray)):return [clean(x) for x in v]
        if isinstance(v,np.generic):return v.item()
        return v
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(clean(outputs if isinstance(requests,list) else outputs[0]),ensure_ascii=False,allow_nan=False,indent=2)+'\n')


if __name__=='__main__':main()
