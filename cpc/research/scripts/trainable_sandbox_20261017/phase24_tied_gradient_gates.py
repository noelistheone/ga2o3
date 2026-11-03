"""Perturb independent coefficients of the chemically selected affine/spline models."""
from pathlib import Path
from copy import deepcopy
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from sandbox.process_property_predictor import ProcessPropertyPredictor
from sandbox.process_parameter_gradients import zero_anchor_trained_parameter_gradient
from sandbox.chemical_process import ChemicalCalibration
import phase2_energy as e
BASE=e.BASE;OUT=BASE/'phase24_chemical_process'


def main():
    runner=ProcessPropertyPredictor(OUT/'delivery/paired_chemical_bundle',BASE/'phase21_curve_response/delivery/predictor_bundle_v8');rows=[];maximum=0.
    for cname in runner.chem.manifest['models']:
        record=runner.chem.get(cname).record;M=np.array(record.get('coefficient_tie_matrix',np.eye(len(record['coefficients']))));theta=np.array(record.get('independent_coefficients',record['coefficients']))
        for T,v in ((1187.,.86),(1319.,1.3)):
            g=zero_anchor_trained_parameter_gradient(runner,T,v,chemical_model=cname,method='OriginalHall',source_inputs=[]);spec=runner.properties.manifest['variants']['OriginalHall'];model=runner.properties.models[(spec['model_key'],1e-5)];fd=[]
            def value(perturb):
                r=deepcopy(record);r['coefficients']=(M@perturb).tolist();c=ChemicalCalibration(r).predict(T,v)
                row=runner.properties.parse('Si',dict(incorporated_cm3=c['C_cm3'],growth_T_C=525.,observation_T_K=300.));a=model.point(row,source_kind=spec['source_kind'],shift=0.)
                return np.r_[a.values[0],a.transport_values]
            for j in range(len(theta)):
                d=np.zeros(len(theta));d[j]=2e-4;fd.append((value(theta+d)-value(theta-d))/(4e-4))
            fd=np.column_stack(fd);err=float(np.max(abs(fd-g['learned_chemical_parameter_jacobian'])/(1+abs(fd))));maximum=max(maximum,err)
            rows.append(dict(chemical_model=cname,family=record['family'],independent_parameters=len(theta),source_T_C=T,maximum_scaled_error=err,passed=err<5e-6))
    e.write(OUT/'process_gradients/tied_parameter_gates.json',dict(passed=all(r['passed'] for r in rows),cases=len(rows),rows=rows,maximum_scaled_error=maximum,
        zero_Hall_anchors_only=True,new_training_labels=0,goal_complete=False,code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in (Path(__file__),ROOT/'src/sandbox/process_parameter_gradients.py')}))
    print('tied independent-parameter chain',all(r['passed'] for r in rows),len(rows),'cases',maximum,flush=True);assert all(r['passed'] for r in rows)


if __name__=='__main__':main()
