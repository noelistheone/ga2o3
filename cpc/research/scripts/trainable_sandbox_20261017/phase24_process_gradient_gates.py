"""Independent process perturbations and learned-parameter perturbations through the physical state."""
from pathlib import Path
from copy import deepcopy
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from sandbox.process_property_predictor import ProcessPropertyPredictor
from sandbox.process_property_gradients import zero_anchor_process_gradient
from sandbox.chemical_process import ChemicalCalibration
import phase2_energy as e
BASE=e.BASE;OUT=BASE/'phase24_chemical_process';DEST=OUT/'process_gradients'


def main():
    chemical=OUT/'delivery/chemical_bundle';prop=BASE/'phase21_curve_response/delivery/predictor_bundle_v8';checks=[];rows=[]
    runner=ProcessPropertyPredictor(chemical,prop,'functional')
    def gate(name,good,**extra):checks.append(dict(name=name,passed=bool(good),**extra))
    maximum=0.
    for cname in ('central50::ArrheniusMassBalance','central50::MonotoneFlux','central50::DirectProcessSpline'):
        for method in ('OriginalHall','SharedReadout'):
            for T,v in ((1115.,.86),(1187.,.86),(1319.,1.3)):
                g=zero_anchor_process_gradient(runner,T,v,chemical_model=cname,method=method,source_inputs=[dict(source_T_C=T,velocity_um_h=v)])
                record=runner.chem.get(cname).record;coeff=np.array(record['coefficients']);spec=runner.properties.manifest['variants'][method]
                state_model=runner.properties.models[(spec['model_key'],1e-5)]
                def independent_values(temp,velocity,theta):
                    r=deepcopy(record);r['coefficients']=np.asarray(theta).tolist();chem=ChemicalCalibration(r).predict(temp,velocity)
                    row=runner.properties.parse('Si',dict(incorporated_cm3=chem['C_cm3'],growth_T_C=525.,observation_T_K=300.))
                    a=state_model.point(row,source_kind=spec['source_kind'],shift=0.)
                    return np.r_[a.values[0],a.transport_values]
                dT=.01;dlv=2e-4
                fd=np.column_stack([(independent_values(T+dT,v,coeff)-independent_values(T-dT,v,coeff))/(2*dT),
                    (independent_values(T,v*10**dlv,coeff)-independent_values(T,v/10**dlv,coeff))/(2*dlv)])
                parameter_fd=[]
                for j in range(len(coeff)):
                    h=2e-4;direction=np.zeros(len(coeff));direction[j]=h
                    parameter_fd.append((independent_values(T,v,coeff+direction)-independent_values(T,v,coeff-direction))/(2*h))
                parameter_fd=np.column_stack(parameter_fd);err=float(max(np.max(abs(fd-g['process_jacobian'])/(1+abs(fd))),
                    np.max(abs(parameter_fd-g['learned_chemical_parameter_jacobian'])/(1+abs(parameter_fd)))))
                maximum=max(maximum,err);gate('complete zeroanchor process and learnedparameter chain',err<5e-6,model=cname,method=method,source_T_C=T,error=err)
                gate('conductivity derivative identity',np.max(abs(g['process_jacobian'][2]-g['process_jacobian'][0]-g['process_jacobian'][1]))<1e-9,
                    model=cname,method=method,source_T_C=T)
                rows.append(dict(chemical_model=cname,method=method,source_T_C=T,velocity_um_h=v,maximum_scaled_difference_error=err,
                    lognH_dlogC=float(g['property_logC_jacobian'][0]),logmu_dlogC=float(g['property_logC_jacobian'][1]),
                    logsigma_dlogC=float(g['property_logC_jacobian'][2]),nH_dsource_C=float(g['process_jacobian'][0,0]),
                    mu_dsource_C=float(g['process_jacobian'][1,0]),nH_dlogv=float(g['process_jacobian'][0,1])))
    try:zero_anchor_process_gradient(runner,1187.,.86,chemical_model='central50::MonotoneFlux',method='OriginalHall',source_inputs=[dict(source_T_C=1187.,velocity_um_h=.86)],anchors=[dict(input_index=0,measured_nH_cm3=1e18)])
    except ValueError:gate('nonzero sourceadaptation derivatives explicitly refused',True)
    else:gate('nonzero sourceadaptation derivatives explicitly refused',False)
    e.csvwrite(DEST/'sensitivities.csv',rows)
    e.write(DEST/'gates.json',dict(passed=all(c['passed'] for c in checks),created_utc=e.now(),checks=len(checks),rows=checks,cases=len(rows),
        maximum_scaled_derivative_error=maximum,new_training_labels=0,goal_complete=False,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in (Path(__file__),ROOT/'src/sandbox/process_property_gradients.py')}))
    print('zeroanchor coupled process gradient gates',all(c['passed'] for c in checks),len(checks),'maxerror',maximum,flush=True)
    assert all(c['passed'] for c in checks),[c for c in checks if not c['passed']]


if __name__=='__main__':main()
