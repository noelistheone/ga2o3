"""Exact process/learned-chemical-parameter chain through fixed zero-anchor physical readout."""
import numpy as np
from .chemical_process import feature_matrix


def zero_anchor_process_gradient(predictor,source_T_C,velocity_um_h,*,chemical_model,method,
                                source_inputs=(),anchors=(),growth_T_C=525.,observation_T_K=300.,pO2_atm=1e-5):
    if len(anchors):raise ValueError('This complete derivative covers zero Hall anchors;source refitting derivatives are not included')
    properties=predictor.properties
    if properties.manifest.get('version')!=8 or method not in properties.manifest.get('variants',{}):
        raise ValueError('Version8 current physical-state method required;legacy energy and finite quantile decisions are outside gradient scope')
    result=predictor.predict(source_T_C,velocity_um_h,chemical_model=chemical_model,method=method,source_inputs=source_inputs,
        growth_T_C=growth_T_C,observation_T_K=observation_T_K,pO2_atm=pO2_atm)
    spec=properties.manifest['variants'][method];model=properties.models[(spec['model_key'],float(pO2_atm))]
    if not model.physical:raise ValueError('Physical implicit charge/Hall state required for this derivative')
    if abs(result['raw_source_coordinate'])>0:raise ValueError('Zero source coordinate required')
    chem=result['process_calibration'];row=properties.parse('Si',dict(incorporated_cm3=chem['C_cm3'],growth_T_C=growth_T_C,observation_T_K=observation_T_K))
    state=model.point(row,source_kind=spec['source_kind'],shift=0.)
    values=np.r_[state.values[0],state.transport_values]
    if np.max(abs(values-[result['log10_nH_cm3'],result['log10_muH_cm2Vs'],result['log10_sigma_S_cm']]))>2e-9:
        raise RuntimeError('Implicit state does not match saved prediction')
    dproperty_dlogC=np.r_[state.input_jacobian[0,0],state.transport_input_jacobian[:,0]]
    dlogC_dprocess=np.array([chem['dlog10C_dsourceC'],chem['dlog10C_dlog10v']])
    record=predictor.chem.get(chemical_model).record
    dlogC_dtheta=feature_matrix(record['method'],np.array([source_T_C]),np.array([velocity_um_h]),record['rate_center'])[0]
    return dict(prediction=result,log_property_names=('log10_nH_cm3','log10_muH_cm2Vs','log10_sigma_S_cm'),
        process_input_names=('source_T_C','log10_velocity_um_h'),
        property_logC_jacobian=dproperty_dlogC,process_jacobian=dproperty_dlogC[:,None]*dlogC_dprocess[None,:],
        learned_chemical_parameter_jacobian=dproperty_dlogC[:,None]*dlogC_dtheta[None,:],
        scope='fullzero-Hall-anchor chain with frozen physical/shared/hyperparameters;chemical coefficients differentiable;no derivative throughCV/fitting map',
        knot_policy='source-T derivative is right sided at fixed piecewiselinear chemical knots')
