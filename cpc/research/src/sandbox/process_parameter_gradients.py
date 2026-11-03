"""Convert coefficient gradients to actual independent trained chemical parameters."""
import numpy as np
from .process_property_gradients import zero_anchor_process_gradient


def zero_anchor_trained_parameter_gradient(predictor,*args,**kwargs):
    result=zero_anchor_process_gradient(predictor,*args,**kwargs)
    record=predictor.chem.get(kwargs['chemical_model']).record
    coefficient=result['learned_chemical_parameter_jacobian'];matrix=np.asarray(record.get('coefficient_tie_matrix',np.eye(coefficient.shape[1])),float)
    if matrix.shape[0]!=coefficient.shape[1] or not np.isfinite(matrix).all():raise ValueError('Invalid independent-parameter tie map')
    result['coefficient_parameter_jacobian']=coefficient
    result['learned_chemical_parameter_jacobian']=coefficient@matrix
    result['independent_chemical_parameter_count']=matrix.shape[1]
    result['coefficient_tie_matrix']=matrix
    result['parameter_scope']='independent continuous trained parameters;explicit tie map;zeroHallanchors/frozenphysicalparameters only'
    return result
