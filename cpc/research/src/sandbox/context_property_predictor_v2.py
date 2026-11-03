"""JSON-native types around the unchanged frozen phase25 prediction operator."""
import numpy as np
from .context_property_predictor import ContextPropertyPredictor as _Predictor


def json_native(value):
    if isinstance(value,np.generic):return json_native(value.item())
    if isinstance(value,np.ndarray):return json_native(value.tolist())
    if isinstance(value,dict):return {k:json_native(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_native(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):
        raise ValueError('Nonfinite public result cannot be serialized;no replacement or clipping is performed')
    return value


class ContextPropertyPredictorV2(_Predictor):
    def predict(self,**kwargs):return json_native(super().predict(**kwargs))
    def zero_anchor_gradient(self,**kwargs):return json_native(super().zero_anchor_gradient(**kwargs))
