"""Metadata-only clarification for an explicitly selected physical reference."""
from .feasible_property_predictor import FeasiblePropertyPredictor


class FeasiblePropertyPredictorV2(FeasiblePropertyPredictor):
    def zero_anchor_gradient(self, **kwargs):
        result = super().zero_anchor_gradient(**kwargs)
        if result.get('actual_evaluation_method') == 'ParentReference':
            result['learning_coordinate'] = 'explicit physical reference;no learned share coefficients'
            result['includes_threshold_input_chain'] = False
        return result
