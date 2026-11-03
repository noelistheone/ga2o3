"""Disclose reported carbon that is absent from the restricted defect state.

Numerical predictions, qualification flags and frozen source policies stay
identical toV2. A direct Ridge may use a carbon feature; that is disclosed
separately from including carbon in a physical atom/charge budget.
"""
from .feasible_property_predictor_v2 import FeasiblePropertyPredictorV2


class FeasiblePropertyPredictorV3(FeasiblePropertyPredictorV2):
    def predict(self, **kwargs):
        result = super().predict(**kwargs)
        model, _ = self.model(kwargs['preset'])
        direct_carbon_feature = model.fit['method'] == 'DirectRidge'
        affected = []
        for i, (row, point) in enumerate(zip(kwargs['inputs'], result['predictions'])):
            known = bool(row['carbon_known'] and float(row['carbon_cm3']) > 0)
            point.update(reported_carbon_unmodeled_in_physical_state=known,
                carbon_available_to_direct_feature_model=bool(known and direct_carbon_feature))
            if known:
                affected.append(i)
                point['warnings'].append('reported carbon is absent from the restricted Si/Sn/H defect state; numerical validity does not certify the full reported composition')
        result['reported_carbon_unmodeled_indices'] = affected
        result['carbon_scope'] = ('known carbon is not represented in the physical atom/charge budget; direct-feature use is separately marked'
            if affected else 'no positive reported carbon input;this does not establish experimentally carbon-free material')
        return result

    def zero_anchor_gradient(self, **kwargs):
        result = super().zero_anchor_gradient(**kwargs)
        row = kwargs['input']
        result['reported_carbon_unmodeled_in_physical_state'] = bool(row['carbon_known'] and float(row['carbon_cm3']) > 0)
        result['carbon_derivative_not_modeled'] = True
        return result
