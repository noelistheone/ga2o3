"""Versioned mixed predictor retaining all legacy models and adding feasible maps."""
from types import SimpleNamespace
import json
import numpy as np
from .pool_property_predictor import PoolPropertyPredictor, checked_input
from .context_property_predictor_v2 import json_native
from .charge_feasible_saved_model import SavedFeasibleModel


class FeasiblePropertyPredictor(PoolPropertyPredictor):
    def model(self, preset):
        if preset not in self.manifest['presets']:
            raise ValueError('Unknown saved preset')
        spec = self.manifest['presets'][preset]
        if spec.get('model_type') != 'charge_feasible':
            return super().model(preset)
        key = spec['model_key']
        if key not in self.models:
            name = 'models/' + key + '.json'
            if name not in self.paths:
                raise ValueError('Saved model absent from the verified bundle ledger')
            self.fits[key] = json.loads(self.paths[name].read_text())
            self.models[key] = SavedFeasibleModel(self.fits[key])
        return self.models[key], spec

    def predict(self, **kwargs):
        result = super().predict(**kwargs)
        spec = self.manifest['presets'][kwargs['preset']]
        if spec.get('model_type') == 'charge_feasible':
            result.update(conditional_on_electron_conducting_regime=True,
                conducting_scope='supplied Si/Sn n-type conducting condition domain;free floor is not a prediction of conductor versus insulator',
                actual_evaluation_method=self.fits[spec['model_key']].get('evaluation_method', spec['method']),
                envelope_scope='direct endpoint readout when a finite Envelope learner is selected;reference fallback evaluated as its actual physical model')
        return json_native(result)

    def zero_anchor_gradient(self, *, preset, input):
        if self.manifest['presets'][preset].get('model_type') != 'charge_feasible':
            return super().zero_anchor_gradient(preset=preset, input=input)
        row = SimpleNamespace(**checked_input(input))
        model, spec = self.model(preset)
        p = model.operator.point(row, model.fit['parameters'], require_observation=True)
        J = np.r_[p.jacobian[[0]], p.transport_jacobian]
        X = np.r_[p.input_jacobian[[0]], p.transport_input_jacobian]
        return json_native(dict(preset=preset, model_key=spec['model_key'],
            properties=['log10_nH', 'log10_muH', 'log10_sigma'],
            parameter_names=model.operator.parameter_names, parameter_jacobian=J.tolist(),
            input_names=p.input_names, input_jacobian=X.tolist(), includes_learned_input_chain=True,
            includes_threshold_input_chain=not bool(model.fit.get('reference_fallback')),
            physical_state_available=model.physical,
            learning_coordinate='conditional feasible excess share' if model.physical else 'direct endpoint-Hall interval coordinate',
            actual_evaluation_method=model.fit.get('evaluation_method', model.fit['method']),
            source_and_hyperparameter_refit_derivatives=False,
            conditional_on_electron_conducting_regime=True,
            temperature_derivative_scope='local mathematical sensitivity at300K;not new temperature-dependent validation',
            assumption='fixed trained parameters and supplied physical/input scenarios;no confidence guarantee'))
