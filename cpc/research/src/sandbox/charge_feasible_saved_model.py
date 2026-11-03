"""Label-free saved-model adapter for the feasible and endpoint-control families."""
import numpy as np
from . import transport
from .charge_feasible_model_v2 import ChargeFeasibleModel
from .pool_share_saved_model import SavedPoolModel
from .context_saved_model import SavedContextModel, ContextPropertyPoint
from .coherent_fermi import published_coherent_profile


class SavedFeasibleModel(SavedContextModel):
    def __init__(self, fit, *, profile=None):
        self.fit, self.method = dict(fit), fit['method']
        if not fit.get('training_valid') or not fit.get('selected_success', False):
            raise ValueError('Saved feasible fit is invalid or did not stop')
        actual = fit.get('evaluation_method', self.method)
        self.delegate = None
        profile = published_coherent_profile() if profile is None else profile
        if actual == 'ParentReference':
            self.delegate = SavedPoolModel(fit, profile=profile)
            self.operator, self.physical = self.delegate.operator, True
        else:
            prefix = 'Feasible' if actual.startswith('Feasible') else ('Envelope' if actual.startswith('Envelope') else None)
            if prefix is None:
                raise ValueError('Unknown feasible saved evaluation method')
            self.physical = prefix == 'Feasible'
            self.operator = ChargeFeasibleModel(basis=actual.replace(prefix, '').lower(), profile=profile,
                readout='physical' if self.physical else 'endpoint_envelope')
        self.engine = self.operator

    def point(self, row, *, source_kind='none', shift=0., require_observation=True):
        if row.element not in ('Si', 'Sn') or row.Tobs_K != 300.:
            raise ValueError('Saved predictor requires the declared Si/Sn300K comparison domain')
        if self.delegate is not None:
            return self.delegate.point(row, source_kind=source_kind, shift=shift, require_observation=require_observation)
        kind = 'share' if source_kind == 'field' else source_kind
        p = self.operator.point(row, self.fit['parameters'], source_kind=kind, source_shift=shift,
                                require_observation=require_observation)
        g, mu, sigma = float(p.values[0]), float(p.transport_values[0]), float(p.transport_values[1])
        ct = float(np.log10(transport.mu_caughey_thomas(10 ** g, 300.)))
        dg = dict(p.diagnostics, physical_state_available=self.physical,
            actual_evaluation_method=self.fit.get('evaluation_method', self.method),
            explicit_reference_fallback=bool(self.fit.get('reference_fallback', False)),
            conditional_on_electron_conducting_regime=True)
        return ContextPropertyPoint(g, mu, sigma, ct, float(np.log10(transport.E) + g + ct),
                                    float(p.source_jacobian[0]), bool(p.diagnostics['observation_domain_valid']), dg)
