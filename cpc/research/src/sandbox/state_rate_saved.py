"""Isolated saved-model factory for state-rate controls and frozen parents."""
from .correlated_saved_model import CorrelatedSavedModel
from .state_rate_inventory import StateRateInventoryModel
from .coherent_fermi import published_coherent_profile

FEATURES={'ChargeRatioTau':'charge_ratio','DoseTau':'dose',
          'ComplexTau':'complex','GlobalTauResponse':'global'}


class StateRateSavedModel(CorrelatedSavedModel):
    def __init__(self,fit,*,profile=None,db=None):
        method=fit['method']
        if method not in FEATURES:
            super().__init__(fit,profile=profile,db=db);return
        if not fit.get('training_valid') or not fit.get('selected_success'):
            raise ValueError('Saved state-rate fit invalid or optimizer did not stop')
        self.fit=dict(fit);self.method=method;self.physical=True
        self.engine=StateRateInventoryModel(feature=FEATURES[method],
            profile=published_coherent_profile() if profile is None else profile,db=db)
        self.head=fit.get('transport_head','native')
        self.head_offset=float(fit.get('transport_head_offset_dex',0.))
        if self.head!='native' or self.head_offset!=0:
            raise ValueError('State-rate controls require native same-state readout')
