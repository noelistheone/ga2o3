"""Label-free adapter for new spectral models and all retained controls."""
from .joint_saved_model import SavedJointModel
from .correlated_inventory_model import CorrelatedInventoryModel
from .coherent_fermi import published_coherent_profile

FEATURES={'ChargeRatioSpectrum':'charge_ratio','ConstantSpectrum':'constant',
    'DoseSpectrum':'dose','ComplexSpectrum':'complex'}


class CorrelatedSavedModel(SavedJointModel):
    def __init__(self,fit,*,profile=None,db=None):
        method=fit['method']
        if method not in FEATURES:
            super().__init__(fit,profile=profile,db=db);return
        if not fit.get('training_valid') or not fit.get('selected_success'):
            raise ValueError('Saved spectral fit invalid or its optimizer did not stop')
        self.fit=dict(fit);self.method=method
        self.engine=CorrelatedInventoryModel(feature=FEATURES[method],
            profile=published_coherent_profile() if profile is None else profile,db=db)
        self.physical=True
        self.head=fit.get('transport_head','native')
        self.head_offset=float(fit.get('transport_head_offset_dex',0.))
        if self.head!='native' or self.head_offset!=0:
            raise ValueError('New spectral models require their native same-state readout')
