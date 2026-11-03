"""Trained chemical-input layer coupled to saved ML+Sandbox or matched empirical predictors."""
from .chemical_process import SavedChemicalCalibrations


class ProcessPropertyPredictor:
    def __init__(self,chemical_bundle,property_bundle,engine='functional',db=None):
        self.chem=SavedChemicalCalibrations(chemical_bundle)
        if engine=='functional':
            from .functional_predictor import DopedPropertyPredictor
            if db is None:
                from .kroger_db import load
                db=load()
            self.properties=DopedPropertyPredictor(db,property_bundle)
        elif engine=='affine':
            from .affine_predictor import AffinePropertyPredictor
            self.properties=AffinePropertyPredictor(None,property_bundle)
        elif engine=='ensemble':
            from .ensemble_predictor import FiniteEnsemblePredictor
            self.properties=FiniteEnsemblePredictor(property_bundle)
        else:raise ValueError('Unknown saved property engine')

    def predict(self,source_T_C,velocity_um_h,*,chemical_model,method,source_inputs,anchors=(),
                growth_T_C=525.,observation_T_K=300.,pO2_atm=1e-5):
        if float(growth_T_C)!=525.:raise ValueError('Chemical process calibration only covers reported525C substrate conditions')
        model=self.chem.get(chemical_model);c=model.predict(source_T_C,velocity_um_h);design=[]
        for x in source_inputs:
            p=model.predict(x['source_T_C'],x['velocity_um_h']);design.append(dict(incorporated_cm3=p['C_cm3'],growth_T_C=525.))
        result=self.properties.predict('Si',c['C_cm3'],method=method,source_inputs=design,anchors=anchors,
            growth_T_C=525.,observation_T_K=observation_T_K,pO2_atm=pO2_atm)
        result.update(process_calibration=c,chemical_model=chemical_model,
            process_scope='learned nominalsourceT/rate->chemicalproxy->saved property model;not jointHall retraining or new independentlab',
            process_derivatives='chemical input derivatives only;full property derivative not exported',
            source_T_C=float(source_T_C),velocity_um_h=float(velocity_um_h))
        return result
