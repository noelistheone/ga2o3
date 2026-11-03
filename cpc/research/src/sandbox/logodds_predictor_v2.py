"""Domain-checked research API; unchanged numerical predictor underneath."""
import numpy as np
from .logodds_predictor import LogOddsResearchPredictor


class CheckedLogOddsResearchPredictor(LogOddsResearchPredictor):
    @staticmethod
    def validate_inputs(r):
        if r['element'] not in ('Si','Sn'):raise ValueError('This trained relationship is validated only for conditional Si/Sn research;other dopants require a new protocol')
        values=[r[k] for k in ('incorporated_cm3','H_cm3','Tobs_K','growth_T_C','thermal_cap_K')]
        if not np.isfinite(values).all() or r['incorporated_cm3']<=0 or r['Tobs_K']<=0:raise ValueError('Finite positive dose/observation temperature required')
        if r['H_cm3']<0 or bool(r['H_known'])!=(r['H_cm3']>0):raise ValueError('Declared H status and concentration disagree')
        if not 600<r['thermal_cap_K']<=1950:raise ValueError('Outside declared conditional thermal-cap domain')

    def shared_log_n(self,inputs,logodds=0.):
        for r in inputs:self.validate_inputs(r)
        if not np.isfinite(logodds) or abs(logodds)>6+1e-12:raise ValueError('Outside declared source log-odds domain')
        return super().shared_log_n(inputs,logodds)

    def calibrate(self,anchors,kind,lam):
        if np.isnan(lam) or lam<0:raise ValueError('Nonnegative regularization required')
        for a in anchors:
            self.validate_inputs(a['input'])
            if not np.isfinite(a['observed_log_nH']):raise ValueError('Finite supplied anchor carrier observation required')
        return super().calibrate(anchors,kind,lam)

    def predict(self,inputs,calibration):
        self.validate_inputs(inputs);kind=calibration['kind'];z=float(calibration['shift'])
        if kind not in ('offset','logodds') or not np.isfinite(z):raise ValueError('Explicit finite source coordinate required')
        if kind=='logodds' and abs(z)>6+1e-12:raise ValueError('Outside declared source log-odds domain')
        return super().predict(inputs,calibration)
