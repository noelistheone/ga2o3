"""Preserve the physical map and use CT(updated label) for envelope offsets.

The original draft model and its gate outputs remain intact. Only the empirical
readout after an output offset changes; charge states and shared training at
zero source shift are identical.
"""
from dataclasses import replace
import numpy as np
from . import transport
from .charge_feasible_model import ChargeFeasibleModel as DraftChargeFeasibleModel


class ChargeFeasibleModel(DraftChargeFeasibleModel):
    def point(self, row, parameters, *, source_kind='none', source_shift=0., require_observation=True):
        point = super().point(row, parameters, source_kind=source_kind, source_shift=source_shift,
                              require_observation=require_observation)
        if self.readout != 'endpoint_envelope' or source_kind != 'offset':
            return point
        g = float(point.values[0])
        mu = float(transport.mu_caughey_thomas(10 ** g, row.Tobs_K))
        power = (10 ** g / transport.CT_N_REF) ** transport.CT_ALPHA
        ceiling = transport.CT_MU_MAX * transport.mu_pop_hall(row.Tobs_K) / transport.mu_pop_hall(300.)
        slope = -(ceiling - transport.CT_MU_MIN) * transport.CT_ALPHA * power / ((1 + power) ** 2 * mu)
        Tmu = -56. * 508. * np.exp(508. / row.Tobs_K) / row.Tobs_K ** 2
        slope_T = row.Tobs_K * transport.CT_MU_MAX * Tmu / transport.mu_pop_hall(300.) / ((1 + power) * mu)
        parameter_mu = slope * point.jacobian[0]
        source_mu = slope * point.source_jacobian[0]
        input_mu = slope * point.input_jacobian[0]
        input_mu[point.input_names.index('log10_Tobs_K')] += slope_T
        tv = np.array([np.log10(mu), np.log10(transport.E) + g + np.log10(mu)])
        return replace(point, transport_values=tv,
            transport_jacobian=np.vstack([parameter_mu, parameter_mu + point.jacobian[0]]),
            transport_source_jacobian=np.array([source_mu, source_mu + point.source_jacobian[0]]),
            transport_input_jacobian=np.vstack([input_mu, input_mu + point.input_jacobian[0]]),
            diagnostics=dict(point.diagnostics, envelope_offset_CT_recomputed=True,
                readout_scope='CT(updated predicted nH) after envelope output calibration;no physical midpoint state'))
