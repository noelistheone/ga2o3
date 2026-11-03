"""Shared excess-share prediction and a matched endpoint-envelope control."""
import numpy as np
from scipy.special import expit
from . import transport
from .pool_share_model import PoolShareModel
from .relative_inventory_model import RelativeInventoryPrediction, L0, L1
from .context_inventory_model import field_features
from .charge_feasible_share import allocate_feasible_share
from .coherent_transport import fermi_transport

LN10 = float(np.log(10.))


class ChargeFeasibleModel(PoolShareModel):
    def __init__(self, *, basis, profile, readout='physical', db=None):
        if readout not in ('physical', 'endpoint_envelope'):
            raise ValueError('Explicit physical or endpoint-envelope readout required')
        super().__init__(basis=basis, temperature_mode='effective', profile=profile, db=db)
        self.readout = readout

    def point(self, row, parameters, *, source_kind='none', source_shift=0., require_observation=True):
        w, z = np.asarray(parameters, float), float(source_shift)
        n = len(self.parameter_names)
        if w.shape != (n,) or not np.isfinite(w).all():
            raise ValueError('Matching finite excess-share model parameters required')
        if row.element not in ('Si', 'Sn') or not np.isfinite([
                row.incorporated_cm3, row.Tobs_K, row.H_cm3, row.growth_T_C]).all():
            raise ValueError('Explicit finite chemical and temperature inputs required')
        if row.incorporated_cm3 <= 0 or row.Tobs_K <= 0 or row.H_cm3 < 0 or bool(row.H_known) != (row.H_cm3 > 0):
            raise ValueError('Positive dose/temperature and explicit Hknown/unknown-zero scenario required')
        if source_kind not in ('none', 'share', 'offset') or not np.isfinite(z) or abs(z) > 6 or (source_kind == 'none' and z != 0):
            raise ValueError('Declared supported source coordinate in [-6,6] required')
        layer, solver = self.engine_for(row).layer_for(row)
        ni = 3 if row.H_known else 2
        nv, source_col, input_col = n + 1 + ni + 1, n, n + 1
        growth_col = nv - 1
        chain = np.zeros((layer.np, nv))
        logistic = float(expit(w[0]))
        T = 10 ** (L0 + (L1 - L0) * logistic)
        chain[0, 0] = (L1 - L0) * logistic * (1 - logistic)
        chain[1, 1] = 1.
        chain[layer.idose, input_col] = 1.
        chain[layer.iobs, input_col + 1] = 1.
        if row.H_known:
            chain[2 + layer.b + 1, input_col + 2] = 1.
        phi = field_features(row, self.basis)
        psi = float(phi @ w[2:])
        dpsi = np.zeros(nv)
        dpsi[2:n] = phi
        if self.basis != 'constant':
            dpsi[input_col] = w[3] / 2.
        if self.basis == 'process':
            dpsi[growth_col] = w[4] / 300.
        if source_kind == 'share':
            psi += LN10 * z
            dpsi[source_col] = LN10
        pars = layer.parameters(T=T, Nd=10 ** w[1], weights=[0.],
            doses=(row.incorporated_cm3, row.H_cm3) if row.H_known else (row.incorporated_cm3,),
            pO2=row.scenario_pO2_atm, Tobs=row.Tobs_K, binding_eV=.0366, screen_s=0.)

        def boundary(endpoint=None):
            inventory = allocate_feasible_share(solver, pars, chain, psi, dpsi, endpoint=endpoint)
            cold = solver.solve(pars, inventory, chain)
            tr = fermi_transport(cold.values[0], cold.values[2], float(row.Tobs_K),
                Nc_ref_cm3=self.profile.Nc_ref, T_ref_K=self.profile.T_ref,
                mass_ratio=self.profile.electron_mass_ratio, extra_mobility_cm2Vs=205., hall=True)
            readout = tr.jacobian[:, :2] @ cold.jacobian[[0, 2]]
            readout[:, input_col + 1] += tr.jacobian[:, 2]
            return cold, tr, readout

        if self.readout == 'physical':
            cold, tr, readout = boundary()
            values, J = cold.values.copy(), cold.jacobian.copy()
            values[0], J[0] = tr.values[0], readout[0]
            tv, TJ = tr.values[1:].copy(), readout[1:].copy()
            physical = values.copy()
            carrier_eligible = bool(cold.diagnostics['single_electron_Hall_valid'])
            dg = dict(cold.diagnostics, physical_state_available=True,
                coherent_transport=tr.diagnostics, hall_factor=tr.diagnostics['readout_hall_factor'])
        else:
            low, tr_low, Jlow = boundary(0)
            high, tr_high, Jhigh = boundary(1)
            raw_low, b = float(tr_low.values[0]), float(tr_high.values[0])
            # This strong control also knows the pre-existing Hall detection
            # floor. It interpolates the endpoint interval intersected with
            # that label domain; no measured query value sets this floor.
            a = max(13., raw_low)
            low_jac = Jlow[0] if raw_low > 13. else np.zeros(nv)
            if not b > a:
                raise ValueError('Nonordered physical endpoint Hall readouts')
            s, complement = float(expit(psi)), float(expit(-psi))
            value = a + (b - a) * s
            dvalue = low_jac + (Jhigh[0] - low_jac) * s + (b - a) * s * complement * dpsi
            mu = float(transport.mu_caughey_thomas(10 ** value, row.Tobs_K))
            power = (10 ** value / transport.CT_N_REF) ** transport.CT_ALPHA
            ceiling = transport.CT_MU_MAX * transport.mu_pop_hall(row.Tobs_K) / transport.mu_pop_hall(300.)
            dg_mu = -(ceiling - transport.CT_MU_MIN) * transport.CT_ALPHA * power / ((1 + power) ** 2 * mu)
            Tmu = -56. * 508. * np.exp(508. / row.Tobs_K) / row.Tobs_K ** 2
            dlogT_mu = row.Tobs_K * transport.CT_MU_MAX * Tmu / transport.mu_pop_hall(300.) / ((1 + power) * mu)
            mu_jac = dg_mu * dvalue
            mu_jac[input_col + 1] += dlogT_mu
            values, J = np.array([value]), dvalue[None, :]
            tv = np.array([np.log10(mu), np.log10(transport.E) + value + np.log10(mu)])
            TJ = np.stack([mu_jac, mu_jac + dvalue])
            physical = np.full(1, np.nan)
            carrier_eligible = bool(low.diagnostics['single_electron_Hall_valid'] and high.diagnostics['single_electron_Hall_valid'])
            dg = dict(low.diagnostics, physical_state_available=False, endpoint_envelope_only=True,
                endpoint_low_log10_nH=raw_low, regression_lower_log10_nH=a, endpoint_high_log10_nH=b,
                Hall_detection_floor_intersection=True, lower_branch_derivative_regular=bool(raw_low != 13.),
                endpoint_high_single_electron=high.diagnostics['single_electron_Hall_valid'],
                readout_scope='log Hall interpolation between physical endpoints;no midpoint inventory or native mobility inferred')
        if source_kind == 'offset':
            values[0] += z
            tv[1] += z
            J[0, source_col] += 1.
            TJ[1, source_col] += 1.
        valid = bool(carrier_eligible and np.isfinite(values[0]) and values[0] >= 13.)
        if require_observation and not valid:
            raise ValueError('Outside implemented single-electron Hall/detection domain')
        dg.update(source_kind=source_kind, source_shift=z, share_basis=self.basis,
            share_coefficients=w[2:].tolist(), Tf_K=T, temperature_mode='effective',
            temperature_scope='effective endpoint;not measured thermal history',
            readout=self.readout, same_physical_state_for_output=bool(self.readout == 'physical' and (source_kind != 'offset' or z == 0)),
            observation_domain_valid=valid, observation_offset_dex=z if source_kind == 'offset' else 0.,
            numerical_observation_continuation=not valid,
            source_share_is_inventory_update=bool(self.readout == 'physical'),
            learning_scope='shared feasible excess fraction' if self.readout == 'physical' else 'matched physical-endpoint regression control')
        names = ('log10_dose_cm3', 'log10_Tobs_K') + (('log10_H_cm3',) if row.H_known else ()) + ('growth_T_C',)
        return RelativeInventoryPrediction(values, J[:, :n], J[:, source_col], physical, dg,
            tv, TJ[:, :n], TJ[:, source_col], J[:, input_col:], TJ[:, input_col:], names)
