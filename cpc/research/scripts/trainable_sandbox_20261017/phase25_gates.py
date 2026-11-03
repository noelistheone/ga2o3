"""Independent derivative and atom-budget gates for conditional inventory learning."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
from sandbox.context_inventory_model import ContextInventoryModel, bounded_field
from sandbox.relative_inventory_model import RelativeInventoryModel
from sandbox.coherent_fermi import published_coherent_profile
import phase25_data as d
import phase2_energy as e


def main():
    table = d.table()
    old = table[table.sample_id.str.endswith('::C')].iloc[0]
    new = table[table.sample_id.eq(d.LEEDY + '::target-1')].iloc[0]
    rows = []
    def check(name, value, tolerance):
        value = float(value)
        rows.append(dict(name=name, value=value, tolerance=tolerance, passed=bool(value <= tolerance)))
    def vector(a):
        return np.r_[a.values, a.transport_values]
    max_chain_omission = 0.
    for kind in ('inventory', 'output'):
        for basis in ('none', 'constant', 'dose', 'process'):
            m = ContextInventoryModel(basis=basis, field=kind, profile=published_coherent_profile())
            for j, record in enumerate((old, new)):
                row = SimpleNamespace(**record.to_dict())
                n = len(m.parameter_names)
                w = np.r_[.5, 16., np.array([.6, .4, -.2, .1])[:n-2]]
                source = 'field' if kind == 'inventory' else 'offset'
                a = m.point(row, w, source_kind=source, source_shift=.2, require_observation=False)
                analytic = np.r_[a.jacobian, a.transport_jacobian]
                numeric = []
                for k in range(n):
                    delta = np.zeros(n)
                    delta[k] = 1e-4
                    p = m.point(row, w+delta, source_kind=source, source_shift=.2, require_observation=False)
                    q = m.point(row, w-delta, source_kind=source, source_shift=.2, require_observation=False)
                    numeric.append((vector(p)-vector(q))/(2e-4))
                numeric = np.array(numeric).T
                label = f'{kind}/{basis}/{j}'
                check(label+'/parameters', np.max(np.abs(numeric-analytic)/(1+np.abs(analytic))), 3e-5)
                analytic_input = np.r_[a.input_jacobian, a.transport_input_jacobian]
                cols = []
                for input_name in a.input_names:
                    p = SimpleNamespace(**vars(row)); q = SimpleNamespace(**vars(row))
                    if input_name == 'growth_T_C':
                        step = .02; p.growth_T_C += step; q.growth_T_C -= step
                    else:
                        step = 1e-4
                        name = {'log10_dose_cm3':'incorporated_cm3', 'log10_Tobs_K':'Tobs_K', 'log10_H_cm3':'H_cm3'}[input_name]
                        setattr(p, name, getattr(p,name)*10**step)
                        setattr(q, name, getattr(q,name)*10**(-step))
                    cols.append((vector(m.point(p,w,source_kind=source,source_shift=.2,require_observation=False)) -
                        vector(m.point(q,w,source_kind=source,source_shift=.2,require_observation=False)))/(2*step))
                numeric_input = np.array(cols).T
                check(label+'/inputs', np.max(np.abs(numeric_input-analytic_input)/(1+np.abs(analytic_input))), 3e-5)
                p = m.point(row,w,source_kind=source,source_shift=.2001,require_observation=False)
                q = m.point(row,w,source_kind=source,source_shift=.1999,require_observation=False)
                analytic_source = np.r_[a.source_jacobian, a.transport_source_jacobian]
                check(label+'/source', np.max(np.abs((vector(p)-vector(q))/.0002-analytic_source)/(1+np.abs(analytic_source))), 3e-5)
                # A zero conditional field must reproduce the existing physical
                # point, including each condition's declared oxygen scenario.
                zero = w.copy(); zero[2:] = 0.
                z = m.point(row,zero,require_observation=False)
                raw = RelativeInventoryModel(dict(field=kind), profile=published_coherent_profile(), pO2_atm=row.scenario_pO2_atm)
                r = raw.point(row,np.r_[w[:2],0.],require_observation=False)
                check(label+'/zero_replay', np.max(np.abs(vector(z)-vector(r))), 1e-9)
                # Reconstruct atom-share odds and budgets from actual saved
                # parent populations, independently of the chain-rule wrapper.
                a = m.point(row,w,source_kind=source,source_shift=.2,require_observation=False)
                engine = m.engine_for(row); layer, solver = engine.layer_for(row)
                inv = solver.last_inventory; ref = solver._reference_cache[1]
                cx = solver.complex; iso = solver.receivers; kx = solver.dopant_count
                U0 = float(kx[cx]@ref.totals[cx]); V0 = float(ref.totals[iso].sum())
                U = float(kx[cx]@inv.totals[cx]); V = float(inv.totals[iso].sum())
                h = 0. if kind == 'output' else a.diagnostics['context_field_value'] + .2
                check(label+'/odds', abs(np.log(V/U)-np.log(V0/U0)-np.log(10.)*h), 1e-10)
                check(label+'/nonnegative', max(0.,-float(inv.totals.min())), 0.)
                check(label+'/atom_budget', abs(U+V-U0-V0)/(U0+V0), 1e-12)
                check(label+'/foreign_budget', a.diagnostics['foreign_atom_relative_error'], 3e-8)
                check(label+'/native_unchanged', np.max(np.abs(inv.totals[solver.native]-ref.totals[solver.native])), 0.)
                host = layer.db.dm[layer.first,:2].T@(inv.totals-ref.totals)
                check(label+'/O_budget', abs(host[1])/(U0+V0), 1e-12)
                if basis in ('dose','process'):
                    fixed = engine.point(row,np.r_[w[:2],a.diagnostics['context_field_value']],
                        source_kind=source,source_shift=.2,require_observation=False)
                    max_chain_omission = max(max_chain_omission,abs(a.input_jacobian[0,0]-fixed.input_jacobian[0,0]))
    check('chain_term_is_not_silently_zero', 0. if max_chain_omission > 1e-3 else 1., 0.)
    code = [Path(__file__), ROOT/'src/sandbox/context_inventory_model.py', ROOT/'src/sandbox/relative_inventory_model.py',
        ROOT/'src/sandbox/relative_inventory.py', ROOT/'src/sandbox/fixed_inventory_charge.py']
    payload = dict(passed=all(r['passed'] for r in rows), checks=len(rows), rows=rows,
        maximum_scaled_derivative_error=max(r['value'] for r in rows if r['name'].endswith(('/parameters','/inputs','/source'))),
        maximum_missing_chain_term=max_chain_omission, no_property_training_done=True, goal_complete=False,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in code})
    e.write(d.OUT/'numerical_gates.json',payload)
    print('phase25 gates',payload['passed'],payload['checks'],payload['maximum_scaled_derivative_error'],flush=True)
    if not payload['passed']:
        raise RuntimeError('Context inventory numerical gate failed; preserve this result before any repair')


if __name__ == '__main__':
    main()
