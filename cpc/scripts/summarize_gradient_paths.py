"""Recompute the gradient-path figure/table and equal-budget acquisition numbers."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
V = BASE/'verification/gradient_paths_20261031'
records = pd.read_csv(V/'coordinates.csv')
stored = pd.read_csv(V/'settings.csv')
fd = pd.read_csv(V/'finite_differences.csv')
checks = []

def check(name, passed):
    checks.append(dict(name=name, passed=bool(passed)))

check('eight_settings', len(stored)==8)
check('all_energetic_coordinates', len(records)==208)
check('two_steps_two_derivatives', len(fd)*2==832)
check('unique_coordinate_conditions', not records.duplicated(['element','Tf_K','C_cm3','label']).any())
for key, g in records.groupby(['element','Tf_K','C_cm3']):
    s = stored.set_index(['element','Tf_K','C_cm3']).loc[key]
    norm = np.linalg.norm(g.full)
    omission = np.linalg.norm(g.full-g.occupation)/norm
    check(str(key)+' norm', np.isclose(norm,s.full_norm,rtol=1e-12,atol=1e-15))
    check(str(key)+' omission', np.isclose(omission,s.omitted_inventory_relative_norm,rtol=1e-10))
    check(str(key)+' decomposition', np.max(abs(g.full-g.occupation-g.inventory))<1e-8)
for row in fd.itertuples():
    check('FD '+str((row.element,row.Tf_K,row.C_cm3,row.label,row.step)),
          abs(row.full-row.full_FD)/max(1.,abs(row.full),abs(row.full_FD))<3e-4 and
          abs(row.fixed_inventory-row.fixed_inventory_FD)/max(1.,abs(row.fixed_inventory),abs(row.fixed_inventory_FD))<3e-4)

lines=[r'\begin{table}[htbp]\centering',
       r'\caption{Energetic Jacobians with and without the preparation-inventory pathway.}\label{tab:Sgradientpaths}',
       r'\begin{tabular}{llrrr}\toprule',
       r'Dopant & $T_{\rm f}$ (K) & $C_X$ ($\mathrm{cm}^{-3}$) & $\|J_{\rm full}\|_2$ (dex/eV) & $\epsilon_J$ (\%) \\ \midrule']
for r in stored.itertuples():
    dose=r'$3\times10^{17}$' if r.C_cm3<1e18 else r'$3\times10^{19}$'
    norm=f'{r.full_norm:.5f}' if r.full_norm>.01 else f'${r.full_norm/1e-5:.3f}\\times10^{{-5}}$'
    lines.append(f'{r.element} & {int(r.Tf_K)} & {dose} & {norm} & {100*r.omitted_inventory_relative_norm:.2f}'+r' \\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
(BASE/'tables/gradient_paths.tex').write_text('\n'.join(lines))

acq=pd.read_csv(BASE/'source_data/energy_acquisition_replays.csv')
sel=acq[acq.model.eq('extra_trees') & acq.budget.eq(2)]
summary=sel.groupby(['element','policy']).MAE.agg(['size','min','median','max'])
summary.to_csv(V/'equal_budget_acquisition.csv')
for element in ('Si','Sn'):
    check(element+' random budget coverage', summary.loc[(element,'random'),'size']==32)
    check(element+' random complete', sel[sel.element.eq(element)].complete.all())
result=dict(checks=len(checks),passed=sum(c['passed'] for c in checks),failed=[c for c in checks if not c['passed']],items=checks)
(V/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='items'},indent=2))
if result['failed']:
    raise RuntimeError('Summary checks failed')
