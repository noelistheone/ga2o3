"""Inspect dilution and thermal interpretation without changing predictions."""
from pathlib import Path
from types import SimpleNamespace
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from sandbox.coherent_hall_model import CoherentHallModel
from sandbox.coherent_fermi import CoherentMaterialProfile
import phase2_energy as energy

OUT=energy.BASE/'phase15_coherent_hall';DEST=OUT/'state_scope'


def main():
    runtime=OUT/'research_runtime';inputs=(runtime/'tasks.json',runtime/'independent_outputs.json',runtime/'verification.json')
    assert json.loads(inputs[-1].read_text())['passed']
    protocol=dict(code_sha256=energy.sha(Path(__file__)),input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in inputs},
        scope='postfit diagnostics onall savedRTstates;not refit/filter or completed Born/material validitygate',
        concentration_note='parent sums/hostatom activities are proxies,not unique site-exclusion accounting',
        thermal_note='growth record is not a measuredcompletepeakthermalhistory;Tf is effective,not unique freeze temperature')
    pp=DEST/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:energy.write(pp,protocol)
    tasks=json.loads(inputs[0].read_text());outputs=json.loads(inputs[1].read_text());z={a['task_id']:a['shift'] for a in outputs['adaptations']}
    expected={(r['task_id'],r['obs_id']):r for r in outputs['rows']};rows=[];errors=[]
    for task in tasks:
        b=json.loads(Path(task['bundle']).read_text());profile=CoherentMaterialProfile.from_dict(b['material_profile'])
        model=CoherentHallModel(b['definition'],profile=profile,pO2_atm=b['pO2_atm'])
        for p in task['points']:
            r=SimpleNamespace(**p['input']);a=model.point(r,b['parameters'],source_kind=b['source_kind'],source_shift=z[task['task_id']])
            wanted=expected[task['task_id'],p['obs_id']];assert wanted['error'] is None
            errors.append(float(max(abs(a.values-np.array(wanted['values'])))))
            elements=(r.element,'H') if r.H_known else (r.element,);layer=model.layers[elements];state=layer.last_state
            dm=layer.db.dm[layer.first];native=np.all(dm[:,2:]==0,axis=1);frac=state['totals']/layer.db.prefactor[layer.first]
            native_events=float(np.sum(state['totals'][native])/profile.N_site)
            absGa=float(abs(dm[native,0])@state['totals'][native]/(2*profile.N_site))
            absO=float(abs(dm[native,1])@state['totals'][native]/(3*profile.N_site))
            bound=float(state['totals']@state['cold_susceptibility']);free=a.diagnostics['coherent_transport']['n_screen_cm3']
            rows.append(dict(task_id=task['task_id'],method=b['method'],base_method=b['base_method'],held=b['held'],budget=b['budget'],
                obs_id=p['obs_id'],max_parent_fraction=float(max(frac)),max_native_parent_fraction=float(max(frac[native])),
                native_events_per_formula_unit=native_events,absolute_native_Ga_activity_fraction=absGa,absolute_native_O_activity_fraction=absO,
                bound_equilibrium_response_over_free_screen=bound/free,effective_Tf_K=a.diagnostics['Tf_K'],recorded_growth_K=r.growth_T_C+273.15,
                effective_Tf_minus_recorded_growth_K=a.diagnostics['Tf_K']-(r.growth_T_C+273.15)))
    d=pd.DataFrame(rows);d.to_csv(DEST/'state_diagnostics.csv',index=False)
    assert len(d)==1566 and max(errors)<1e-6
    energy.write(DEST/'contract_audit.json',dict(passed=True,states=len(d),maximum_replay_error=max(errors),
        max_native_parent_fraction=float(d.max_native_parent_fraction.max()),
        native_parent_above_10percent_rows=int(d.max_native_parent_fraction.gt(.1).sum()),
        max_native_events_per_formula_unit=float(d.native_events_per_formula_unit.max()),
        max_absolute_native_Ga_activity_fraction=float(d.absolute_native_Ga_activity_fraction.max()),
        max_absolute_native_O_activity_fraction=float(d.absolute_native_O_activity_fraction.max()),
        max_bound_response_over_free_screen=float(d.bound_equilibrium_response_over_free_screen.max()),
        effective_Tf_above_growth_record_rows=int(d.effective_Tf_minus_recorded_growth_K.gt(0).sum()),
        max_effective_Tf_growth_difference_K=float(d.effective_Tf_minus_recorded_growth_K.max()),
        changed_predictions=0,new_fit_jobs=0,material_Born_or_site_exclusion_validated=False,goal_complete=False))
    print(json.dumps(json.loads((DEST/'contract_audit.json').read_text()),indent=2))


if __name__=='__main__':main()
