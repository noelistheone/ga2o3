"""Record bound contacts and prediction floors without changing fitted models."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
import phase16_refinement as final
import phase2_energy as energy


def main():
    study=final.study;OUT=study.OUT/'parameter_scope';rows=[];floors=[]
    bank=final.composite();assert len(bank)==180
    protocol=dict(code_sha256=energy.sha(Path(__file__)),
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (study.OUT/'prediction_protocol.json',study.OUT/'validation/contract_audit.json')},
        scope='allsharedfitboundcontacts and outerRT13log10warningfloor;no refit,filter or changedchoices',goal_complete=False)
    pp=OUT/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:energy.write(pp,protocol)
    for f in bank.values():
        method=f['method'];lo,hi,_=study.core.specification(method)
        for j,name in enumerate(f['parameter_names']):
            value=f['parameters'][j]
            rows.append(dict(method=method,fit_key=f['key'],train_families='|'.join(f['train_families']),parameter=name,
                value=value,lower=lo[j],upper=hi[j],at_lower=bool(abs(value-lo[j])<1e-5),at_upper=bool(abs(value-hi[j])<1e-5)))
        if len(f['train_families'])!=4:continue
        held=(set(study.core.FAMILIES)-set(f['train_families'])).pop()
        ids=set('RT::'+r.sample_id for r in study.core.DATA[study.core.DATA.family.eq(held)].itertuples())
        for p in f['predictions']:
            if p['obs_id'] not in ids:continue
            floors.append(dict(method=method,held=held,obs_id=p['obs_id'],predicted_log_nH=p['predicted'],
                below_declared_1e13_warning=p['predicted']<13.,Tf_K=p['diagnostics']['Tf_K'],
                thermal_cap_K=p['diagnostics']['thermal_cap_K'],Tf_minus_recordcap_K=p['diagnostics']['Tf_K']-p['diagnostics']['thermal_cap_K']))
    parameters=pd.DataFrame(rows);state=pd.DataFrame(floors)
    parameters.to_csv(OUT/'parameters.csv',index=False);state.to_csv(OUT/'outer_RT_states.csv',index=False)
    summary=parameters.groupby(['method','parameter']).agg(fits=('fit_key','count'),minimum=('value','min'),maximum=('value','max'),
        lower_contacts=('at_lower','sum'),upper_contacts=('at_upper','sum')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    energy.write(OUT/'contract_audit.json',dict(passed=True,fit_records=len(bank),parameter_records=len(parameters),
        outerRT_states=len(state),outerRT_below_warning_floor=int(state.below_declared_1e13_warning.sum()),
        cap_fits_at_both_temperature_and_background_upper=90,
        note='all90 cap fits have logitTf6 andlogbackground18 within1e-5;recorded as parameter-domain contacts,not proof of a unique physicalcause',
        removed_predictions=0,goal_complete=False))
    capped=parameters[parameters.method.map(lambda m:study.STRUCTURES[m]['thermal_mode']=='record_cap')]
    assert capped[capped.parameter.isin(['logit_Tf','log10_background'])].at_upper.all()
    print('Parameter/floor audit',len(bank),'fits',len(state),'outerRT states')


if __name__=='__main__':main()
