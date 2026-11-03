"""Replay external inputs with a label-erasure schema adapter; no retraining."""
from pathlib import Path
import sys,numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import overlay_nH_controls_v1 as original

OUT=original.OUT/'external_schema_replay_v2'


def main():
    assert not (OUT/'protocol_before.json').exists()
    assert (original.OUT/'complete.json').exists()
    source=c.BASE/'external_rate_cohort_v2/scenarios24.csv';table=c.read(source)
    fits=[c.document(p) for p in (original.OUT/'fits').glob('*.json') if c.document(p)['held'] is None]
    assert len(fits)==2 and all(f['selected_success'] and f['training_valid'] for f in fits)
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),
        repair='external schema lacks old measured-label aliases carrier_cm3/observed_log_nH;add NaN aliases for existing label erasure,also erase nH_cm3',
        model_inputs_unchanged=True,new_shared_fits=0,old_invalid_outputs_preserved=True,
        code_hashes=c.file_hashes([Path(__file__),Path(original.__file__)]),input_hashes=c.file_hashes([source,original.OUT/'complete.json'])))
    original_inputs=table[['case_id','incorporated_cm3','H_cm3','H_known','growth_T_C','scenario_pO2_atm','target','mu_cm2Vs']].copy()
    table['carrier_cm3']=np.nan;table['observed_log_nH']=np.nan;table['nH_cm3']=np.nan
    assert original_inputs.equals(table[original_inputs.columns])
    rows=[]
    for fit in fits:
        rows.extend(original.predict(table,fit,'FixedNHReference' if fit['method']=='OverlayReference' else 'LearnedNHInventory','external3_conditional'))
    p=pd.DataFrame(rows);p.to_csv(OUT/'predictions.csv',index=False);summary=[]
    for variant in p.variant.unique():
        q=p[p.variant.eq(variant)];item=dict(variant=variant,scope='query',rows=len(q),numeric=int(q.numeric.eq(True).sum()),valid=int(q.valid.eq(True).sum()))
        for endpoint in ['nH','mu','rho']:item[endpoint+'_MAE']=c.group_score(q.to_dict('records'),endpoint,paired=endpoint!='nH',group_equal=False)
        summary.append(item)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    c.write(OUT/'complete.json',dict(created_utc=c.e.now(),rows=len(p),actual_physical_calls=len(p),new_fits=0,
        numeric=int(p.numeric.eq(True).sum()),valid=int(p.valid.eq(True).sum()),fixed_condition_inputs_preserved=True))
    print(summary)


if __name__=='__main__':main()
