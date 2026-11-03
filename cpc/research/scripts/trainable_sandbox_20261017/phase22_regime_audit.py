"""Descriptive source/temperature decomposition of frozen predictions, no refitting."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from csv_io_v2 import read_result_csv as read
import phase2_energy as energy
BASE=energy.BASE;PARENT=BASE/'phase21_curve_response';OUT=BASE/'phase22_predictive_inference'


def main():
    paths=[BASE/'dataset_v1.csv',PARENT/'cohort_properties/predictions.csv',
        PARENT/'chemical_transport_controls/predictions.csv',BASE/'artifact_manifest_phase21.json']
    manifest=json.loads(paths[-1].read_text());assert manifest['goal_status']=='active'
    net=ROOT.parent/'Ga2O3-Net'
    sources=[dict(doi='10.1002/aelm.202300679',path=net/'extra-paper/v56_markdown/ec3fc6d0e0ba6300.md',
        depth='local published full-text extraction: Table1, SIMS/Hall, microstructure and ExperimentalSection; no original PDF/raw Hall',
        scope='100nm/sapphire, low activation and twins/stacking/grain discussion; not a uniquely demonstrated scattering mechanism'),
        dict(doi='10.1063/5.0101132',path=net/'extra-paper/doping_extracted/doping/si/072108_1_online.pdf',
        depth='local full-text methods/TableI/Fig4 and author-hosted primary PDF rechecked; figure/table visual inspection separate',
        scope='three same-specimen SIMS/Hall curves; below reported Mott scale, freeze out; 1T,300to20K; not high-field multichannel identification')]
    protocol=dict(created_utc=energy.now(),previous_goal_turn='progress: phase21 complete prediction/controls checkpoint and parent freeze',
        design='retrospective descriptive decomposition; no new model, fitting, labels, selection or data exclusions',
        code_hashes={str(Path(__file__).relative_to(ROOT)):energy.sha(Path(__file__))},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in paths},
        fixed_windows_K={'low':'T<60','middle':'60<=T<=150','high':'T>150'},
        methods=['OriginalHall','FieldCurveSafe','MedianActivation','DoseAffine'],budget=0,
        role_scope='all published existing points; query and anchor/guard counted separately; no favorable-error filtering',
        goal_complete=False)
    dest=OUT/'regime_audit';dest.mkdir(parents=True,exist_ok=True)
    p=dest/'protocol.json'
    if p.exists():assert json.loads(p.read_text())['code_hashes']==protocol['code_hashes']
    else:energy.write(p,protocol)
    energy.write(dest/'sources.json',dict(records=[dict(**{k:v for k,v in s.items() if k!='path'},
        path=str(s['path']),sha256=energy.sha(s['path'])) for s in sources],new_samples=0,raw_instrument_records=False))
    rt=read(paths[0]);d=read(paths[1]);d=d[d.budget.eq(0)&d.method.isin(protocol['methods'])].copy()
    assert d.numeric.all();d['signed_error']=d.predicted-d.observed;d['absolute_error']=abs(d.signed_error)
    d['temperature_window']=np.where(d.T_K<60,'low',np.where(d.T_K<=150,'middle','high'))
    d.to_csv(dest/'pointwise.csv',index=False)
    rows=[]
    for key,g in d.groupby(['method','family','sample_id','endpoint','kind','temperature_window','role'],dropna=False):
        r=dict(zip(['method','family','sample_id','endpoint','kind','temperature_window','role'],key));r.update(reads=len(g),
            T_min=float(g.T_K.min()),T_max=float(g.T_K.max()),MAE=float(g.absolute_error.mean()),
            bias=float(g.signed_error.mean()),maximum_error=float(g.absolute_error.max()),
            observed_min=float(g.observed.min()),observed_max=float(g.observed.max()),
            predicted_min=float(g.predicted.min()),predicted_max=float(g.predicted.max()))
        rows.append(r)
    pd.DataFrame(rows).to_csv(dest/'by_specimen_window.csv',index=False)
    chem=read(paths[2]);c=chem[chem.dataset.eq('old_mu')&chem.budget.eq(0)&chem.method.isin(protocol['methods'])].copy()
    assert c.numeric.all();c['temperature_window']=np.where(c.T_K<60,'low',np.where(c.T_K<=150,'middle','high'))
    c['absolute_error']=abs(c.predicted_log_mu-c.observed_log_mu);c['signed_error']=c.predicted_log_mu-c.observed_log_mu
    c.to_csv(dest/'chemical_pointwise.csv',index=False)
    heads=[]
    for key,g in c.groupby(['method','family','sample_id','qA','kind','temperature_window','role'],dropna=False):
        r=dict(zip(['method','family','sample_id','qA','kind','temperature_window','role'],key));r.update(reads=len(g),
            MAE=float(g.absolute_error.mean()),bias=float(g.signed_error.mean()),
            minimum_log_mu=float(g.predicted_log_mu.min()),maximum_log_mu=float(g.predicted_log_mu.max()))
        heads.append(r)
    pd.DataFrame(heads).to_csv(dest/'chemical_by_specimen_window.csv',index=False)
    main=d[d.endpoint.eq('nH')&d.kind.eq('RT')].merge(rt[['sample_id','incorporated_cm3','growth_T_C','substrate','thickness_um','method']],
        on='sample_id',validate='many_to_one',suffixes=('','_growth'))
    main['observed_activation']=10**main.observed/main.incorporated_cm3
    main['predicted_activation']=10**main.predicted/main.incorporated_cm3
    main.to_csv(dest/'RT_state_errors.csv',index=False)
    invariant=[]
    for method in protocol['methods']:
        q=d[d.method.eq(method)]
        for endpoint,kind,count in [('nH','curve',142),('muH','curve',54),('muH','RT',5)]:
            n=len(q[q.endpoint.eq(endpoint)&q.kind.eq(kind)])
            invariant.append(dict(method=method,endpoint=endpoint,kind=kind,reads=n,expected=count,passed=n==count))
    assert all(x['passed'] for x in invariant),invariant
    energy.write(dest/'audit.json',dict(passed=True,rows=len(d),chemical_rows=len(c),invariants=invariant,
        original_query_set_unchanged=True,new_fitting=0,new_training_labels=0,new_material_measurements=0,goal_complete=False))
    print('regime audit',len(d),'point rows',len(c),'chemical rows; no new fitting',flush=True)
    print(pd.DataFrame(rows).query("method=='OriginalHall' and endpoint=='nH' and kind=='curve'")
        [['sample_id','temperature_window','reads','MAE','bias']].to_string(index=False),flush=True)


if __name__=='__main__':main()
