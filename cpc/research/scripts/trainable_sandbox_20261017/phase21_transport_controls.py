"""Fixed chemical-compensation transport comparisons at the same predicted nH.

Both qA1/qA3 are reported; no target-mobility label chooses a charge or head.
The source prediction, shared weights and scalar adaptation are unchanged.
"""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from sandbox.chemical_transport_control import chemical_hall_transport
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.coherent_transport import fermi_transport
from csv_io_v2 import read_result_csv as read
import phase21_prediction as s
OUT=s.OUT/'chemical_transport_controls'


def freeze():
    paths=(Path(__file__),ROOT/'src/sandbox/chemical_transport_control.py',ROOT/'src/sandbox/coherent_transport.py')
    p=dict(created_utc=s.energy.now(),code_hashes={str(p.relative_to(ROOT)):s.energy.sha(p) for p in paths},
        design='post-prediction controls only;preserve eachpredictednH;no source/shared refit',
        charges=[1,3],background_cm3=1e14,extra_mobility_cm2Vs=205.,
        closure='D=C+H+1e14,ND=max(D,n),Nq2=ND+qA*max(D-n,0);nH=n/rH inverted',
        excess='ifn>D,unobserveddonor excessND-D isexplicitlyreported;not aclosedatomresolveddefectstate',
        input_scope='alloldmu predictionsownT;allLeedy scenarios/policies/pressures;allnumericmodel outputs',
        score='sameprimaryqueryandcommonsets;fullcoverage,allqchoicesreported;no bestheadpostselection',
        interpretation='heuristicchemical/samestatisticscomparison,not microscopiccharge-state identification orRTAvalidation',
        extra_mobility_supervision=0,new_training_labels=0,goal_complete=False)
    path=OUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text());assert old['code_hashes']==p['code_hashes']
    else:s.energy.write(path,p)


def gates():
    freeze();profile=published_coherent_profile();checks=[]
    for T in (200.,300.,450.):
        for fraction in (.01,.3,1.5):
            nh=np.log10(1e19*fraction);results=[]
            for charge in (1,3):
                r=chemical_hall_transport(nh,1e19,1e17,T,acceptor_charge=charge,profile=profile);results.append(r)
                n=10**r['log_free_n'];D=1e19+1e17+1e14
                expected=max(D,n)+charge*max(D-n,0.)
                # Independently reconstruct the compensation charge count.
                ND=max(D,n);NA=max(D-n,0.)/charge
                neutrality_error=abs(ND-charge*NA-n)/max(ND,1.)
                p=fermi_transport(r['log_free_n'],np.log10(r['Nq2_cm3']),T,Nc_ref_cm3=profile.Nc_ref,T_ref_K=profile.T_ref,
                    mass_ratio=profile.electron_mass_ratio,extra_mobility_cm2Vs=205.,hall=True)
                err=max(abs(r['Nq2_cm3']/expected-1),neutrality_error,abs(p.values[0]-nh),abs(p.values[1]-r['log_muH']))
                checks.append(dict(T=T,fraction=fraction,qA=charge,error=float(err),passed=bool(err<2e-7)))
            if fraction==1.5:
                err=abs(results[0]['log_muH']-results[1]['log_muH']);checks.append(dict(test='no_compensation_charge_limit',error=err,passed=err<2e-7))
    result=dict(passed=all(r['passed'] for r in checks),checks=len(checks),rows=checks,new_material_measurements=0)
    s.energy.write(OUT/'gates.json',result);print('chemicaltransport gates',result['passed'],len(checks),flush=True);assert result['passed']


def evaluate():
    freeze();assert json.loads((OUT/'gates.json').read_text())['passed'];profile=published_coherent_profile()
    cp=s.OUT/'cohort_properties/predictions.csv';ep=s.OUT/'external_leedy/predictions.csv'
    s.energy.write(OUT/'execution_inputs.json',dict(created_utc=s.energy.now(),input_hashes={str(p.relative_to(ROOT)):s.energy.sha(p) for p in (cp,ep)}))
    cohort=read(cp);cohort=cohort[cohort.endpoint.eq('muH')];targets=s.parent.inputs.TARGETS.set_index('obs_id');rows=[]
    for r in cohort.itertuples():
        t=targets.loc[r.obs_id]
        base=dict(dataset='old_mu',method=r.method,budget=int(r.budget),sample_id=r.sample_id,obs_id=r.obs_id,family=r.family,
            role=r.role,kind=r.kind,common_query=bool(r.common_query),T_K=float(r.T_K),observed_log_mu=float(r.observed),
            original_log_mu=float(r.predicted) if r.numeric else None,predicted_log_nH=float(r.predicted_log_nH) if r.numeric else None)
        rows += heads(base,bool(r.numeric),float(t.incorporated_cm3),float(t.H_cm3),profile)
    ext=read(ep)
    for r in ext.itertuples():
        base=dict(dataset='Leedy',method=r.method,budget=int(r.budget),sample_id=r.sample_id,obs_id=r.id,family='10.1063/1.5047214',role=r.role,kind='RT',
            common_query=bool(r.common_query),T_K=300.,observed_log_mu=float(r.observed_log_mu),original_log_mu=float(r.predicted_log_mu) if r.numeric else None,
            predicted_log_nH=float(r.predicted_log_nH) if r.numeric else None,scenario=r.scenario,policy=r.policy,pressure_atm=float(r.pressure_atm),uniform_scalar_input=bool(r.uniform_scalar_input))
        rows += heads(base,bool(r.numeric),float(r.incorporated_cm3),0.,profile)
    d=pd.DataFrame(rows);d.to_csv(OUT/'predictions.csv',index=False);by=[]
    old=d[d.dataset.eq('old_mu')]
    for scope in ('query','common_query'):
        q=old[old.role.eq('query')] if scope=='query' else old[old.common_query]
        for keys,g in q.groupby(['method','budget','qA','kind','family','sample_id']):
            r=dict(zip(['method','budget','qA','kind','family','sample_id'],keys));r.update(scope=scope,reads=len(g),numeric=int(g.numeric.sum()),
                MAE=float(abs(g.predicted_log_mu-g.observed_log_mu).mean()) if g.numeric.all() else np.nan);by.append(r)
    by=pd.DataFrame(by);by.to_csv(OUT/'old_by_specimen.csv',index=False);family=[]
    for keys,g in by.groupby(['method','budget','qA','kind','scope','family']):
        r=dict(zip(['method','budget','qA','kind','scope','family'],keys));r.update(specimens=len(g),reads=int(g.reads.sum()),numeric=int(g.numeric.sum()),
            MAE=float(g.MAE.mean()) if g.MAE.notna().all() else np.nan);family.append(r)
    f=pd.DataFrame(family);summary=[]
    for keys,g in f.groupby(['method','budget','qA','kind','scope']):
        r=dict(zip(['method','budget','qA','kind','scope'],keys));r.update(families=len(g),specimens=int(g.specimens.sum()),reads=int(g.reads.sum()),numeric=int(g.numeric.sum()),
            MAE=float(g.MAE.mean()) if g.MAE.notna().all() else np.nan);summary.append(r)
    pd.DataFrame(summary).to_csv(OUT/'old_summary.csv',index=False);extsummary=[]
    for keys,g in d[d.dataset.eq('Leedy')&d.role.eq('query')].groupby(['method','budget','qA','scenario','policy','pressure_atm']):
        for scope in ('all_query','common3'):
            q=g if scope=='all_query' else g[g.common_query];r=dict(zip(['method','budget','qA','scenario','policy','pressure_atm'],keys));r.update(scope=scope,queries=len(q),numeric=int(q.numeric.sum()),
                MAE=float(abs(q.predicted_log_mu-q.observed_log_mu).mean()) if q.numeric.all() else np.nan);extsummary.append(r)
    pd.DataFrame(extsummary).to_csv(OUT/'external_summary.csv',index=False)
    s.energy.write(OUT/'audit.json',dict(passed=True,rows=len(d),numeric_rows=int(d.numeric.sum()),qA_choices=[1,3],
        rows_requiring_heuristic_extra_donors=int(d.implied_extra_donor_cm3.fillna(0.).gt(0.).sum()),
        source_predictions_unchanged=True,new_mu_training_labels=0,no_head_postselection=True,independent_sigma_labels=0,goal_complete=False))
    print('transport controls',len(d),'rows',int(d.numeric.sum()),'numeric',flush=True)


def heads(base,numeric,C,H,profile):
    rows=[]
    for q in (1,3):
        r=dict(base,qA=q,incorporated_cm3=C,H_cm3=H,numeric=False)
        if numeric:
            try:
                a=chemical_hall_transport(base['predicted_log_nH'],C,H,base['T_K'],acceptor_charge=q,profile=profile)
                r.update(numeric=True,predicted_log_mu=a['log_muH'],predicted_log_sigma=a['log_sigma'],
                    implied_extra_donor_cm3=a['implied_extra_donor_cm3'],Nq2_cm3=a['Nq2_cm3'],inversion_error=a['inversion_error_dex'],error=None)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:r['error']=repr(exc)
        else:r['error']='upstreampredictionunavailable'
        rows.append(r)
    return rows


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--gates',action='store_true');a=ap.parse_args();gates() if a.gates else evaluate()
