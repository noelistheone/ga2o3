"""Input-only state-family scan; labels are read for a separate final diagnostic.

No training or choice of a grid/model uses Hall query values. The reported
envelope is sampled, not a global reachable-set or connected-domain theorem.
"""
from pathlib import Path
from types import SimpleNamespace
import sys,json
import numpy as np
import pandas as pd
from scipy.special import logit
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.pool_share_model import PoolShareModel
from sandbox.relative_inventory_model import RelativeInventoryModel,L0,L1
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.context_saved_model import INPUT_NAMES
from csv_io_v2 import read_result_csv as read
import phase2_energy as e

OUT=e.BASE/'phase26_pool_share';PARENT=e.BASE/'phase25_shared_process'
F=np.array([.001,.01,.10,.25,.49,.50,.51,.65,.66,2/3,.67,.68,.74,.75,.76,.80,.85,.90,.95,.99,.999])
H=np.linspace(-6,6,len(F))


def main():
    assert not (OUT/'reachability/complete.json').exists(),'Do not overwrite completed diagnostics'
    gates=json.loads((OUT/'numerical_gates.json').read_text());assert gates['passed']
    for n,h in gates['code_hashes'].items():assert e.sha(ROOT/n)==h,n
    data=read(PARENT/'data/conditions33.csv');manifest=json.loads((PARENT/'delivery/predictor_bundle/manifest.json').read_text())
    contexts={}
    for name in ('DOI::full','cluster::full'):
        spec=manifest['presets'][name+'::FixedPhysical'];p=PARENT/'delivery/predictor_bundle/models'/(spec['model_key']+'.json')
        contexts[name]=dict(parameters=json.loads(p.read_text())['parameters'],path=str(p.relative_to(ROOT)),sha256=e.sha(p))
    code=[Path(__file__),ROOT/'src/sandbox/pool_share_model.py',ROOT/'src/sandbox/absolute_pool_share.py']
    e.write(OUT/'reachability/protocol_before.json',dict(created_utc=e.now(),contexts=contexts,
        fraction_nodes=F.tolist(),relative_log10_nodes=H.tolist(),nodes_per_family=len(F),
        grid_rationale='fixed fractions include constantcharge threshold neighborhoods;no Hall label chosen grid',
        modes=['effective','growth'],state_rows_planned=2*2*33*2*len(F),
        labels_scope='only after all state outputs written,describe fixed-label position in sampledvalid envelope',
        no_new_fits=True,no_hyperparameter_selection=True,retrospective=True,goal_complete=False,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in code},
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in [PARENT/'data/conditions33.csv',PARENT/'delivery/predictor_bundle/manifest.json',OUT/'numerical_gates.json']}))
    profile=published_coherent_profile();states=[];baselines=[];equivalence=[]
    # Strip labels and source identity from every actual model input.
    inputs=[dict(sample_id=r.sample_id,node={k:e.clean(getattr(r,k)) for k in INPUT_NAMES}) for r in data.itertuples()]
    for context,spec in contexts.items():
        for mode in ('effective','growth'):
            model=PoolShareModel(basis='constant',temperature_mode=mode,profile=profile);oldmodels={}
            for entry in inputs:
                row=SimpleNamespace(**entry['node']);pressure=row.scenario_pO2_atm;w=np.array(spec['parameters'],float)
                if mode=='growth':
                    frac=(np.log10(row.growth_T_C+273.15)-L0)/(L1-L0)
                    if not 0<frac<1:raise ValueError('Growth scenario cannot be represented by reference effective-temperature coordinates')
                    old_w=np.r_[logit(frac),w[1],0.];base_w=w[1:2]
                else:old_w=np.r_[w,0.];base_w=w
                if pressure not in oldmodels:oldmodels[pressure]=RelativeInventoryModel(dict(field='inventory'),profile=profile,pO2_atm=pressure)
                old=oldmodels[pressure];ref=old.point(row,old_w,source_kind='field',source_shift=0.,require_observation=False)
                dg=ref.diagnostics;U=dg['complex_reference_atom_cm3'];V=dg['isolated_reference_atom_cm3'];psi0=float(np.log(V)-np.log(U))
                baselines.append(dict(context=context,mode=mode,sample_id=entry['sample_id'],reference_share=V/(U+V),reference_logit=psi0,
                    reference_log_nH=float(ref.values[0]),reference_valid=dg['observation_domain_valid'],
                    root_residual=dg['root_residual_log'],Tf_K=dg['Tf_K']))
                try:
                    same=model.point(row,np.r_[base_w,psi0],source_kind='share',source_shift=0.,require_observation=False)
                    equivalence.append(dict(context=context,mode=mode,sample_id=entry['sample_id'],numeric=True,
                        value_error=float(np.max(abs(np.r_[same.values,same.transport_values]-np.r_[ref.values,ref.transport_values]))),
                        source_derivative_error=float(np.max(abs(np.r_[same.source_jacobian,same.transport_source_jacobian]-
                            np.r_[ref.source_jacobian,ref.transport_source_jacobian])/(1+abs(np.r_[ref.source_jacobian,ref.transport_source_jacobian]))))))
                except (ValueError,RuntimeError,ArithmeticError) as exc:
                    equivalence.append(dict(context=context,mode=mode,sample_id=entry['sample_id'],numeric=False,error=repr(exc)))
                for kind,nodes in [('absolute_share',logit(F)),('relative_odds',H)]:
                    for j,coordinate in enumerate(nodes):
                        record=dict(context=context,mode=mode,sample_id=entry['sample_id'],kind=kind,node=j,coordinate=float(coordinate),numeric=False,valid=False)
                        try:
                            p=model.point(row,np.r_[base_w,coordinate],source_kind='share',source_shift=0.,require_observation=False) if kind=='absolute_share' else old.point(
                                row,np.r_[old_w[:2],coordinate],source_kind='field',source_shift=0.,require_observation=False)
                            d=p.diagnostics
                            record.update(numeric=True,valid=bool(d['observation_domain_valid']),log_nH=float(p.values[0]),
                                log_muH=float(p.transport_values[0]),log_sigma=float(p.transport_values[1]),
                                share=float(d['isolated_atom_fraction_after']),dlogn_dsource=float(p.source_jacobian[0]),
                                foreign_error=float(d['foreign_atom_relative_error']),root_residual=float(d['root_residual_log']))
                        except (ValueError,RuntimeError,ArithmeticError) as exc:record['error']=repr(exc)
                        states.append(record)
            print('phase26 state families scanned',context,mode,flush=True)
    e.csvwrite(OUT/'reachability/states.csv',states);e.csvwrite(OUT/'reachability/reference.csv',baselines)
    e.csvwrite(OUT/'reachability/same_state_equivalence.csv',equivalence)
    # Model computation is now finished. Inspect the already-fixed labels only
    # to describe finite-grid coverage, not to fit or choose the model/grid.
    d=pd.DataFrame(states);labels=data.set_index('sample_id').target;envelopes=[]
    for key,g in d.groupby(['context','mode','sample_id','kind']):
        valid=g[g.valid.eq(True)&g.numeric.eq(True)];y=float(labels[key[2]])
        lo=float(valid.log_nH.min()) if len(valid) else None;hi=float(valid.log_nH.max()) if len(valid) else None
        envelopes.append(dict(context=key[0],mode=key[1],sample_id=key[2],kind=key[3],nodes=len(g),numeric=int(g.numeric.sum()),valid=len(valid),
            observed=y,sampled_valid_min=lo,sampled_valid_max=hi,observed_within_sampled_extrema=bool(lo is not None and lo<=y<=hi),
            no_continuum_or_connected_domain_certificate=True))
    e.csvwrite(OUT/'reachability/sampled_envelopes.csv',envelopes)
    summary=[];ee=pd.DataFrame(envelopes)
    for key,g in ee.groupby(['context','mode','kind']):
        summary.append(dict(context=key[0],mode=key[1],kind=key[2],conditions=len(g),
            all_nodes_numeric=int((g.numeric==len(F)).sum()),any_valid=int((g.valid>0).sum()),
            labels_within_sampled_extrema=int(g.observed_within_sampled_extrema.sum()),
            not_a_prediction_accuracy_oracle=True))
    e.csvwrite(OUT/'reachability/summary.csv',summary)
    value=dict(passed=bool(len(states)==5544 and all(r['numeric'] and r['value_error']<2e-8 and r['source_derivative_error']<3e-5 for r in equivalence)),
        rows=len(states),numeric=int(d.numeric.sum()),valid=int(d.valid.sum()),equivalence_cases=len(equivalence),
        maximum_same_state_value_error=max([r.get('value_error',np.inf) for r in equivalence]),
        maximum_same_state_source_derivative_error=max([r.get('source_derivative_error',np.inf) for r in equivalence]),
        critical_theory='same physical state has the same source derivative;different cross-input coupling is the tested hypothesis',
        new_shared_fits=0,goal_complete=False)
    e.write(OUT/'reachability/complete.json',value);print('phase26 state scan',value,flush=True)
    if not value['passed']:raise RuntimeError('Preserve scan before resolving representation/equivalence failures')


if __name__=='__main__':main()
