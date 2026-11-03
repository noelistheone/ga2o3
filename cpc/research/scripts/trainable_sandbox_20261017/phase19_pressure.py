"""Measured-oxygen scenario check, keeping all trained parameters frozen."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from csv_io_v2 import read_result_csv as read
from sandbox.property_predictor_v5 import DopedPropertyPredictor,ConstantStock
from sandbox.coherent_hall_model import CoherentHallModel
from sandbox.thermal_coherent_model import ThermalCoherentHallModel
from sandbox.donor_mixture_model import DonorMixtureModel
from sandbox.coherent_fermi import CoherentMaterialProfile
from sandbox.material_profile import MaterialProfile
from sandbox.parent_allocation import ParentAllocationLayer
from sandbox.kroger_db import load
import phase2_energy as energy

BASE=energy.BASE;OUT=BASE/'phase19_delivery';DEST=OUT/'oxygen_sensitivity';BUNDLE=OUT/'predictor_bundle_v5'


class PressureLayer:
    def __init__(self,layer,pressure):self.layer=layer;self.pressure=pressure
    def point(self,**kw):
        kw['pO2_atm']=self.pressure
        return self.layer.point(**kw)


def main():
    source=OUT/'leedy2018_v2';d=read(source/'samples.csv');manifest=json.loads((BUNDLE/'manifest.json').read_text())
    protocol=dict(created_utc=energy.now(),code_sha256=energy.sha(Path(__file__)),
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (BUNDLE/'manifest.json',source/'samples.csv',source/'protocol.json')},
        purpose='check whether reportedoxygenalone explains failure;no fitting/no newhyperparameterselection',
        scenarios={'old_unmeasured_pressure':1e-5,'reported_partial_pressure':.05*1.3/101325.},
        methods='all14nonempiricalpresets,zeroanchors,all5sourcepoints;stockaliasduplicatesretained',
        thermal='keepeffectiveTf,fixedenergies,background,allMLweights;onlyoxygeninputchanges',
        grade='retrospective fixed-parameter scenario,not calibratednewprocessmodel')
    pp=DEST/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())['code_sha256']==protocol['code_sha256']
    else:energy.write(pp,protocol)
    db=load();collection=DopedPropertyPredictor(db,BUNDLE);rows=[];parity=[]
    for method,spec in manifest['methods'].items():
        if spec['branch']=='empirical':continue
        for scenario,pressure in protocol['scenarios'].items():
            if spec['branch']!='energy':
                record=json.loads((BUNDLE/'models'/f"{spec['model_key']}.json").read_text());kind=record['kind']
                profile=(CoherentMaterialProfile if kind in ('coherent','thermal') else MaterialProfile).from_dict(record['profile'])
                cls={'legacy':DonorMixtureModel,'coherent':CoherentHallModel,'thermal':ThermalCoherentHallModel}.get(kind)
                model=cls(record['definition'],profile=profile,pO2_atm=pressure,db=db) if cls else ConstantStock(record,db)
                if not cls:
                    model.layers[('Si',)]=PressureLayer(ParentAllocationLayer(db,('Si',),profile=profile,extension=False),pressure)
            for r in d.itertuples():
                row=collection._row('Si',float(r.incorporated_cm3),300.,False,0.,550.)
                if spec['branch']=='energy':
                    result=collection.energy.predict('Si',row.incorporated_cm3,method=method,pO2_atm=pressure)
                    n=result['log10_nH_cm3'];dg=result['raw_Sandbox_diagnostics']
                else:
                    result=model.point(row,record['parameters']);n=float(result.values[0]);dg=result.diagnostics
                if scenario=='old_unmeasured_pressure':
                    old=collection.predict('Si',row.incorporated_cm3,method=method,growth_T_C=550.)
                    parity.append(abs(n-old['log10_nH_cm3']))
                rows.append(dict(method=method,scenario=scenario,sample_id=r.sample_id,
                    incorporated_cm3=r.incorporated_cm3,pO2_atm=pressure,predicted_log_nH=n,
                    observed_log_nH=float(np.log10(r.observed_nH_cm3)),
                    root_residual_log=dg.get('root_residual_log'),free_n_cm3=dg.get('free_n_cm3')))
    result=pd.DataFrame(rows);result.to_csv(DEST/'predictions.csv',index=False)
    result['absolute_error']=abs(result.predicted_log_nH-result.observed_log_nH)
    summary=result.groupby(['method','scenario']).agg(queries=('sample_id','size'),MAE=('absolute_error','mean')).reset_index()
    summary.to_csv(DEST/'summary.csv',index=False)
    audit=dict(passed=max(parity)<2e-6,maximum_old_scenario_replay_error=max(parity),old_scenario_checks=len(parity),
        prediction_rows=len(rows),sourceparameters_refit=False,scope='oxygeninputonly;not a unique physical failure attribution')
    energy.write(DEST/'audit.json',audit);print(summary.to_string(index=False),flush=True);assert audit['passed']


if __name__=='__main__':main()
