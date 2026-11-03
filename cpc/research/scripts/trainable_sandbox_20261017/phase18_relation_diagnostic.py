"""Training-only relation diagnostics; no additional material-model fitting."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import phase18_refinement as final
import phase2_energy as energy
study=final.study;OUT=study.OUT


def main():
    old=study.prior.composite();new=final.composite();rows=[];summaries=[]
    for basis in ('original_RT','binding_joint'):
        for f in new.values():
            if f['method']!='Logit_inventory_'+basis:continue
            for relation in ('constant','dose_plus','dose_minus'):
                o=next(v for v in old.values() if v['method']=='Stock_complex_'+relation+'_'+basis and set(v['train_families'])==set(f['train_families']))
                a=min(t['loss'] for t in f['trials'] if t['loss'] is not None);b=min(t['loss'] for t in o['trials'] if t['loss'] is not None)
                s0=o['parameters'][-2] if relation!='constant' else o['parameters'][-1]
                rows.append(dict(basis=basis,train_families='|'.join(f['train_families']),old_relation=relation,
                    new_fit_key=f['key'],old_fit_key=o['key'],new_fit_origin=f['composite_origin'],old_fit_origin=o['composite_origin'],
                    new_training_loss=a,old_training_loss=b,difference=a-b,old_s0=s0,old_nonzero_floor=float(-np.expm1(-s0)),new_logit_slope=f['parameters'][-1]))
        r=[d for d in rows if d['basis']==basis and d['old_relation']=='dose_minus']
        summaries.append(dict(basis=basis,contexts=len(r),new_training_loss_lower=sum(d['difference']<-1e-8 for d in r),
            new_training_loss_higher=sum(d['difference']>1e-8 for d in r),old_s0_min=min(d['old_s0'] for d in r),old_s0_max=max(d['old_s0'] for d in r),
            new_logit_slope_min=min(d['new_logit_slope'] for d in r),new_logit_slope_max=max(d['new_logit_slope'] for d in r)))
    energy.csvwrite(OUT/'relation_training_diagnostic.csv',rows)
    result=dict(passed=True,code_sha256=energy.sha(Path(__file__)),comparisons=len(rows),summaries=summaries,
        no_new_material_model_fits=True,no_outer_labels_used_in_losscomparison=True,
        interpretation='newlogitnotstrictsuperset:oldnonzerofloor/tailcurvature differ;lowertrainlossbutworseLOFO suggests inspectcomplexitypriorandshift,notuniquecause',
        old_reference_preserved=True,goal_complete=False)
    energy.write(OUT/'relation_diagnostic.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
