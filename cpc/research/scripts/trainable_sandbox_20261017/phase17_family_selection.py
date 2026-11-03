"""Select shared inventory/output/activation shapes using only inner DOI data."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import phase17_refinement as final
import phase2_energy as energy

study=final.study;OUT=study.OUT;GROUPS={}
for basis in ('original_RT','binding_joint'):
    for kind in ('complex','output'):
        GROUPS[('Inventory_selected_' if kind=='complex' else 'Output_selected_')+basis]=[
            'Stock_'+kind+'_'+relation+'_'+basis for relation in ('constant','dose_plus','dose_minus')]
for training in ('RT','joint'):
    GROUPS['Activation_selected_'+training]=['Matched_activation_'+r+'_'+training for r in ('constant','dose_plus','dose_minus')]


def freeze():
    study.freeze();final.controls.freeze()
    payload=dict(created_utc=energy.now(),code_sha256=energy.sha(Path(__file__)),groups=GROUPS,
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (OUT/'prediction_protocol.json',final.controls.OUT/'protocol.json',energy.BASE/'episodes_v1.csv')},
        selection='innerwholeDOImean nHMAE;chooseconstant/plus/minus andsame7lambda;ties roundscore1e-10/preferlargerlambda/earliermember',
        information='no outerquerylabels,all25individualmodels retained;no new fitting;groups addedbefore newouterevaluation',
        expected_inner_scores=5400,goal_complete=False)
    p=OUT/'family_selection_protocol.json'
    if p.exists():
        old=json.loads(p.read_text())
        for k in ('code_sha256','groups','input_hashes'):assert old[k]==payload[k],k
    else:energy.write(p,payload)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze-only',action='store_true');a=ap.parse_args();freeze()
    if a.freeze_only:return
    inner_count=0
    for group,members in GROUPS.items():
        choices=[];rows=[];temps=[]
        for held in study.FAMILIES:
            train=[f for f in study.FAMILIES if f!=held]
            for budget in (0,1,2):
                candidates=[]
                for index,member in enumerate(members):
                    for lam in ([np.inf] if budget==0 else study.core.LAM):
                        fits=[final.fit([f for f in train if f!=inner],member,{}) for inner in train]
                        scores=[study.core.evaluate(f,inner,budget,lam)[0] for f,inner in zip(fits,train)];inner_count+=len(scores)
                        candidates.append(dict(index=index,member=member,config={},lam=lam,score=float(np.mean(scores)),inner_keys=[f['key'] for f in fits]))
                best=min(candidates,key=lambda c:(round(c['score'],10),-c['lam'],c['index']))
                f=final.fit(train,best['member'],{});score,result=study.core.evaluate(f,held,budget,best['lam'])
                for r in result:r.update(method=group,selected_member=best['member'],fit_key=f['key'])
                rows+=result
                for sample in study.core.prior.prior.SPECS[study.core.prior.prior.SPECS.family.eq(held)].index:
                    result=study.core.prior.prior.temperature(f,sample,budget,best['lam'])
                    for r in result:r.update(method=group,selected_member=best['member'],fit_key=f['key'])
                    temps+=result
                choices.append(dict(method=group,held=held,budget=budget,candidates=candidates,selected=best,fit_key=f['key'],outer_MAE=score))
        folder=OUT/'methods'/group;energy.csvwrite(folder/'primary_predictions.csv',rows);energy.csvwrite(folder/'temperature_predictions.csv',temps)
        energy.write(folder/'selections.json',choices);print('familyselected',group,flush=True)
    assert inner_count==5400
    energy.write(OUT/'family_selection_complete.json',dict(created_utc=energy.now(),groups=6,inner_scores=inner_count,goal_complete=False))


if __name__=='__main__':main()
