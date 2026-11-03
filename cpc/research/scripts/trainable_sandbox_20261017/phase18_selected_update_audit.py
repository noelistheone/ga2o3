"""Distinguish selected zero updates from successful internal adaptation."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as energy
OUT=energy.BASE/'phase18_logodds_prediction'


def main():
    rows=[];paths=list((OUT/'source_adaptation_v2/methods').glob('*/selections.json'));assert len(paths)==6
    for p in paths:
        records=json.loads(p.read_text())
        for b in (1,2):
            d=[r for r in records if r['budget']==b];assert len(d)==5
            rows.append(dict(method=p.parent.name,budget=b,infinite_lambda=sum(r['selected']['lam'] is None for r in d),
                zero_shift=sum(r['source_z']==0 for r in d),shifts=[r['source_z'] for r in d]))
    result=dict(passed=True,code_sha256=energy.sha(Path(__file__)),rows=rows,
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in paths},
        one_anchor_infinite_lambda=sum(r['infinite_lambda'] for r in rows if r['budget']==1),
        two_anchor_infinite_lambda=sum(r['infinite_lambda'] for r in rows if r['budget']==2),
        note='relativeoutputbenefit canmeanavoidingharmfulupdate,not learningincrement;commonquery comparisonrequired',goal_complete=False)
    energy.write(OUT/'source_selected_update_audit.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
