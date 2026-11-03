"""Read actual CV candidate availability;do not infer unavailable-score causes."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import phase2_energy as e

BASE = e.BASE / 'phase27_charge_feasible'
OUT = BASE / 'selection_readback'


def main():
    assert json.loads((BASE / 'prediction/audit.json').read_text())['passed']
    assert not (OUT / 'audit.json').exists()
    parent = e.BASE / 'phase26_pool_share/prediction/shared_choices.json'
    current = BASE / 'prediction/shared_choices.json'
    rows = []
    for stage, path, method in [(26, parent, 'ShareEffectiveProcess'), (27, current, 'FeasibleProcess')]:
        for choice in json.loads(path.read_text()):
            if choice['method'] != method: continue
            candidates = choice['candidates']; finite = [c for c in candidates if c['reg'] is not None
                and c['score'] is not None and np.isfinite(c['score'])]
            ref = next(c for c in candidates if c['reg'] is None)
            learned = choice['reg'] is not None
            status = 'selected_learned' if learned else ('no_finite_learned_CV' if not finite else 'reference_better_than_finite_learners')
            rows.append(dict(stage=stage, scheme=choice['scheme'], held_unit=choice['held'],
                final_context=choice['held'] is None, declared_method=method, selected_reg=choice['reg'],
                selected_reference=not learned, finite_learned_candidates=len(finite),
                learned_candidates=len(candidates) - 1, reference_CV_score=ref['score'],
                best_finite_learned_CV_score=min(c['score'] for c in finite) if finite else None,
                reference_selection_status=status,
                reference_is_not_a_continuous_infinite_penalty_limit=True,
                actual_material_accuracy_gain_certified=False))
    assert len(rows) == 26
    e.csvwrite(OUT / 'primary_contexts.csv', rows)
    summaries = []
    for stage in (26, 27):
        for scheme in ('DOI', 'cluster'):
            a = [r for r in rows if r['stage'] == stage and r['scheme'] == scheme and not r['final_context']]
            assert len(a) == (6 if scheme == 'DOI' else 5)
            summaries.append(dict(stage=stage, scheme=scheme, outer_contexts=len(a),
                selected_learned=sum(not r['selected_reference'] for r in a),
                selected_reference=sum(r['selected_reference'] for r in a),
                reference_only_finite=sum(r['reference_selection_status'] == 'no_finite_learned_CV' for r in a),
                reference_better_than_finite_learners=sum(r['reference_selection_status'] == 'reference_better_than_finite_learners' for r in a),
                contexts_with_any_finite_learned_CV=sum(r['finite_learned_candidates'] > 0 for r in a)))
    e.csvwrite(OUT / 'summary.csv', summaries)
    # This reads the published selection records,not new model fits or new
    # external labels. An unavailable score alone does not identify its cause.
    e.write(OUT / 'audit.json', dict(passed=True, rows=len(rows), summary_rows=len(summaries),
        no_fits_predictions_or_new_material_labels=True, shared_choices_unchanged=True,
        comparison='actual inner candidate availability and selected reference,not a new accuracy test',
        code_hashes={str(Path(__file__).relative_to(ROOT)): e.sha(Path(__file__))},
        input_hashes={str(p.relative_to(ROOT)): e.sha(p) for p in (parent, current, BASE / 'prediction/audit.json')},
        output_hashes={str((OUT / n).relative_to(ROOT)): e.sha(OUT / n) for n in ('primary_contexts.csv', 'summary.csv')},
        goal_complete=False))
    print('Actual primary CV selection readback', summaries, flush=True)


if __name__ == '__main__': main()
