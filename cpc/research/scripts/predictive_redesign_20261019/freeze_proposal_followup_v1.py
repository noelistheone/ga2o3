"""Freeze this completed bounded round; verify all existing bound inputs first."""
from pathlib import Path
import sys, hashlib, json, re
sys.path.insert(0, str(Path(__file__).parent))
import phase30_core as c

PHYSICS = c.ROOT.parent
FOLLOW = c.BASE/'proposal_followup_20261021_v1'
OUT = FOLLOW/'delivery_freeze_v1'
DOC = PHYSICS/'新预测方案的理论核对与执行记录_20261021.md'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not OUT.exists(), 'completed delivery must not be overwritten'
    audits = [FOLLOW/'independent_audit_v1/audit.json', FOLLOW/'additional_independent_audit_v1/audit.json']
    for p in audits:
        assert c.document(p)['passed'], p
    protocols = list(FOLLOW.rglob('protocol_before.json')) + [
        c.BASE/'phase35_association_spatial/all_training_pilot_v1/protocol_before.json',
        c.BASE/'phase35_association_spatial/related_group_heldout_v1/protocol_before.json']
    bound = set()
    for p in protocols:
        j = c.document(p)
        c.verify_hashes(j)
        for k in ('code_hashes', 'input_hashes', 'evidence_sha256'):
            bound.update((c.ROOT/n).resolve() for n in j.get(k, {}))
    previous = c.document(PHYSICS/'research_progress_20261021/evidence_index.json')
    for e in previous['entries']:
        assert sha(PHYSICS/e['path']) == e['sha256'], e['id']
    assert sha(PHYSICS/previous['document']['path']) == previous['document']['sha256']
    pdf = c.document(FOLLOW/'pdf_export_v2/pdf_export.json')
    assert not pdf['warnings'] and pdf['pages'] == 20
    assert sha(DOC) == pdf['markdown_sha256']
    assert sha(DOC.with_suffix('.pdf')) == pdf['pdf_sha256']
    # Local document links and image destinations must resolve.
    docs = [DOC, *sorted((c.ROOT/'paper/npj/predictive_redesign_20261021').glob('*.md'))]
    local_links = 0
    for p in docs:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', p.read_text()):
            if target.startswith(('https://', 'http://', '#')):
                continue
            assert (p.parent/target).exists(), (str(p), target)
            local_links += 1
    paths = {
        'all_training_pilot': c.BASE/'phase35_association_spatial/all_training_pilot_v1/fits',
        'related_group_heldout': c.BASE/'phase35_association_spatial/related_group_heldout_v1/fits',
        'carrier_primary': FOLLOW/'constrained_block_v1/primary_fits',
        'conditional_secondary': FOLLOW/'constrained_block_v1/secondary_fits',
        'matched_carrier_controls': FOLLOW/'overlay_nH_controls_v1/fits'}
    counts = {}
    for name, path in paths.items():
        rows = [c.document(p) for p in path.glob('*.json')]
        counts[name] = dict(records=len(rows), optimizer_starts=sum(len(j['trials']) for j in rows))
    assert sum(j['records'] for j in counts.values()) == 93
    assert sum(j['optimizer_starts'] for j in counts.values()) == 192
    refinement = c.document(FOLLOW/'constrained_block_refinement_v2/complete.json')
    assert refinement['records'] == 15 and refinement['optimizer_starts'] == 2
    quality = dict(created_utc=c.e.now(),contact_sheet_pages_reviewed=list(range(1,21)),
        detailed_math_pages_reviewed=[18,19],two_original_result_plots_reviewed=True,
        missing_glyphs_first_render_retained=True,final_render_missing_glyphs=0,
        final_render_overfull_warnings=0,local_document_links_resolved=local_links,
        scope='layout/readability and link integrity;not independent materials validation')
    c.write(FOLLOW/'pdf_export_v2/visual_review.json',quality)
    sources = [FOLLOW,
        c.BASE/'phase35_association_spatial/all_training_pilot_v1',
        c.BASE/'phase35_association_spatial/related_group_heldout_v1',
        c.ROOT/'paper/npj/predictive_redesign_20261021']
    files = set(bound)
    for source in sources:
        files.update(p.resolve() for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.update(p.resolve() for p in [DOC,DOC.with_suffix('.pdf'),PHYSICS/'AGENTS.md',c.ROOT/'AGENTS.md',
        c.BASE/'CONTINUE_HERE.md',Path(__file__),Path(__file__).with_name('build_proposal_followup_report_v1.py'),
        Path(__file__).with_name('export_proposal_followup_pdf_v1.py'),Path(__file__).with_name('export_proposal_followup_pdf_v2.py')])
    entries = [dict(path=p.relative_to(PHYSICS).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
    manifest = dict(created_utc=c.e.now(),path_base=str(PHYSICS),entries=entries,
        preserves_previous_85_evidence=True,stable_full_prediction_objective_achieved=False,
        goal_created_or_updated=False)
    c.write(OUT/'manifest.json',manifest)
    for e in entries:
        assert sha(PHYSICS/e['path']) == e['sha256'], e['path']
    receipt = dict(created_utc=c.e.now(),files=len(entries),manifest_sha256=sha(OUT/'manifest.json'),
        protocol_bindings_verified=len(protocols),all_bound_code_and_inputs_unchanged=True,
        previous_evidence_files_unchanged=len(previous['entries']),previous_master_report_unchanged=True,
        new_fit_records=sum(j['records'] for j in counts.values()),fit_counts=counts,
        primary_and_conditional_optimizer_starts=192,additional_grid_refinement_optimizer_starts=2,
        training_grid_nodes=375,independent_audit_checks=sum(c.document(p)['checks'] for p in audits),
        pdf_pages=pdf['pages'],pdf_warnings=pdf['warnings'],local_links=local_links,
        no_new_material_samples=True,GPU_used=False,all_this_round_jobs_ended=True,
        stable_full_prediction_objective_achieved=False,persistent_goal_state='none',
        limitations='retrospective cohorts;no universal budget improvement;external carrier loss retained;spatial mechanism not established')
    c.write(OUT/'receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
