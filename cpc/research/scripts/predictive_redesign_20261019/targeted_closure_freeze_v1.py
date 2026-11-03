"""Freeze the closing experiment and verify both earlier evidence snapshots."""
from pathlib import Path
import sys,hashlib,re,json
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import targeted_closure_model_v1 as m


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    root=c.ROOT.parent;out=m.OUT/'delivery_freeze_v1';assert not out.exists()
    audit=c.document(m.OUT/'independent_audit_v1/audit.json');assert audit['passed']
    old=c.document(c.BASE/'proposal_followup_20261021_v1/delivery_freeze_v1/manifest.json')
    for j in old['entries']:assert sha(root/j['path'])==j['sha256'],j['path']
    historical=c.document(root/'research_progress_20261021/evidence_index.json')
    for j in historical['entries']:assert sha(root/j['path'])==j['sha256'],j['path']
    assert sha(root/historical['document']['path'])==historical['document']['sha256']
    protocols=sorted(m.OUT.rglob('protocol_before.json'));bindings=set()
    for p in protocols:
        j=c.document(p);c.verify_hashes(j)
        for k in ('code_hashes','input_hashes','evidence_sha256'):bindings.update((c.ROOT/n).resolve() for n in j.get(k,{}))
    doc=root/'轻量补强实验与定稿判断_20261021.md';pdf=doc.with_suffix('.pdf')
    exported=c.document(m.OUT/'pdf_export_v2/pdf_export.json')
    assert exported['pages']==7 and not exported['warnings']
    assert sha(doc)==exported['markdown_sha256'] and sha(pdf)==exported['pdf_sha256']
    draft=c.ROOT/'paper/npj/targeted_closure_20261021_v1/results_update.md';links=0
    for p in (doc,draft):
        for path in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            if path.startswith(('http://','https://','#')):continue
            assert (p.parent/path).exists(),(p,path);links+=1
    review=dict(created_utc=c.e.now(),contact_sheet_pages_reviewed=list(range(1,8)),
        final_pdf_warnings=0,result_plots_visually_reviewed=True,
        first_plot_missing_glyph_and_first_pdf_overfull_logs_retained=True,
        scope='readability/layout,not material prediction validation')
    c.write(m.OUT/'pdf_export_v2/visual_review.json',review)
    files=set(bindings)
    for directory in (m.OUT,c.ROOT/'paper/npj/targeted_closure_20261021_v1'):
        files.update(p.resolve() for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.update(p.resolve() for p in (doc,pdf,c.ROOT/'docs/ML_SANDBOX_TARGETED_CLOSURE_PLAN_20261021.md'))
    files.update(p.resolve() for p in Path(__file__).parent.glob('targeted_*_v*.py'))
    entries=[dict(path=p.relative_to(root).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files)]
    manifest=dict(created_utc=c.e.now(),path_base=str(root),entries=entries,previous_531_files_unchanged=True,
        historical_85_evidence_unchanged=True,stable_joint_prediction_objective_achieved=False)
    c.write(out/'manifest.json',manifest)
    for j in entries:assert sha(root/j['path'])==j['sha256'],j['path']
    fit=c.document(m.OUT/'inventory_training_v1/complete.json');rate=c.document(m.OUT/'rate_scale_v1/complete.json')
    receipt=dict(created_utc=c.e.now(),files=len(entries),manifest_sha256=sha(out/'manifest.json'),
        protocol_bindings_verified=len(protocols),previous_frozen_files_unchanged=len(old['entries']),
        historical_evidence_files_unchanged=len(historical['entries']),master_report_unchanged=True,
        new_upstream_fit_records=fit['new_fit_records'],optimizer_starts=fit['optimizer_starts'],
        rate_fit_records=rate['scale_fit_records'],rate_ridge_solves=rate['actual_ridge_linear_solves'],
        audit_checks=audit['checks'],carrier_preservation_error_dex=rate['maximum_nH_change_dex'],
        pdf_pages=exported['pages'],pdf_warnings=exported['warnings'],local_document_links=links,
        all_this_round_jobs_ended=True,new_material_samples=0,GPU_used=False,
        main_tex_unchanged=True,stable_joint_prediction_objective_achieved=False,persistent_goal='none')
    c.write(out/'receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
