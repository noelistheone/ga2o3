"""Preserve the actual report; repair heading math in a new layout version."""
from pathlib import Path
import sys,json,subprocess,hashlib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as e
from phase29_report_claims import audit

OUT=e.BASE/'phase29_joint_scattering'
TITLE='ML与Sandbox_共享库存与正散射联合预测研究'


def main():
    dest=OUT/'report_layout_v2'
    assert not dest.exists(),'Preserve earlier layout repair attempts'
    dest.mkdir()
    old=Path(__file__).with_name('export_research_pdf_phase29.py')
    renderer=Path(__file__).with_name('export_research_pdf_phase29_layout_v2.py')
    assert not renderer.exists()
    source=old.read_text()
    changes=[('esc(line[3:])','inline(line[3:])'),
             ('esc(line[4:])','inline(line[4:])')]
    for before,after in changes:
        assert source.count(before)==1
        source=source.replace(before,after)
    renderer.write_text(source)
    original=OUT/(TITLE+'.md')
    repaired=OUT/(TITLE+'_排版修订V2.md')
    assert not repaired.exists() and not repaired.with_suffix('.pdf').exists()
    repaired.write_bytes(original.read_bytes())
    assert e.sha(original)==e.sha(repaired)
    claims=json.loads((OUT/'report_v1/numeric_claims.json').read_text())
    proof=audit(repaired,claims)
    assert proof['passed'] and proof['claims_checked']==654
    e.write(dest/'protocol_before.json',dict(created_utc=e.now(),
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in (Path(__file__),old,renderer)},
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in
            (original,repaired,OUT/'report_v1/numeric_claims.json',
             OUT/'rendering_check_v1.json',OUT/'visual_review_v1.json')},
        renderer_changes=changes,source_content_unchanged=True,
        results_models_scores_and_theory_unchanged=True,
        reason='Manual review: appendix E heading displayed literal inline LaTeX delimiters',
        goal_complete=False))
    with (dest/'render.log').open('w') as log:
        r=subprocess.run([sys.executable,str(renderer),str(repaired)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    r.check_returncode()
    build=OUT/('_pdf_build_'+hashlib.sha256(repaired.name.encode()).hexdigest()[:8])
    export=json.loads((build/'export.json').read_text())
    assert not export['warnings'],export['warnings']
    pages=dest/'pages';pages.mkdir()
    subprocess.run(['pdftoppm','-png','-scale-to','1400',str(repaired.with_suffix('.pdf')),
        str(pages/'page')],check=True,capture_output=True)
    old_render=json.loads((OUT/'rendering_check_v1.json').read_text())
    images=sorted(pages.glob('page-*.png'))
    assert len(images)==old_render['pages']==17
    old_by_name={Path(p).name:d for p,d in old_render['page_image_sha256'].items()}
    same=[];changed=[]
    for p in images:
        (same if e.sha(p)==old_by_name[p.name] else changed).append(p.name)
    e.write(dest/'audit.json',dict(passed=True,created_utc=e.now(),
        source_sha256=e.sha(repaired),original_source_sha256=e.sha(original),
        source_content_unchanged=True,numeric_claims_rechecked=proof['claims_checked'],
        original_manual_review_finished=True,
        unchanged_page_images=same,changed_page_images=changed,
        new_training=0,new_predictions=0,theory_and_result_values_unchanged=True,
        evidence_sha256={str(p.relative_to(ROOT)):e.sha(p) for p in
            (original,repaired,renderer,dest/'protocol_before.json')},
        goal_complete=False))
    e.write(dest/'rendering_check_v2.json',dict(passed=True,created_utc=e.now(),
        pages=17,warnings=export['warnings'],source_sha256=e.sha(repaired),
        pdf_sha256=e.sha(repaired.with_suffix('.pdf')),
        page_image_sha256={str(p.relative_to(ROOT)):e.sha(p) for p in images},
        manual_review_complete=False,goal_complete=False))
    print('Layout V2: 654 unchanged claims; identical pages',same,'changed pages',changed,flush=True)


if __name__=='__main__':main()
