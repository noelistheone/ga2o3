"""Render the actual-results report and save every page for visual review."""
from pathlib import Path
import sys,json,subprocess,hashlib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase29_data as data
import phase2_energy as e

OUT=data.OUT;TITLE='ML与Sandbox_共享库存与正散射联合预测研究'


def main():
    target=OUT/'rendering_check_v1.json'
    assert not target.exists(),'preserve previous renderer output before versioned repair'
    assert not (OUT/'report_render_protocol_before.json').exists(),'inspect prior rendering attempt before versioned repair'
    assert json.loads((OUT/'report_v1/independent_audit.json').read_text())['passed']
    source=OUT/(TITLE+'.md');renderer=Path(__file__).with_name('export_research_pdf_phase29.py')
    e.write(OUT/'report_render_protocol_before.json',dict(created_utc=e.now(),
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in (Path(__file__),renderer)},
        input_hashes={str(source.relative_to(ROOT)):e.sha(source)},
        all_pages_require_visual_review=True,goal_complete=False))
    with (OUT/'report_render.log').open('w') as log:
        result=subprocess.run([sys.executable,str(renderer),str(source)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        e.write(OUT/'report_rendering_failure_v1.json',dict(created_utc=e.now(),exit_code=result.returncode,
            original_source_and_log_preserved=True))
        raise RuntimeError('Preserve original actual-report rendering failure')
    build=OUT/('_pdf_build_'+hashlib.sha256(source.name.encode()).hexdigest()[:8])
    export=json.loads((build/'export.json').read_text());pdf=source.with_suffix('.pdf')
    images=OUT/'report_pages_v1';images.mkdir()
    proc=subprocess.run(['pdftoppm','-png','-scale-to','1400',str(pdf),str(images/'page')],capture_output=True,text=True)
    proc.check_returncode();pages=sorted(images.glob('page-*.png'))
    value=dict(passed=not export['warnings'],created_utc=e.now(),pages=len(pages),warnings=export['warnings'],
        pdf_sha256=e.sha(pdf),source_sha256=e.sha(source),manual_review_complete=False,
        page_image_sha256={str(p.relative_to(ROOT)):e.sha(p) for p in pages},
        final_manual_visual_review_still_required=True,no_new_prediction_gain_claim=True,goal_complete=False)
    e.write(target,value);print('Actual report rendered',len(pages),'pages;warnings',len(export['warnings']),flush=True)
    if export['warnings']:raise RuntimeError('Preserve original glyph/overflow warnings before versioned repair')


if __name__=='__main__':main()
