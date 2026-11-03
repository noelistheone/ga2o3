"""Render current method equations before the final actual-results report."""
from pathlib import Path
import sys,json,subprocess,hashlib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase2_energy as e
from phase29_report_claims import sanitize_units

OUT=e.BASE/'phase29_joint_scattering/report_preflight_v2'


def main():
    assert not (OUT/'protocol_before_render.json').exists(),'preserve prior rendering attempt before versioned repair'
    source=ROOT/'docs/ML_SANDBOX_METHOD_PHASE29_V2_20261019.md'
    renderer=Path(__file__).with_name('export_research_pdf_phase19.py')
    e.write(OUT/'protocol_before_render.json',dict(created_utc=e.now(),scope='method formulas only,no new prediction results',
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in (
            Path(__file__),Path(__file__).with_name('phase29_report_claims.py'),renderer)},
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in (source,ROOT/'docs/ML_SANDBOX_SCATTER_BUDGET_CORRECTION_PHASE29_20261019.md')},goal_complete=False))
    text,counts=sanitize_units(source.read_text())
    target=OUT/'方法公式排版预检.md';target.write_text(text)
    with (OUT/'render.log').open('w') as log:
        process=subprocess.run([sys.executable,str(renderer),str(target)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if process.returncode:
        e.write(OUT/'rendering_failure.json',dict(exit_code=process.returncode,original_source_and_log_retained=True))
        raise RuntimeError('Preserve original TeX/PDF rendering failure')
    build=OUT/('_pdf_build_'+hashlib.sha256(target.name.encode()).hexdigest()[:8])
    info=json.loads((build/'export.json').read_text())
    checks=dict(source_control_characters_absent=not any(ord(c)<32 and c!='\n' for c in text),
        inline_math_balanced=text.count(r'\(')==text.count(r'\)'),
        display_math_balanced=text.count(r'\[')==text.count(r'\]'),
        missing_glyph_and_overflow_warnings_absent=not info['warnings'])
    pdf=target.with_suffix('.pdf');images=OUT/'pages';images.mkdir()
    result=subprocess.run(['pdftoppm','-png','-scale-to','1400',str(pdf),str(images/'page')],capture_output=True,text=True)
    result.check_returncode()
    e.write(OUT/'rendering_check.json',dict(passed=all(checks.values()),checks=checks,
        unit_substitutions=counts,warnings=info['warnings'],source_sha256=e.sha(target),pdf_sha256=e.sha(pdf),
        page_images=len(list(images.glob('page-*.png'))),manual_visual_review_complete=False,
        not_prediction_or_material_validation=True,goal_complete=False))
    print('Math preflight',checks,'page images',len(list(images.glob('page-*.png'))),flush=True)
    if not all(checks.values()):raise RuntimeError('Preserve all rendering warnings before repair')


if __name__=='__main__':main()
