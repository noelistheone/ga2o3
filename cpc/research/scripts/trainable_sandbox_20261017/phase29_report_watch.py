"""Generate the actual report after completed independent prediction checks."""
from pathlib import Path
import sys,json,os,time,subprocess

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase29_data as data
import phase2_energy as e

OUT=data.OUT


def check(protocol):
    for name,digest in protocol['bound_files'].items():assert e.sha(ROOT/name)==digest,name


def run(script,logname,target):
    result=OUT/target
    if result.exists():
        assert json.loads(result.read_text())['passed'];return
    logpath=OUT/logname
    assert not logpath.exists(),'inspect existing report child/log before versioned resume'
    with logpath.open('w') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).with_name(script))],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        e.write(OUT/(logname+'_started.json'),dict(created_utc=e.now(),pid=child.pid,script=script))
        code=child.wait()
    e.write(OUT/(logname+'_finished.json'),dict(created_utc=e.now(),exit_code=code,script=script))
    if code:raise RuntimeError('Preserve failed report stage '+script)
    assert json.loads(result.read_text())['passed']


def main():
    target=OUT/'report_protocol_before_wait.json'
    assert not target.exists(),'inspect actual report watcher before any versioned resume'
    files=[Path(__file__),*[Path(__file__).with_name(s) for s in (
        'phase29_delivery_docs.py','phase29_report.py','phase29_report_claims.py',
        'phase29_report_audit.py','phase29_report_export.py','export_research_pdf_phase29.py')],
        *[ROOT/'docs'/s for s in (
            'ML_SANDBOX_METHOD_PHASE29_V2_20261019.md',
            'ML_SANDBOX_JOINT_TRAINING_MATH_PHASE29_20261019.md',
            'ML_SANDBOX_SCENARIO_INFORMATION_PHASE29_20261019.md',
            'ML_SANDBOX_TRANSPORT_SCOPE_PHASE29_20261019.md',
            'ML_SANDBOX_SCATTER_BUDGET_CORRECTION_PHASE29_20261019.md',
            'ML_SANDBOX_JOINT_SCATTERING_THEORY_PHASE28_20261019.md')]]
    protocol=dict(created_utc=e.now(),pid=os.getpid(),waits_for_existing_postwatch=2954829,
        bound_files={str(p.relative_to(ROOT)):e.sha(p) for p in files},
        no_training_prediction_or_metric_changes=True,manual_visual_review_required=True,goal_complete=False)
    e.write(target,protocol)
    assert json.loads((OUT/'scatter_budget_notation_audit_v1/audit.json').read_text())['passed']
    print('Report watcher PID',os.getpid(),'waits for existing postwatch2954829',flush=True)
    while not (OUT/'post_prediction_independent_checks_complete.json').exists():
        state=Path('/proc/2954829/stat')
        if not state.exists() or state.read_text().split()[2]=='Z':
            raise RuntimeError('Existing post-prediction pipeline stopped;inspect original log,no blind restart')
        time.sleep(10.)
    check(protocol)
    run('phase29_delivery_docs.py','delivery_documentation.log','delivery/documentation_audit.json')
    check(protocol)
    run('phase29_report.py','report_generation.log','report_v1/numeric_claims_audit.json')
    check(protocol)
    run('phase29_report_audit.py','report_independent_audit.log','report_v1/independent_audit.json')
    check(protocol)
    run('phase29_report_export.py','report_export.log','rendering_check_v1.json')
    check(protocol)
    rendered=json.loads((OUT/'rendering_check_v1.json').read_text())
    e.write(OUT/'report_ready_for_visual_review.json',dict(created_utc=e.now(),passed=True,
        report='ML与Sandbox_共享库存与正散射联合预测研究.pdf',pages=rendered['pages'],
        manual_all_page_visual_review_complete=False,
        independent_report_audit_passed=True,all_adverse_results_retained=True,
        outstanding=['actual all-page visual review','complete scientific result/goal audit','final artifact freeze'],
        overall_goal_complete=False))
    print('Actual report/PDF ready;manual all-page review and full goal audit stillrequired',flush=True)


if __name__=='__main__':main()
