"""Reuse the native-math progress renderer without changing its frozen files."""
from pathlib import Path
import importlib.util

ROOT = Path('/home/lawrence/Physics')
SPEC = importlib.util.spec_from_file_location(
    'historical_progress_renderer', ROOT/'research_progress_20261021/export_progress_pdf.py')


def main():
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    module.SOURCE = ROOT/'新预测方案的理论核对与执行记录_20261021.md'
    module.OUT = ROOT/'Ga2O3-Sandbox/results/predictive_redesign_20261019/proposal_followup_20261021_v1/pdf_export_v1'
    module.OUT.mkdir(exist_ok=False)
    module.BUILD = module.OUT/'pdf_build'
    module.main()


if __name__ == '__main__':
    main()
