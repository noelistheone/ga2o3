"""Fix missing superscript glyphs; keep the first render and logs intact."""
from pathlib import Path
import importlib.util

ROOT = Path('/home/lawrence/Physics')


def main():
    path = ROOT/'research_progress_20261021/export_progress_pdf.py'
    spec = importlib.util.spec_from_file_location('historical_progress_renderer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.esc

    def esc(s):
        text = original(s)
        for old, new in [('ᵀ', 'T'), ('⁰', '0')]:
            text = text.replace(old, r'\textsuperscript{' + new + '}')
        return text

    module.esc = esc
    module.SOURCE = ROOT/'新预测方案的理论核对与执行记录_20261021.md'
    module.OUT = ROOT/'Ga2O3-Sandbox/results/predictive_redesign_20261019/proposal_followup_20261021_v1/pdf_export_v2'
    module.OUT.mkdir(exist_ok=False)
    module.BUILD = module.OUT/'pdf_build'
    module.main()


if __name__ == '__main__':
    main()
