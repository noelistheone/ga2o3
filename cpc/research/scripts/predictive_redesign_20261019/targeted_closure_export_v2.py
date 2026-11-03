"""Readable native-math PDF, with explicit Unicode super/subscript handling."""
from pathlib import Path
import importlib.util,re

ROOT=Path('/home/lawrence/Physics')


def main():
    source=ROOT/'research_progress_20261021/export_progress_pdf.py'
    spec=importlib.util.spec_from_file_location('frozen_native_math_renderer',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    oldesc=module.esc
    sup=dict(zip('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺ᵀᵈ','0123456789-+Td'))
    sub=dict(zip('₀₁₂₃₄₅₆₇₈₉₋₊','0123456789-+'))
    pattern=re.compile('(['+''.join(sup)+']+|['+''.join(sub)+']+)')
    def escape(text):
        out=[]
        for token in pattern.split(text):
            if token and all(c in sup for c in token):out.append(r'\textsuperscript{'+''.join(sup[c] for c in token)+'}')
            elif token and all(c in sub for c in token):out.append(r'\textsubscript{'+''.join(sub[c] for c in token)+'}')
            else:out.append(oldesc(token))
        return ''.join(out)
    module.esc=escape
    original_convert=module.convert
    def convert():
        title,body=original_convert()
        toc=r'\clearpage\tableofcontents\clearpage'
        body=body.replace(toc,'__CONTENTS_BLOCK__').replace(r'\clearpage','').replace('__CONTENTS_BLOCK__',toc)
        return title,body
    module.convert=convert
    module.SOURCE=ROOT/'轻量补强实验与定稿判断_20261021.md'
    module.OUT=ROOT/'Ga2O3-Sandbox/results/predictive_redesign_20261019/targeted_closure_20261021_v1/pdf_export_v2'
    module.OUT.mkdir(exist_ok=False);module.BUILD=module.OUT/'pdf_build'
    module.main()


if __name__=='__main__':main()
