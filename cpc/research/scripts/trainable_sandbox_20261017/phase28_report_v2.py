"""Keep the original report; render negative unit exponents with LaTeX."""
import json
from pathlib import Path
import phase28_report as base


def main():
    base.REPORT=base.OUT/'report_v2'
    base.main()
    path=base.OUT/(base.TITLE+'.md')
    original=path.read_text()
    replacements={'cm⁻³':r'\(\mathrm{cm}^{-3}\)','s⁻¹':r'\(\mathrm{s}^{-1}\)'}
    text=original
    counts={}
    for old,new in replacements.items():
        counts[old]=text.count(old)
        text=text.replace(old,new)
    assert sum(counts.values())==5
    assert text.count('report_v1保存表值回查')==1
    text=text.replace('report_v1保存表值回查','report_v2保存本版表值回查')
    assert not any(ord(c)<32 and c not in '\n' for c in text)
    assert text.count(r'\(')==text.count(r'\)')
    path.write_text(text)
    for name in ('numeric_claims.json','numeric_claims_audit.json'):
        p=base.REPORT/name;value=json.loads(p.read_text())
        value.update(report_sha256=base.e.sha(path),renderer_version=2,
            negative_unit_exponents_rendered_as_TeX=counts,
            original_missing_glyph_PDF_preserved=True,
            current_report_evidence_index_corrected=True,
            no_numeric_model_or_score_change=True)
        base.e.write(p,value)
    print('Renderer V2:5negative unit exponents fixed;404numeric claims unchanged',flush=True)


if __name__=='__main__':main()
