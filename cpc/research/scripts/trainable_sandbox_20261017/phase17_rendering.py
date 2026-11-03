"""Verify the rendered checkpoint and its numeric claims against result files."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import fitz
from csv_io_v2 import read_result_csv as read
import phase2_energy as energy


def main():
    out=energy.BASE/'phase17_inventory_learning';claims=json.loads((out/'report_claims.json').read_text())
    report=out/'ML与Sandbox_守恒库存学习与跨来源预测.md';pdf=report.with_suffix('.pdf')
    assert energy.sha(report)==claims['report_sha256']
    for filename,digest in claims['evidence_hashes'].items():assert energy.sha(ROOT/filename)==digest,filename
    cache={};md=report.read_text();doc=fitz.open(pdf);text='\n'.join(p.get_text() for p in doc)
    assert len(doc)==8
    for claim in claims['numeric_claims']:
        key=claim['path']
        if key not in cache:cache[key]=read(ROOT/key)
        d=cache[key]
        for name,value in claim['selectors'].items():d=d[d[name].eq(value)]
        assert len(d)==1,claim
        value=float(d[claim['column']].iloc[0]);assert abs(value-claim['value'])<1e-12
        assert f'{value:.6f}' in md and f'{value:.6f}' in text,claim
    build=next(out.glob('_pdf_build_*'));export=json.loads((build/'export.json').read_text())
    assert export['source_sha256']==energy.sha(report) and export['pdf_sha256']==energy.sha(pdf)
    assert not export['warnings'],export['warnings']
    assert text.count('ML+Sandbox')>=1 or 'ML与Sandbox' in text
    folder=out/'rendering';folder.mkdir(exist_ok=True)
    for i in (0,2,4,7):doc[i].get_pixmap(matrix=fitz.Matrix(1.25,1.25)).save(folder/f'page_{i+1}.png')
    (folder/'extracted_text.txt').write_text(text)
    result=dict(passed=True,code_sha256=energy.sha(Path(__file__)),report_sha256=energy.sha(report),pdf_sha256=energy.sha(pdf),
        pages=len(doc),numeric_claims_checked=len(claims['numeric_claims']),independent_result_reader='csv_io_v2',
        checked_visual_pages=[1,3,5,8],visual_review='root inspected representative pages including tables/equations;final page5 label spacing rechecked',
        export_warnings=export['warnings'],goal_complete=False)
    energy.write(out/'rendering_check.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
