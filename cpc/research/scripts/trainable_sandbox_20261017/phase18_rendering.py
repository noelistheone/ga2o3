"""Verify numerical report claims and render representative PDF pages."""
from pathlib import Path
import sys,json,re
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import fitz
from csv_io_v2 import read_result_csv as read
import phase2_energy as energy
OUT=energy.BASE/'phase18_logodds_prediction'


def main():
    claim=json.loads((OUT/'report_claims.json').read_text());mdfile=OUT/'ML与Sandbox_分配学习与来源泛化检验.md';pdf=mdfile.with_suffix('.pdf')
    assert energy.sha(mdfile)==claim['report_sha256']
    for p,h in claim['evidence_hashes'].items():assert energy.sha(ROOT/p)==h,p
    md=mdfile.read_text();doc=fitz.open(pdf);text='\n'.join(p.get_text() for p in doc);cache={}
    for c in claim['numeric_claims']:
        if c['path'] not in cache:cache[c['path']]=read(ROOT/c['path'])
        s=cache[c['path']]
        for k,v in c['selectors'].items():s=s[s[k].eq(v)]
        assert len(s)==1 and abs(float(s[c['column']].iloc[0])-c['value'])<1e-12,c
        token=f'{c["value"]:.6f}';pattern=r'(?<![\d.])'+(r'0?'+re.escape(token[1:]) if 0<c['value']<1 else re.escape(token))
        assert re.search(pattern,md) and re.search(pattern,text),c
    u=json.loads((OUT/'source_selected_update_audit.json').read_text());assert u['one_anchor_infinite_lambda']==27
    row=next(r for r in u['rows'] if r['method']=='SourceLogit_Logit_inventory_original_RT' and r['budget']==1)
    assert row['infinite_lambda']==5 and row['zero_shift']==5
    export=json.loads(next(OUT.glob('_pdf_build_*/export.json')).read_text())
    assert export['source_sha256']==energy.sha(mdfile) and export['pdf_sha256']==energy.sha(pdf) and not export['warnings']
    assert len(doc)==6
    folder=OUT/'rendering';folder.mkdir(exist_ok=True)
    for i in (0,2,4,5):doc[i].get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(folder/f'page_{i+1}.png')
    (folder/'extracted_text.txt').write_text(text)
    result=dict(passed=True,code_sha256=energy.sha(Path(__file__)),report_sha256=energy.sha(mdfile),pdf_sha256=energy.sha(pdf),
        pages=len(doc),numeric_claims_checked=len(claim['numeric_claims']),selected_zero_update_counts_checked=True,
        representative_rendered_pages=[1,3,5,6],export_warnings=export['warnings'],goal_complete=False)
    energy.write(OUT/'rendering_check.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
