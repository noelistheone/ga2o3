"""Predeclare source qualification, then preserve every fulltext attempt."""
from pathlib import Path
import sys, hashlib, subprocess
import requests
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c

OUT=c.BASE/'additional_sources_v2'
SOURCES={
 'wen2023':dict(doi='10.1063/5.0142107',urls=[
     'https://deepblue.lib.umich.edu/bitstream/handle/2027.42/196045/wzhuoqun_1.pdf?isAllowed=y&sequence=1',
     'https://pubs.aip.org/apl/article/122/8/082101/2868000/Si-doping-of-Ga2O3-by-disilane-via-hybrid-plasma']),
 'goto2018':dict(doi='10.1016/j.tsf.2018.09.006',urls=[
     'https://www.sciencedirect.com/science/article/pii/S0040609018305959',
     'https://www.sciencedirect.com/science/article/pii/S0040609018305959/pdfft'])}


def main():
    protocol=OUT/'qualification_before_extraction.json'
    assert not protocol.exists(),'Preserve original acquisition;use a new version'
    c.write(protocol,dict(created_utc=c.e.now(),candidates=SOURCES,
      purpose='additional retrospective external evaluation;not update frozen35-condition training',
      acquisition_not_admission=True,
      eligibility=[
        'primary paper or author institutional thesis with exact chapter/paper provenance',
        'Si or Sn beta-Ga2O3 film with identifiable physical sample and matched volume chemical dose and room-temperature Hall carrier label',
        'chemical concentration from quantitative SIMS or declared equivalent;precursor flow,targetfraction,XPSsurfacepercent and donorfit are not substitutions',
        'preserve chemical profiles/thickness ambiguity and stated growth/anneal history',
        'Hall mobility/resistivity paired only within same sample;derived compatible sigma is not third label',
        'intentional compensating dopants require compatible composition model;detectionlimits are censored,not exact doses',
        'deduplicate source-team/sample reuse against existing7DOI/5groups before independent-source claim',
        'retain all eligible conditions regardless of favorable or unfavorable labels'],
      missing_input_policy='same previously declared O/H/process conditional scenarios;no invented measured inputs',
      future_evaluation='freeze primary full-five-group preset and source lambda selection from training only;preserve0/1/2Hall roles before prediction',
      prior_abstract_exposure=True,not_blind=True,
      code_sha256=c.e.sha(__file__),training_protocol_sha256=c.e.sha(c.BASE/'phase31_rate_redesign/training_protocol.json'),
      goal_complete=False))
    results=[]
    for name,spec in SOURCES.items():
        item=dict(id=name,doi=spec['doi'],attempts=[],admitted=False)
        for i,url in enumerate(spec['urls']):
            attempt=dict(url=url)
            try:
                response=requests.get(url,timeout=(15,45))
                body=response.content
                pdf=response.status_code==200 and body.startswith(b'%PDF')
                path=OUT/f'{name}_attempt{i+1}.{ "pdf" if pdf else "response"}'
                path.write_bytes(body)
                attempt.update(status=response.status_code,final_url=response.url,
                    bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),is_pdf=pdf,
                    path=str(path.relative_to(c.ROOT)))
                if pdf:
                    text=path.with_suffix('.txt')
                    subprocess.run(['pdftotext','-layout',str(path),str(text)],check=True)
                    item.update(pdf=str(path.relative_to(c.ROOT)),text=str(text.relative_to(c.ROOT)),
                                acquired=True,read_depth='not_yet_reviewed')
            except Exception as exc:attempt['error']=repr(exc)
            item['attempts'].append(attempt)
            if item.get('acquired'):break
        results.append(item)
        c.write(OUT/'acquisition_progress.json',dict(created_utc=c.e.now(),sources=results,goal_complete=False))
        print(name,'PDF',item.get('acquired',False),'attempts',len(item['attempts']),flush=True)
    c.write(OUT/'acquisition.json',dict(created_utc=c.e.now(),sources=results,
        failures_preserved=True,qualification_pending=True,goal_complete=False))


if __name__=='__main__':main()
