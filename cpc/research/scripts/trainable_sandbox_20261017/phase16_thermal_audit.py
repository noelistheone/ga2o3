"""Trace thermal records without inventing a film's peak thermal history."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from csv_io_v2 import read_result_csv as read
import phase2_energy as energy

OUT=energy.BASE/'phase16_thermal_joint'
TEXTDIR=energy.BASE/'phase8_process_temperature'
SOURCES={
 '10.1002/aelm.202300679':dict(text='/data/lawrence/Physics/Ga2O3-Net/extra-paper/v56_markdown/ec3fc6d0e0ba6300.md',
   locator='Fig3/Table1 (825C cohort); ExperimentalSection; separate875C series not these specimens',
   interval=None,pre_C=None,pre_minutes=None,pre_scope='not specified in read methods',
   post_C=None,post_minutes=None,post_scope='explicitly no postmetallization anneal',
   thermometer='not identified in read methods',checks=['at825°C','withoutpostmetallizationannealing']),
 '10.1007/s10853-015-9693-6':dict(text=str(TEXTDIR/'baldini2015_text.txt'),
   locator='Experimental, PDFp2/printed3650; all samples850C; Fig5 cohort',
   interval=None,pre_C=1000.,pre_minutes=60.,pre_scope='oxygen-rich substrate treatment before film deposition',
   post_C=None,post_minutes=None,post_scope='InGa eutectic contacts; no film anneal specified',
   thermometer='not identified in read methods',checks=['All\nsamples were grown at 850','atmosphere at 1000']),
 '10.1007/s40843-022-2167-x':dict(text=str(TEXTDIR/'xiang2022_text.txt'),
   locator='Experimental PDFpp1-2; Table1/adjacent text',interval=None,
   pre_C=None,pre_minutes=None,pre_scope='not specified in read methods',
   post_C=470.,post_minutes=1.,post_scope='post-deposition contact anneal in N2',
   thermometer='not identified in read methods',checks=['830°C','470°C under N2 ambient for 1 min']),
 '10.1063/5.0027870':dict(text=str(TEXTDIR/'mauze2020_text.txt'),
   locator='Methods PDFp2; Fig4a,b MOCATAXY cohort, PDFp4; per-markerT absent',interval=[700.,900.],
   pre_C=800.,pre_minutes=30.,pre_scope='substrate Ga etch10min then O polish30min800C, before film growth',
   post_C=500.,post_minutes=1.,post_scope='post-deposition contact RTA in N2',
   thermometer='thermocouple explicitly stated for conventional PAMBE; not separately established for every MOCATAXY marker',
   checks=['range of 700','500','in an N2 atmosphere for 1 min']),
 '10.1063/5.0101132':dict(text=str(TEXTDIR/'mccandless2022_text.txt'),
   locator='Methods PDFp2; uniform samples TableI/PDFp4,523/502/523C',interval=None,
   pre_C=900.,pre_minutes=30.,pre_scope='in-situ vacuum substrate desorption before UID buffer and doped film',
   post_C=None,post_minutes=None,post_scope='no post-growth/contact anneal identified in main text',
   thermometer='substrate pyrometer stated in growth discussion; TableI Tsub is the sample-specific record',
   checks=['heated to 900','for 30 min','pyrometer pointed at the substrate']),
}


def build():
    data=read(energy.BASE/'dataset_v1.csv');rows=[];sources=[]
    assert len(data)==29 and data.family.nunique()==5
    for doi,spec in SOURCES.items():
        source=Path(spec['text']);text=source.read_text()
        for fragment in spec['checks']:assert fragment in text,(doi,fragment)
        sources.append(dict(family=doi,url='https://doi.org/'+doi,source_file=str(source),sha256=energy.sha(source),
            **{k:v for k,v in spec.items() if k not in ('text','checks')},
            fulltext_read='local cached primary text; thermal methods/figure membership checked',
            online_retrieval='DOI opens attempted2026-10-18, inaccessible/403; no newly downloaded fulltext claimed'))
    for r in data.itertuples():
        s=SOURCES[r.family];interval=s['interval'];lo=float(r.growth_T_C) if interval is None else interval[0]
        hi=float(r.growth_T_C) if interval is None else interval[1]
        post=s['post_C'];cap=max(hi,post if post is not None else hi)+273.15
        rows.append(dict(sample_id=r.sample_id,family=r.family,original_growth_T_C=float(r.growth_T_C),
            growth_record_status='reported_sample_or_cohort_value' if interval is None else 'reported_cohort_interval_original700C_is_lower_bound_proxy',
            growth_lower_C=lo,growth_upper_C=hi,per_marker_growth_T_known=interval is None,
            pre_deposition_substrate_T_C=s['pre_C'],pre_deposition_minutes=s['pre_minutes'],pre_deposition_scope=s['pre_scope'],
            post_deposition_T_C=post,post_deposition_minutes=s['post_minutes'],post_deposition_scope=s['post_scope'],
            thermometer_scope=s['thermometer'],complete_film_peak_history_known=False,
            actual_film_Tmax_K=None,cooling_trajectory_recorded=False,
            thermal_cap_K=cap,thermal_cap_lower_scenario_K=max(lo,post if post is not None else lo)+273.15,
            thermal_cap_status='conditional_reported_growth_or_postanneal_upper_not_proven_actual_Tmax',
            pre_deposition_temperature_used_as_film_cap=False,source_file=s['text'],source_locator=s['locator']))
    return rows,sources


def main():
    rows,sources=build();audit=OUT/'thermal_audit';audit.mkdir(parents=True,exist_ok=True)
    for name,content in [('specimens.csv',rows)]:
        path=audit/name
        if path.exists():
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                temp=Path(td)/name;energy.csvwrite(temp,rows);assert temp.read_bytes()==path.read_bytes()
        else:energy.csvwrite(path,content)
    payload=dict(version=1,sources=sources,specimens=29,DOIs=5,reported_point_or_cohort_temperature=18,
        interval_only_specimens=11,complete_actual_peak_histories=0,original_dataset_modified=False,
        pre_deposition_hotter_events_excluded_from_film_cap=True,
        correction='Baldini current specimens all850C;875C refers to other published precursor growth,not this cohort',
        cap_choice='main conditional cap uses each reported film growthT or reported intervalupper,plus any reported postfilm anneal if hotter;Mauze900C is not a measured marker assignment',
        lower_cap='audit retains Mauze700C scenario endpoints;not a second tuned training option in TierA',
        limits='no completeness claim,unknown ramps/cooldown/calibration;no specimen removed;all29pO2 still explicit1e-5 scenario',
        goal_complete=False)
    pp=audit/'audit.json'
    if pp.exists():assert json.loads(pp.read_text())==energy.clean(payload)
    else:energy.write(pp,payload)
    print(json.dumps({k:v for k,v in payload.items() if k!='sources'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
