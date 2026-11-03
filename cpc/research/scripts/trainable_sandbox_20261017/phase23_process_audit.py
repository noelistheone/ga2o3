"""Audit archived growth events, publication identity and independent metrology inputs."""
from pathlib import Path
from datetime import datetime
import sys,json,zipfile,io,re,hashlib,configparser
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from process_log_io_v1 import read_monitor,shutter_open,at,exposure_segments,duration_summary
import phase2_energy as e
BASE=e.BASE;OUT=BASE/'phase23_process_inputs';PARENT=BASE/'phase22_predictive_inference/source_leads'
CHANNELS={'SiO2':'Data/SiO2_base.Measured.txt','SiO':'Data/SiO_baseOld.Measured.txt','GaOx':'Data/GaOx_base.Measured.txt',
    'substrate_controller':'Data/subs_center.Measured.txt','pyrometer':'Data/Pyro.Reading.txt',
    'main':'Data/Main_Shutter.Value.txt','Ga_shutter':'Data/GaOx_base.ShutterStatus.txt',
    'Si_shutter':'Data/SiO2_base.ShutterStatus.txt','packed':'Data/ShutterControl.Value.txt'}
TII=[('a',124,2.7e17,1286,1000,1e15),('b',119,2.9e17,1134,1000,6e14),('c',111,3.5e17,1115,1000,6e14),
    ('d',129,3.5e17,1119,1000,1e15),('e',104,9.8e17,1187,1000,6e14),('f',86,1.3e18,1225,1000,1e15),
    ('g',98,1.5e18,1171,1000,6e14),('h',91,2.4e18,1176,1000,1e15),('i',75,5.5e18,1262,1000,1e15),
    ('j',79,1.1e19,1286,1000,1e15),('k',62,3.0e19,1319,1000,1e15),('l',68,4.4e19,1350,1000,1e15)]


def freeze():
    inputs=(PARENT/'MBE_Growth_Logs.zip',PARENT/'Azizie2023.pdf',OUT/'literature/SiO2_Calibrations.zip',BASE/'artifact_manifest_phase22.json')
    files=(Path(__file__),Path(__file__).parent/'process_log_io_v1.py',ROOT/'docs/ML_SANDBOX_PROCESS_INPUT_NEXT_20261018.md')
    p=dict(created_utc=e.now(),previous_goal_turn='progress: phase22 complete predictive experiment and968-artifact freeze',
        purpose='trace process inputs and same-specimen chemical/Hall pairing before model extension',
        input_hashes={str(x.relative_to(ROOT)):e.sha(x) for x in inputs},code_hashes={str(x.relative_to(ROOT)):e.sha(x) for x in files},
        no_experiment_code_or_macros_executed=True,no_raw_archive_extraction=True,
        shutter_policy='explicit direct status labels,not unverified packed-bit prose; actual masks require main and source shutters',
        analog_policy='event-held weighted summaries; invalid header-range values preserved,not clipped',
        matching='input-temperature tolerance2C gives candidates only,never confirmed identities or Hall-based matching',
        chemical_policy='Hall-inferred flux cannot be chemical input; independent XRR/SIMS need separate calibration audit',
        model_changes=0,new_training_labels=0,goal_complete=False)
    path=OUT/'process_audit/protocol.json'
    if path.exists():assert json.loads(path.read_text())['code_hashes']==p['code_hashes']
    else:e.write(path,p)


def main():
    freeze();dest=OUT/'process_audit';details=[];summary=[];candidates=[];file_index=[]
    with zipfile.ZipFile(PARENT/'MBE_Growth_Logs.zip') as outer:
        for archive in sorted(a for a in outer.namelist() if a.endswith('.zip')):
            raw=outer.read(archive)
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                info=z.read('SampleInformation.txt').decode('utf-8-sig');meta={}
                for l in info.splitlines():
                    if ':' in l:k,v=l.split(':',1);meta[k.strip()]=v.strip()
                meta['description']=info.split('Sample description:',1)[-1].strip();run=meta['Sample ID']
                end=(datetime.fromisoformat(meta['End time'])-datetime.fromisoformat(meta['Start time'])).total_seconds()
                assert end>0;monitors={};errors=[];hashes={}
                for key,path in CHANNELS.items():
                    if path not in z.namelist():errors.append(dict(channel=key,error='missing'));continue
                    blob=z.read(path);hashes[path]=hashlib.sha256(blob).hexdigest()
                    try:monitors[key]=read_monitor(blob)
                    except (ValueError,UnicodeError) as exc:errors.append(dict(channel=key,error=repr(exc)))
                exposures={}
                for name,mask in {'Ga_main':{'Ga_shutter':True,'main':True},'doped_Ga_main':{'Ga_shutter':True,'main':True,'Si_shutter':True},
                    'Si_main_no_Ga':{'Si_shutter':True,'main':True,'Ga_shutter':False},'Ga_main_closed':{'Ga_shutter':True,'main':False}}.items():
                    if not set(mask)<=monitors.keys():continue
                    segments,unknown=exposure_segments(monitors,end,mask);stats={}
                    for channel in ('SiO2','SiO','GaOx','substrate_controller','pyrometer'):
                        if channel in monitors:stats[channel]=duration_summary(monitors[channel],segments)
                    exposures[name]=dict(segments_s=segments,unknown_shutter_duration_s=unknown,temperature_summaries=stats)
                bits={}
                if 'packed' in monitors:
                    for channel,bit in (('Ga_shutter',128),('Si_shutter',8)):
                        if channel not in monitors:continue
                        t=monitors['packed'].time;opened,good=shutter_open(monitors[channel],t+.5);pv=monitors['packed'].value
                        expected=(np.rint(pv).astype(int)&bit)!=0;bits[channel]=dict(bit=bit,rows=int(good.sum()),matches=int(np.sum((opened==expected)&good)),
                            tolerance_s=.5,note='crosscheck after0.5s channel-update lag;direct status is authoritative')
                record=dict(run_id=run,archive_path=archive,archive_sha256=hashlib.sha256(raw).hexdigest(),metadata=meta,duration_s=end,
                    source_configuration=z.read('SourceConfiguration.txt').decode('utf-8-sig'),
                    input_file_hashes=hashes,exposures=exposures,parse_errors=errors,bit_crosschecks=bits,
                    channels={k:dict(header=m.header,rows=len(m.time),invalid_rows=int((~m.valid).sum()),last_time_s=float(m.time[-1]),declared_range=m.declared_range) for k,m in monitors.items()})
                details.append(record);file_index.extend(dict(run_id=run,path=i.filename,bytes=i.file_size) for i in z.infolist())
                row=dict(run_id=run,description=meta['description'],duration_s=end,parse_errors=len(errors),raw_shutter_to_paper_map='unconfirmed')
                for name,x in exposures.items():
                    row[name+'_duration_s']=float(np.sum(np.diff(x['segments_s'],axis=1))) if len(x['segments_s']) else 0.
                    for channel,stat in x['temperature_summaries'].items():
                        for label in ('mean','minimum','maximum','invalid_duration_s'):row[name+'_'+channel+'_'+label]=stat[label]
                summary.append(row)
                mean=row.get('doped_Ga_main_SiO2_mean');span=row.get('doped_Ga_main_duration_s',0.)
                if mean is not None and span>0:
                    for sample,mu,nh,T,thickness,flux in TII:
                        if abs(mean-T)<=2:candidates.append(dict(run_id=run,sample_label=sample,logged_source_mean=mean,table_source_C=T,absolute_difference=abs(mean-T),
                            basis='source-temperature input only,not verified identity',uses_Hall_label=False))
    e.write(dest/'log_records.json',details);e.csvwrite(dest/'log_summary.csv',summary);e.csvwrite(dest/'temperature_candidates.csv',candidates);e.csvwrite(dest/'archive_file_index.csv',file_index)
    table=[dict(sample_id='10.1063/5.0139622::'+a,sample_label=a,muH_cm2Vs=mu,nH_cm3=n,SiO2_source_C=T,thickness_table_nm=t,Ga2O_flux_cm2s=f,
        Tsub_reported_C=525.,distilled_ozone_pressure_Torr=2.5e-6,oxygen_source='approx80pctO3,balanceO2;not equilibriumO2',
        chemical_cm3=None,chemical_status='same-specimen SIMS plot identified,not yet digitized' if a in ('e','k') else 'not independent same-specimen SIMS',
        thickness_caveat='STEM approximately40pct thicker than table nominal; preserve both scenarios' if a=='a' else 'calibration assumed substrate growth-rate equality; see methods',
        source_locator='PDFp5 TableII, methods PDFp3') for a,mu,n,T,t,f in TII]
    e.csvwrite(dest/'published_Hall_table.csv',table)
    ratio=TII[9][2]/TII[0][2];delta=float(np.log10(ratio))
    e.write(dest/'source_contract.json',dict(doi='10.1063/5.0139622',published_Hall_specimens=12,labels='a-l,not a-k',
        published_SIMS_stacks=['m','n'],independent_same_specimen_SIMS_labels=['e','k'],
        correctable_thickness_factor_a='approximately1.4 fromSTEM;not permissionto correctsourceT or Hall automatically',
        duplicate_reported_inputs=dict(labels=['a','j'],carrier_ratio=ratio,log10_carrier_difference=delta,
            deterministic_MAE_lower_bound_pair=delta/2,deterministic_MAE_lower_bound_all12=delta/12,
            condition='identical reported TableII input vector;not theorem for actual unknown process history'),
        Hall_flux_warning='low-Tsource stars inferred with100pct Hall activation;must not feed them as independent chemical input',
        logs_to_published_sample_identity='not established;temperature candidates only',new_training_labels=0,new_fitting=0,goal_complete=False))
    fits=[];scans=[]
    with zipfile.ZipFile(OUT/'literature/SiO2_Calibrations.zip') as z:
        for f in z.namelist():
            if f.endswith('.sam'):
                c=configparser.ConfigParser(interpolation=None,strict=False);c.read_string(z.read(f).decode('utf-8-sig'))
                fits.append(dict(path=f,sha256=hashlib.sha256(z.read(f)).hexdigest(),layers=[dict(section=k,thickness=c[k].get('Thickness'),density=c[k].get('Density'),phase=c[k].get('Phase1ID')) for k in c.sections() if re.fullmatch(r'LayerGroup\d+Layer\d+',k)],
                    units='saved vendor model units;must verify with XRDML/wavelength/fringes before dose conversion',used_as_ground_truth=False))
            elif f.endswith('.xrdml'):scans.append(dict(path=f,bytes=z.getinfo(f).file_size,sha256=hashlib.sha256(z.read(f)).hexdigest()))
    e.write(dest/'XRR_model_candidates.json',dict(saved_models=fits,scans=scans,no_model_selected_by_Hall=True,new_dose_inputs=0))
    e.write(dest/'audit.json',dict(passed=len(details)==23 and len(table)==12,log_archives=len(details),indexed_files=len(file_index),published_Hall_specimens=12,
        temperature_identity_candidates=len(candidates),verified_log_to_Hall_identities=0,ready_strict_same_specimen_chemical_labels=0,
        model_changes=0,new_training_labels=0,goal_complete=False))
    print('process audit',len(details),'logs',len(file_index),'files',len(candidates),'temperature-only candidates;no verified label join',flush=True)
    print(pd.DataFrame(summary)[['run_id','Ga_main_duration_s','doped_Ga_main_duration_s','Si_main_no_Ga_duration_s','doped_Ga_main_SiO2_mean','doped_Ga_main_pyrometer_mean']].to_string(index=False),flush=True)


if __name__=='__main__':main()
