"""Independent event/dose checks; calibrated incident-dose scenarios, not material labels."""
from pathlib import Path
import sys,json,zipfile,io
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from process_log_io_v1 import read_monitor,at,shutter_open,exposure_segments,duration_summary
import phase2_energy as e
BASE=e.BASE;OUT=BASE/'phase23_process_inputs';DEST=OUT/'process_checks'
KB=8.617333262145e-5;D=5.3;J0=4.6e28


def flux(source_assumed_C):return J0*np.exp(-D/(KB*(np.asarray(source_assumed_C)+273.15)))


def independent_intervals(m,segments):
    # Elementary intersection of every held raw interval with every selected interval.
    # This does not call the production at()/duration_summary() routines.
    vals=[];dt=[];invalid=0.
    for a,b in segments:
        cursor=a
        for j,t in enumerate(m.time):
            end=m.time[j+1] if j+1<len(m.time) else b
            lo=max(float(a),float(t));hi=min(float(b),float(end))
            if hi<=lo:continue
            cursor=max(cursor,hi)
            if m.valid[j]:vals.append(m.value[j]);dt.append(hi-lo)
            else:invalid+=hi-lo
        if len(m.time) and a<m.time[0]:invalid+=max(0.,min(b,m.time[0])-a)
    return np.asarray(vals),np.asarray(dt),float(invalid)


def main():
    DEST.mkdir(exist_ok=True,parents=True);checks=[]
    def gate(name,good,**extra):
        checks.append(dict(name=name,passed=bool(good),**extra))
        if not good:raise AssertionError(name)
    synthetic=read_monitor(b'Elapsed time [s]\tValue [0,2000]\n2\t100\n4\t-999.9\n7\t200\n')
    values,valid,_=at(synthetic,np.array([0.,2.,3.999,4.,6.99,7.,9.]))
    gate('no extrapolation before first sample',not valid[0])
    gate('right continuous held events',np.array_equal(values[1:],[100.,100.,-999.9,-999.9,200.,200.]))
    gate('invalid sentinel is retained and flagged',np.array_equal(valid,[False,True,True,False,False,True,True]))
    stat=duration_summary(synthetic,np.array([[0.,10.]]))
    gate('exact time mass including unknown interval',abs(stat['valid_duration_s']-5.)<1e-12 and abs(stat['invalid_duration_s']-5.)<1e-12)
    gate('time weights rather than event-count mean',abs(stat['mean']-160.)<1e-12)
    main=read_monitor(b'Elapsed time [s]\tValue [0,1]\tDescription [Open,Close]\n0\t1\n1\t0\n5\t1\n7\t0\n')
    source=read_monitor(b'Elapsed time [s]\tValue [0,1]\tDescription [Open,Closed]\n0\t1\tClosed\n3\t0\tOpen\n8\t1\tClosed\n')
    seg,unknown=exposure_segments(dict(main=main,source=source),10.,dict(main=True,source=True))
    gate('source and main intersections',np.array_equal(seg,[[3.,5.],[7.,8.]]) and unknown==0.)
    bad=read_monitor(b'Elapsed time [s]\tValue [0,1]\n0\t0\tUnknown\n')
    _,v=shutter_open(bad,np.array([0.,1.]));gate('unknown shutter label is not closed proof',not v.any())
    T=1500.;step=.01;analytic=D/(KB*T*T*np.log(10.))
    diff=(np.log10(flux(T-273.15+step))-np.log10(flux(T-273.15-step)))/(2*step)
    gate('source-temperature log flux sensitivity',abs(analytic-diff)<1e-10,analytic_dex_per_K=analytic,finite_difference=diff)
    gate('one um per hour SI-centimetre conversion',abs(1e-4/3600.-1./3.6e7)<1e-20)
    gate('constant-flux layer concentration identity',abs((float(flux(1225))*3600)/(1e-4)-float(flux(1225))/(1e-4/3600.))/float(flux(1225)/(1e-4/3600.))<1e-14)
    records=json.loads((OUT/'process_audit/log_records.json').read_text());rows=[]
    parent=BASE/'phase22_predictive_inference/source_leads/MBE_Growth_Logs.zip'
    with zipfile.ZipFile(parent) as outer:
        for r in records:
            with zipfile.ZipFile(io.BytesIO(outer.read(r['archive_path']))) as z:
                path='Data/SiO2_base.Measured.txt'
                if path not in z.namelist():
                    rows.append(dict(run_id=r['run_id'],status='source channel missing;retained',material_label=False));continue
                m=read_monitor(z.read(path))
            for kind,x in r['exposures'].items():
                segments=np.asarray(x['segments_s'],float).reshape(-1,2)
                if not len(segments):continue
                v,dt,invalid=independent_intervals(m,segments);s=x['temperature_summaries']['SiO2'];mass=float(dt.sum())
                gate(r['run_id']+' '+kind+' independent valid time',abs(mass-s['valid_duration_s'])<1e-7)
                gate(r['run_id']+' '+kind+' independent invalid time',abs(invalid-s['invalid_duration_s'])<1e-7)
                if mass:
                    avg=float(v@dt/mass);mean_flux=float(flux(v)@dt/mass)
                    gate(r['run_id']+' '+kind+' independent weighted mean',abs(avg-s['mean'])<1e-9)
                    dose=float(flux(v)@dt);naive=float(flux(avg)*mass)
                    rows.append(dict(run_id=r['run_id'],exposure=kind,status='conditional calibrated source-dose proxy',
                        valid_duration_s=mass,invalid_duration_s=invalid,source_temperature_assumed_C=avg,
                        mean_flux_proxy_cm2s=mean_flux,integrated_incident_dose_proxy_cm2=dose,
                        flux_at_mean_temperature_dose_proxy_cm2=naive,nonlinear_mean_difference_fraction=dose/naive-1,
                        valid_time_fraction=mass/(mass+invalid),raw_unit_verified=False,material_label=False,
                        assumptions='raw source numerical scale interpreted asC;printed independent source law;event held;not incorporated dose'))
    e.csvwrite(DEST/'incident_dose_scenarios.csv',rows)
    e.write(DEST/'gates.json',dict(passed=all(c['passed'] for c in checks),created_utc=e.now(),checks=checks,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in (Path(__file__),Path(__file__).parent/'process_log_io_v1.py')},
        new_material_labels=0,raw_temperature_unit_calibration_verified=False,verified_log_to_Hall_identities=0,goal_complete=False))
    print('event/dose independent checks',len(checks),'passed;incident proxy rows',len(rows),flush=True)


if __name__=='__main__':main()
