"""Train coherent FD/Hall readouts under the original doping prediction task."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import phase14_refinement as parent
import phase2_energy as energy
from sandbox.coherent_hall_model import CoherentHallModel
from sandbox.coherent_fermi import TABLE,published_coherent_profile

core=parent.study;OUT=energy.BASE/'phase15_coherent_hall'
NEW_METHODS={
    'FD_original_RT':dict(fraction='zero',training='RT',readout='rH1'),
    'FD_binding_joint':dict(fraction='one',binding='screen',training='joint',readout='rH1'),
    'HallII_original_RT':dict(fraction='zero',training='RT',readout='ionized'),
    'HallII_binding_joint':dict(fraction='one',binding='screen',training='joint',readout='ionized'),
    'HallExtra_original_RT':dict(fraction='zero',training='RT',readout='ionized_extra'),
    'HallExtra_binding_joint':dict(fraction='one',binding='screen',training='joint',readout='ionized_extra'),
}
BASE_MAPPER=core.mapper;BASE_FIT=parent.base.OLD_FIT;MODELS={};BANK={}
core.METHODS.update(NEW_METHODS);core.OUT=OUT


def mapper(method):
    if method not in NEW_METHODS:return BASE_MAPPER(method)
    if method not in MODELS:MODELS[method]=CoherentHallModel(NEW_METHODS[method],profile=published_coherent_profile(),pO2_atm=1e-5)
    return MODELS[method]


core.mapper=mapper


def oldbank():
    global BANK
    if not BANK:
        BANK={(f['method'],tuple(sorted(f['train_families'])),json.dumps(f['config'],sort_keys=True)):f for f in parent.load_composite().values()}
    return BANK


def getfit(families,method,config):
    if method not in NEW_METHODS:return oldbank()[method,tuple(sorted(families)),json.dumps(config,sort_keys=True)]
    f=BASE_FIT(families,method,config)
    if f.get('coherent_transport_saved'):return f
    start=time.perf_counter();model=mapper(method);before=model.counters()
    for row,saved in zip(core.PRED.itertuples(),f['predictions']):
        assert row.obs_id==saved['obs_id']
        if saved['predicted'] is None:continue
        point=model.point(row,f['parameters']);assert abs(point.values[0]-saved['predicted'])<1e-8
        saved['transport_values']=point.transport_values.tolist()
        saved['diagnostics'].update(hall_factor=point.diagnostics['hall_factor'],
            coherent_transport=point.diagnostics['coherent_transport'],statistics_scope=point.diagnostics['statistics_scope'])
    f['coherent_transport_saved']=True
    f['transport_enrichment_seconds']=time.perf_counter()-start
    f['transport_enrichment_counters']={k:model.counters()[k]-before[k] for k in before}
    energy.write(OUT/'fits'/(f['key']+'.json'),f)
    return f


core.fit=getfit


def freeze():
    gate=json.loads((OUT/'numerical_gates.json').read_text());assert gate['passed']
    for path,digest in gate['code_hashes'].items():assert energy.sha(ROOT/path)==digest,path
    sources=(Path(__file__),ROOT/'src/sandbox/coherent_fermi.py',ROOT/'src/sandbox/coherent_transport.py',
        ROOT/'src/sandbox/coherent_hall_model.py',ROOT/'src/sandbox/donor_mixture.py',ROOT/'src/sandbox/donor_mixture_model.py',
        ROOT/'scripts/trainable_sandbox_20261017/phase13_prediction.py',ROOT/'docs/ML_SANDBOX_COHERENT_HALL_THEORY_20261018.md')
    inputs=(OUT/'numerical_gates.json',TABLE,energy.BASE/'artifact_manifest_phase14.json',energy.BASE/'dataset_v1.csv',
        energy.BASE/'episodes_v1.csv',core.prior.prior.TEMPDIR/'paired_specimens.csv',core.prior.prior.TEMPDIR/'temperature_observations.csv')
    payload=dict(created_utc=energy.now(),new_methods=NEW_METHODS,all_methods=core.METHODS,
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in sources},input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in inputs},
        inherited_controls=[dict(key=f['key'],**f['composite_origin']) for f in oldbank().values()],
        profile=published_coherent_profile().to_dict(),new_jobs=90,new_starts=180,
        primary='original29Si/Sn5DOI,0/1/2sampleanchors,29/23/18queries,common18',
        training='same sourceequalRT orRT+.1curve loss,analyticfullstate/readoutgradient;no mu labels used',
        optimizer='same2starts,max_nfev120,ftol/xtol/gtol1e-7;ALLfailedstartrecords eligibleuniform360refinement',
        physical_change='C2FDfreebands/scalarDOSmass/Hallmoments;legacySTH/energetics/pressure unchanged',
        extra_channel='fixedconstanttau mu0=205cm2/Vs,Tindependent;not computedphonons',
        adaptation='same outputoffset/7lambda innerDOI selection;no newsourceinternaladaptation ininitialbatch',
        transport_evaluation='save rawmu/sigma fromsamephysicalstate;5chemicalpairs2DOI;no newlabels;compareCTinformationseparately',
        limits='pointchargeBorn/RTA andsingleband assumed;n>p not evidenceofoneelectronchannel;historicalretrospective',
        new_DFT_MD=0,new_material_measurements=0,goal_complete=False)
    path=OUT/'prediction_protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        for key in ('new_methods','all_methods','code_hashes','input_hashes','inherited_controls','profile'):
            assert old[key]==energy.clean(payload[key]),key
    else:energy.write(path,payload)


def composite():
    rows={f['key']:f for f in oldbank().values()}
    for path in (OUT/'fits').glob('*.json'):
        f=json.loads(path.read_text());f['composite_origin']=dict(path=str(path.relative_to(ROOT)),sha256=energy.sha(path));rows[f['key']]=f
    return rows


def worker(job):return getfit(*job)['key']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--fit-only',action='store_true');a=ap.parse_args();freeze()
    jobs=[(families,method,{}) for n in (3,4) for families in combinations(core.FAMILIES,n) for method in NEW_METHODS];assert len(jobs)==90
    energy.write(OUT/'training_execution_started.json',dict(created_utc=energy.now(),workers=a.workers,jobs=len(jobs)))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for _ in pool.map(worker,jobs):pass
    energy.write(OUT/'training_execution_complete.json',dict(created_utc=energy.now(),jobs=len(jobs),goal_complete=False))
    if not a.fit_only:
        for method in core.METHODS:core.select(method)
        energy.write(OUT/'selection_execution_complete.json',dict(created_utc=energy.now(),methods=len(core.METHODS),goal_complete=False))


if __name__=='__main__':main()
