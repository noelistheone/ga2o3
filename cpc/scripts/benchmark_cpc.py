"""Frozen CPU benchmark: IFT, genuine forward-only finite differences, and roots.

Run from any directory. Results go to --output; no historical result is replaced.
The protocol is written before the first numerical experiment.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, hashlib, json, platform, time
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from cpc_numerics import BASE, RESEARCH, forward, reset, central_jacobian
from sandbox.kroger_db import load, NPZ
from sandbox.energy_learning import relative_charge_records
from sandbox.implicit_defects import ImplicitDefectLayer
from sandbox import engine, transport


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    args=ap.parse_args(); out=args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    protocol=dict(version=1, design_T_K=[1100.,1500.], design_C_cm3=[3e17,3e19],
        elements=['Si','Sn'], energetic_dimensions=[1,4,16,'all'], repetitions=3,
        fd_steps=[1e-3,1e-4,1e-5,1e-6], primary_step=1e-5,
        timings='three interleaved complete Jacobians; one cold-start per method/repetition; FD perturbations may warm-start',
        computation='single-thread CPU float64; same hot state Jacobian and residual code, same cold roots; forward skips IFT backsolve/output parameter derivatives',
        coordinates='relative charge energies in eV; deterministic input label order, not selected by observed errors',
        independent_reference='pre-existing nested charge/chemical-potential bracketing engine; same catalogue/thermochemistry/statistics, different nonlinear root path; not third-party program execution',
        thresholds=dict(forward_dex=1e-9, primary_gradient_scaled=3e-4, engine_dex=1e-5),
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     (Path(__file__),BASE/'scripts/cpc_numerics.py')},
        software=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
                      processor=platform.processor(),cpu_model=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),'')))
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text()) != protocol:
            raise RuntimeError('Frozen protocol mismatch: use a new output directory')
    else: dump(out/'protocol.json',protocol)
    db=load(); records,_=relative_charge_records(db,np.load(NPZ)['cs_num_each_site'])
    timing=[]; checks=[]; gradients=[]; failures=[]
    for element,T,C,dimension in product(protocol['elements'],protocol['design_T_K'],protocol['design_C_cm3'],protocol['energetic_dimensions']):
        labels=[r for r in records if r['element']==element]
        selected=labels if dimension=='all' else labels[:dimension]
        B=np.zeros((db.nCS,len(selected)))
        for j,r in enumerate(selected): B[r['index'],j]=-r['delta_q']
        layer=ImplicitDefectLayer(db,(element,),B)
        p=layer.parameters(T=T,Nd=9.1e15,doses=(C,),pO2=1e-5)
        common=dict(element=element,Tf_K=T,C_cm3=C,parameters=len(selected))
        try:
            reset(layer); analytic=layer.predict(p)
            reset(layer); value,diag=forward(layer,p,diagnostics=True)
            checks.append(dict(**common,value_error_dex=abs(value-analytic.values[0]),
                               root_residual=analytic.diagnostics['root_residual_log'],
                               condition=analytic.diagnostics['jacobian_condition']))
            J=analytic.jacobian[0,2:2+len(selected)]
            for step in protocol['fd_steps']:
                reset(layer); fd=central_jacobian(layer,p,range(2,2+len(selected)),step)
                for j,(a,b) in enumerate(zip(J,fd)):
                    gradients.append(dict(**common,step=step,label=selected[j]['label_id'],IFT=a,FD=b,
                        scaled_error=abs(a-b)/max(1.,abs(a),abs(b))))
            for rep in range(protocol['repetitions']):
                for method in (('IFT','FD') if rep%2==0 else ('FD','IFT')):
                    reset(layer); start=time.perf_counter()
                    if method=='IFT': layer.predict(p)
                    else: central_jacobian(layer,p,range(2,2+len(selected)),protocol['primary_step'])
                    timing.append(dict(**common,repeat=rep,method=method,seconds=time.perf_counter()-start,
                                       forward_calls=layer.root_calls,hot_residual_evaluations=layer.root_evaluations))
            print('Jacobian',common,flush=True)
        except Exception as exc:
            failures.append(dict(**common,error=repr(exc)))
    reference=[]
    for element,T,C in product(protocol['elements'],protocol['design_T_K'],protocol['design_C_cm3']):
        common=dict(element=element,Tf_K=T,C_cm3=C)
        try:
            layer=ImplicitDefectLayer(db,(element,)); p=layer.parameters(T=T,Nd=9.1e15,doses=(C,),pO2=1e-5)
            a=layer.predict(p)
            h=engine.solve_single(db,T,pO2=1e-5,fixed_conc={element:C},fd=True,Nd_bg=9.1e15)
            cold=engine.quench(db,h,fd=True,Nd_bg=9.1e15)
            ref=np.log10(cold.n/transport.hall_factor(300.,cold.n))
            reference.append(dict(**common,IFT_log_nH=a.values[0],bracket_log_nH=ref,
                error_dex=abs(a.values[0]-ref),hot_EF_error_eV=abs(a.state[0]-h.EF),
                cold_EF_error_eV=abs(a.state[-1]-cold.EF),reference_converged=bool(h.meta['converged'])))
            print('Independent roots',common,flush=True)
        except Exception as exc: failures.append(dict(**common,task='independent_root',error=repr(exc)))
    for name,rows in [('timings',timing),('forward_checks',checks),('gradient_checks',gradients),('independent_roots',reference)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    dump(out/'failures.json',failures)
    t=pd.DataFrame(timing); med=t.groupby(['element','Tf_K','C_cm3','parameters','method']).seconds.median().unstack()
    med['speedup']=med.FD/med.IFT;med.to_csv(out/'timing_summary.csv')
    g=pd.DataFrame(gradients);primary=g[g.step==protocol['primary_step']]
    summary=dict(configurations=len(checks),timed_jacobians=len(timing),gradient_coordinates=len(g),
        independent_root_cases=len(reference),failures=len(failures),
        max_forward_error_dex=max(x['value_error_dex'] for x in checks),
        max_primary_gradient_scaled=float(primary.scaled_error.max()),
        max_independent_error_dex=max(x['error_dex'] for x in reference),
        speedup_by_dimension={str(k):float(v) for k,v in med.groupby('parameters').speedup.median().items()})
    summary['passed']=bool(not failures and summary['max_forward_error_dex']<1e-9 and
        summary['max_primary_gradient_scaled']<3e-4 and summary['max_independent_error_dex']<1e-5)
    dump(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
    if not summary['passed']:raise RuntimeError('Numerical benchmark gate failed; inspect complete outputs')


if __name__=='__main__':main()
