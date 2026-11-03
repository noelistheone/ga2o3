"""Same-overlay complete loss gradient checks, no optimization."""
from pathlib import Path
import sys,time,numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
import phase34_learning_v1 as l

OUT=l.OUT/'loss_gates_v1'


def main():
    assert not (OUT/'protocol_before.json').exists()
    config=dict(gamma=1.,alpha=.1);rows=l.subset(c.units());plans=[];inputs=[c.OLD/'data/scenarios49.csv']
    for method in l.METHODS:
        initial,ref=l.initializer(c.units(),method,config);w=initial[0].copy()
        if len(w)>6:w[6]=0.
        if len(w)>7:w[7]=1.
        plans.append(dict(method=method,parameters=w.tolist(),initializer=ref));inputs.append(c.ROOT/ref['path'])
    codes=[Path(__file__),Path(l.__file__),*[Path(__file__).with_name(v) for v in ['neutral_association_model_v1.py','neutral_association_inventory_v1.py','si_multidonor_overlay_v1.py']]]
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),code_hashes=c.file_hashes(codes),input_hashes=c.file_hashes(inputs),
        config=config,plans=plans,FD_step=2e-5,tolerance=1e-4,query_or_external_labels=False,new_fits=0,goal_complete=False))
    start=time.perf_counter();records=[]
    for plan in plans:
        method=plan['method'];w=np.array(plan['parameters']);r,J,q=l.loss_arrays(method,w,rows,config)
        for k in range(len(w)):
            h=2e-5;wp=w.copy();wm=w.copy();wp[k]+=h;wm[k]-=h
            fd=(l.loss_arrays(method,wp,rows,config)[0]-l.loss_arrays(method,wm,rows,config)[0])/(2*h)
            error=float(max(abs(fd-J[:,k])/(1+abs(fd)+abs(J[:,k]))))
            records.append(dict(method=method,coordinate=k,error=error,passed=error<1e-4))
        print('association full-loss gate',method,flush=True)
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    c.write(OUT/'gates.json',dict(created_utc=c.e.now(),passed=all(r['passed'] for r in records),checks=len(records),records=records,
        max_error=max(r['error'] for r in records),seconds=time.perf_counter()-start,new_fits=0,goal_complete=False))
    assert all(r['passed'] for r in records)


if __name__=='__main__':main()
