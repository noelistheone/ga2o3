"""Verify actual weighted training objectives and nested split boundaries before fitting."""
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase25_training as t
import phase2_energy as e


def main():
    rows=[]
    def check(name,error,tol):
        rows.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<=tol)))
    # Includes reported-H old chemistry and high-dose new chemistry.
    d=t.TABLE[t.TABLE.sample_id.str.endswith('::C') | t.TABLE.sample_id.eq(t.data.LEEDY+'::target-1')].copy()
    assert len(d)==2
    for method in t.SPECS:
        spec=t.SPECS[method];n=dict(none=0,constant=1,dose=2,process=4)[spec['basis']]
        w=np.r_[18.,.8,np.array([.6,.4,-.2,.1])[:n]] if spec['kind']=='direct' else np.r_[.5,16.,np.array([.6,.4,-.2,.1])[:n]]
        sw=np.sqrt(t.weights(d,'DOI'));y=d.target.to_numpy(float);gamma=.1
        def calc(w):
            v,j,delta,dj,valid=t.forward(method,w,d,penalty=True)
            return np.r_[(v-y)*sw,np.sqrt(gamma)*delta*sw],np.r_[j*sw[:,None],np.sqrt(gamma)*dj*sw[:,None]]
        r,j=calc(w);fd=[]
        for k in range(len(w)):
            a=np.zeros(len(w));a[k]=1e-4
            fd.append((calc(w+a)[0]-calc(w-a)[0])/2e-4)
        fd=np.asarray(fd).T
        check(method+'/actual_objective_gradient',np.max(np.abs(fd-j)/(1+np.abs(j))),3e-5)
    ctx={}
    for scheme in ('DOI','cluster'):
        units=t.units(scheme);contexts=t.contexts(scheme);ctx[scheme]=len(contexts)
        assert len(contexts)==len(set(contexts))
        for held in units:
            train=tuple(u for u in units if u!=held)
            tr=t.subset(scheme,train)
            check(scheme+'/'+held+'/outer_exclusion',float(tr[t.unit_column(scheme)].eq(held).any()),0.)
            check(scheme+'/'+held+'/normalized_weights',abs(t.weights(tr,scheme).sum()-1.),1e-12)
            for inner in train:
                inner_train=tuple(u for u in train if u!=inner)
                assert inner_train in contexts and train in contexts
                it=t.subset(scheme,inner_train)
                check(scheme+'/'+held+'/'+inner+'/nested_exclusion',float(it[t.unit_column(scheme)].isin([held,inner]).any()),0.)
        for train in contexts:
            d=t.subset(scheme,train);sw=np.sqrt(t.weights(d,scheme));x=np.log10(d.incorporated_cm3.to_numpy(float))-18.
            coef=np.linalg.lstsq(np.c_[np.ones(len(d)),x]*sw[:,None],d.target.to_numpy(float)*sw,rcond=None)[0]
            check(scheme+'/line_fallback_bounds/'+t.digest(train),float(not(13<=coef[0]<=24 and -2<=coef[1]<=4)),0.)
    p=dict(passed=all(r['passed'] for r in rows),checks=len(rows),rows=rows,contexts=ctx,
        total_contexts=sum(ctx.values()),new_training_labels=4,retrospective=True,goal_complete=False,
        code_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in (Path(__file__),ROOT/'scripts/trainable_sandbox_20261017/phase25_training.py')})
    e.write(t.OUT/'learning_gates.json',p)
    print('phase25 learning gates',p['passed'],p['checks'],'contexts',ctx,flush=True)
    if not p['passed']:raise RuntimeError('Preserve failed learning gate before making a repair')


if __name__=='__main__':main()
