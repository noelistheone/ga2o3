"""Reconstruct all finite-state posteriors and nested selections independently."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from scipy.special import logsumexp
from csv_io_v2 import read_result_csv as read
import phase22_prediction as p
s=p.s;OUT=p.OUT/'audit'


def main():
    assert json.loads((p.OUT/'execution_complete.json').read_text())['jobs']==406
    checks=[]
    def check(test,passed,**extra):checks.append(dict(test=test,passed=bool(passed),**extra))
    parent=json.loads((s.BASE/'artifact_manifest_phase20.json').read_text());protected={r['path']:r['sha256'] for r in parent['files']}
    eps={q.stem:json.loads(q.read_text()) for q in (p.OUT/'episodes').glob('*.json')};check('complete_groups',len(eps)==406)
    actual=nominal=pointfail=invalid=0;maxweight=maxscore=0.
    for ek,e in eps.items():
        path=ROOT/e['fit_path'];check('frozen_fit',e['fit_path'] in protected and s.energy.sha(path)==e['fit_sha256']==protected.get(e['fit_path']),key=ek)
        check('held_family_excluded',e['held'] not in e['train_families'],key=ek)
        d=s.table(e['held'],e['budget']);nodes,ai,y,aids=s.design_inputs(d)
        check('only_allowed_anchor_labels',list(ai)==e['anchor_indices'] and aids==e['anchor_ids'] and np.array_equal(y,e['anchor_labels']),key=ek)
        values=np.array(e['state_values'],float);valid=np.array(e['state_valid'],bool);u=np.array(e['coordinates'],float)
        actual+=e['actual_forward_points'];nominal+=e['nominal_forward_points'];pointfail+=len(e['point_failures']);invalid+=int((~valid).sum())
        check('point_budget',e['nominal_forward_points']<=32768,key=ek)
        for candidate in e['candidates']:
            c=candidate['config']
            for n,record in candidate['grids'].items():
                n=int(n);sel=np.arange(129)[::128//(n-1)] if n>1 else np.array([0]);g=values[sel,:,0]
                zero=int(np.flatnonzero(u[sel]==0.)[0]);v=np.all(valid[sel],axis=1);q=np.mean((g-g[zero])**2,axis=1)
                uu=u[sel]
                if n>1:
                    edges=np.r_[-6,(uu[:-1]+uu[1:])/2,6];base=np.diff(edges)/12
                else:base=np.array([1.])
                good=v&np.isfinite(g).all(axis=1)&(q<=36);lp=np.full(len(sel),-np.inf);lp[good]=np.log(base[good])-q[good]/(2*c['tau']**2);lp-=logsumexp(lp)
                prior=(1-c['rho'])*np.exp(lp);prior[zero]+=c['rho'];ok=prior>0
                lp[:]=-np.inf;lp[ok]=np.log(prior[ok])-len(ai)*np.log(np.sqrt(2*np.pi)*c['sigma'])-np.sum((g[ok][:,ai]-y)**2,axis=1)/(2*c['sigma']**2)
                weights=np.exp(lp-logsumexp(lp));err=float(np.max(abs(weights-np.array(record['weights']))));maxweight=max(maxweight,err)
                check('independent_posterior_normalization',err<2e-11 and np.isclose(weights.sum(),1,atol=1e-11,rtol=0) and not weights[~good].any(),key=ek,grid=n,error=err)
                index={s.identity(r):j for j,r in enumerate(nodes)};errs=[];nll=[]
                for r in d.itertuples():
                    if r.role=='guard':continue
                    j=index[s.identity(r)];order=np.argsort(np.nan_to_num(g[:,j],nan=np.inf),kind='stable');z=int(order[min(np.searchsorted(np.cumsum(weights[order]),.5,side='left'),len(order)-1)])
                    expected=values[sel[z],j];saved=next(a for a in record['records'] if a['sample_id']==r.sample_id)
                    err=float(np.max(abs(expected-[saved['predicted'],saved['predicted_log_mu'],saved['predicted_log_sigma']])));maxscore=max(maxscore,err)
                    check('same_state_property_decision',saved['numeric'] and err<1e-9,key=ek,grid=n,error=err)
                    if r.role=='query':
                        errs.append(abs(expected[0]-r.target));terms=np.log(weights[weights>0])-np.log(np.sqrt(2*np.pi)*c['sigma'])-(g[weights>0,j]-r.target)**2/(2*c['sigma']**2)
                        nll.append(float(-logsumexp(terms)))
                check('fullquery_score',abs(np.mean(errs)-record['MAE'])<1e-9 and abs(np.mean(nll)-record['NLL'])<1e-8,key=ek,grid=n)
    choices=json.loads((p.OUT/'all_selections.json').read_text());final=read(p.OUT/'predictions.csv');check('all_final_rows',len(final)==870 and int(final.numeric.sum())==850)
    for c in choices:
        b=c['budget'];held=c['held'];train=c['train_families'];check('outer_fit_exclusion',held is None or held not in train,variant=c['variant'],held=held,budget=b)
        if b:
            options=c['source_candidates']
            for option in options:
                es=[eps[k] for k in option['inner_keys']]
                check('strict_training_inner_membership',len(es)==len(train) and {e['held'] for e in es}==set(train) and all(set(e['train_families'])==set(train)-{e['held']} for e in es),variant=c['variant'],held=held,budget=b)
                score=np.mean([p.candidate(e,option['config'])['MAE'] for e in es]);check('reconstructed_inner_selection_score',abs(score-option['score'])<1e-10)
            best=min(options,key=lambda a:(round(a['score'],10),round(a['NLL'],10),-a['config']['rho'],a['config']['tau'],a['config']['sigma']))
            check('declared_hyperselection',best['config']==c['selected_source']['config'],variant=c['variant'],held=held,budget=b)
        if held is None:continue
        part=final[final.variant.eq(c['variant'])&final.family.eq(held)&final.budget.eq(b)];table=s.table(held,b)
        check('original_roles',dict(zip(part.sample_id,part.role))==dict(zip(table.sample_id,table.role)),variant=c['variant'],held=held,budget=b)
        check('all_queries_covered',part[part.role.eq('query')].numeric.all(),variant=c['variant'],held=held,budget=b)
    for method,(parent,_) in p.VARIANTS.items():
        old=read(s.OUT/'variants'/parent/'predictions.csv');now=final[final.variant.eq(method)&final.budget.eq(0)]
        q=now.merge(old[old.budget.eq(0)],on='sample_id',suffixes=('_new','_old'),validate='one_to_one');err=float(abs(q.predicted_new-q.predicted_old).max())
        check('exact_zero_prediction_replay',len(q)==29 and err<1e-10,variant=method,error=err)
    for protocol in ('prediction_protocol.json',):
        d=json.loads((p.OUT/protocol).read_text())
        for name,h in {**d['code_hashes'],**d['input_hashes']}.items():check('protocol_hash',s.energy.sha(ROOT/name)==h,path=name)
    grid=read(p.OUT/'grid_sensitivity.csv');sensitive=grid[grid.coarse_size.eq(65)&~grid.stable_at_02dex]
    result=dict(passed=all(c['passed'] for c in checks),checks=len(checks),rows=checks,groups=len(eps),variants=10,
        primary_rows=len(final),numeric_primary_rows=int(final.numeric.sum()),actual_forward_points=actual,nominal_forward_points=nominal,
        point_exceptions=pointfail,invalid_point_states=invalid,maximum_posterior_weight_error=maxweight,maximum_property_decision_error=maxscore,
        selected_65_129_sensitive_cases=len(sensitive),maximum_65_129_query_change=float(grid[grid.coarse_size.eq(65)].maximum_query_change.max()),
        stable_new_internal_predictive_gain_established=False,continuum_posterior_convergence_established=False,
        new_shared_optimizer_starts=0,new_training_labels=0,new_material_measurements=0,goal_complete=False)
    s.energy.write(OUT/'contract_audit.json',result);print('phase22 audit',result['passed'],len(checks),'weighterr',maxweight,'sensitive65/129',len(sensitive),flush=True)
    assert result['passed'],[c for c in checks if not c['passed']][:20]


if __name__=='__main__':main()
