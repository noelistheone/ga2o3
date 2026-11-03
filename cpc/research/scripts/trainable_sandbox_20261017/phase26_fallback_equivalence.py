"""Read-only old/new method equivalence where both select the same reference."""
from pathlib import Path
import sys,json,argparse
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from csv_io_v2 import read_result_csv as read
import phase2_energy as e

BASE=e.BASE/'phase26_pool_share';OLD=e.BASE/'phase25_shared_process'
OUT=BASE/'fallback_equivalence';PRIMARY='ShareEffectiveProcess_internal';CONTROL='InventoryProcess_internal'


def declare():
    source=BASE/'prediction/plans.json';plans=json.loads(source.read_text())
    oldsel=json.loads((OLD/'prediction/selections.json').read_text())
    old0={(r['scheme'],r['held_unit']):r for r in oldsel if r['variant']==CONTROL and r['budget']==0}
    data=read(OLD/'data/conditions33.csv');contexts=[];expected=0;fitpaths=[]
    for plan in plans:
        if plan['variant']!=PRIMARY or plan['reg'] is not None:continue
        key=(plan['scheme'],plan['held_unit']);old=old0[key];assert old['reg'] is None
        newpath=ROOT/plan['fit_path'];oldpath=ROOT/old['fit_path'];nf=json.loads(newpath.read_text());of=json.loads(oldpath.read_text())
        assert nf['evaluation_method']=='ParentReference' and nf['reference_fallback']
        assert len(nf['parameters'])==2 and len(of['parameters'])==6
        held=plan['held_unit'];col='family' if plan['scheme']=='DOI' else 'related_cluster'
        count=0 if held is None else int(data[col].eq(held).sum())*3;expected+=count;fitpaths.extend([newpath,oldpath])
        contexts.append(dict(scheme=key[0],held_unit=key[1],new_fit_path=plan['fit_path'],old_fit_path=old['fit_path'],expected_point_rows=count))
    assert len(contexts)==11 and expected==153,(len(contexts),expected)
    inputs=[source,BASE/'prediction/shared_choices.json',OLD/'prediction/selections.json',OLD/'prediction/predictions.csv',
        OLD/'data/conditions33.csv',BASE/'pre_evaluation_protocol.json',BASE/'inner_shared_selection_readback.json']+fitpaths
    value=dict(created_utc=e.now(),purpose='same selected reference+same input/sourceprotocol implies matching outputs;not newmaterial validation',
        primary=PRIMARY,control=CONTROL,contexts=contexts,expected_contexts=11,expected_outer_point_rows=153,
        property_tolerance_dex=2e-6,parameter_tolerance=1e-12,
        source_policy_equality='reported separately;minor tie/roundoff policy changes are not a newphysical mechanism',
        declared_before_outer_file=not (BASE/'prediction/execution_complete.json').exists(),
        no_fits_or_new_property_solves=True,no_target_or_model_changes=True,retrospective=True,goal_complete=False,
        code_hashes={str(Path(__file__).relative_to(ROOT)):e.sha(Path(__file__))},
        input_hashes={str(p.relative_to(ROOT)):e.sha(p) for p in inputs})
    dest=OUT/'protocol_before_comparison.json'
    if dest.exists():
        old=json.loads(dest.read_text())
        for k in ('code_hashes','input_hashes','contexts'):assert old[k]==value[k],k
        return old
    e.write(dest,value);return value


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--declare-only',action='store_true');a=ap.parse_args();p=declare()
    if a.declare_only:
        print('Frozen reference-equivalence readback',len(p['contexts']),'contexts',p['expected_outer_point_rows'],'pointrows;no score comparison yet',flush=True);return
    assert json.loads((BASE/'prediction/audit.json').read_text())['passed']
    for kind in ('code_hashes','input_hashes'):
        for name,h in p[kind].items():assert e.sha(ROOT/name)==h,(kind,name)
    new=read(BASE/'prediction/predictions.csv');old=read(OLD/'prediction/predictions.csv')
    newsel=json.loads((BASE/'prediction/selections.json').read_text());oldsel=json.loads((OLD/'prediction/selections.json').read_text())
    nindex={(r['scheme'],r['held_unit'],r['budget']):r for r in newsel if r['variant']==PRIMARY}
    oindex={(r['scheme'],r['held_unit'],r['budget']):r for r in oldsel if r['variant']==CONTROL}
    checks=[];policies=[];rows=[];maximum=0.
    def check(name,x,tol=0.):checks.append(dict(name=name,error=float(x),tolerance=tol,passed=bool(x<=tol)))
    props=['predicted','predicted_log_mu','predicted_log_sigma','predicted_log_mu_CT','predicted_log_sigma_CT']
    for context in p['contexts']:
        scheme=context['scheme'];held=context['held_unit'];label=scheme+'/'+str(held)
        nf=json.loads((ROOT/context['new_fit_path']).read_text());of=json.loads((ROOT/context['old_fit_path']).read_text())
        check(label+'/same_base_parameters',np.max(abs(np.asarray(nf['parameters'])-np.asarray(of['parameters'][:2]))),p['parameter_tolerance'])
        check(label+'/old_shared_field_exact_zero',np.max(abs(np.asarray(of['parameters'][2:]))),0.)
        for b in (0,1,2):
            ns=nindex[scheme,held,b];os=oindex[scheme,held,b]
            policies.append(dict(scheme=scheme,held_unit=held,budget=b,new_lambda=ns['selected_source']['lam'],
                old_lambda=os['selected_source']['lam'],same_lambda=ns['selected_source']['lam']==os['selected_source']['lam']))
        if held is None:continue
        keys=['budget','family','sample_id']
        n=new[new.variant.eq(PRIMARY)&new.scheme.eq(scheme)&new.held_unit.eq(held)].set_index(keys).sort_index()
        o=old[old.variant.eq(CONTROL)&old.scheme.eq(scheme)&old.held_unit.eq(held)].set_index(keys).sort_index()
        check(label+'/row_ids',float(not n.index.equals(o.index)));check(label+'/row_count',abs(len(n)-context['expected_point_rows']))
        for key in n.index:
            nr=n.loc[key];rr=o.loc[key];tag=label+'/'+str(key)
            for flag in ('numeric','valid','role','common_query'):check(tag+'/'+flag,float(nr[flag]!=rr[flag]))
            check(tag+'/same_target_record',abs(nr.observed-rr.observed),1e-12)
            difference=None
            if nr.numeric and nr.valid and rr.numeric and rr.valid:
                difference=float(max(abs(float(nr[c])-float(rr[c])) for c in props));maximum=max(maximum,difference)
                check(tag+'/same_five_property_readouts',difference,p['property_tolerance_dex'])
            rows.append(dict(scheme=scheme,held_unit=held,budget=int(key[0]),family=key[1],sample_id=key[2],
                numeric=bool(nr.numeric),valid=bool(nr.valid),maximum_property_difference=difference))
    check('all_expected_points',abs(len(rows)-p['expected_outer_point_rows']))
    e.csvwrite(OUT/'point_equivalence.csv',rows);e.csvwrite(OUT/'source_policy_equivalence.csv',policies)
    result=dict(passed=all(c['passed'] for c in checks),checks=len(checks),rows=checks,point_rows=len(rows),contexts=len(p['contexts']),
        maximum_property_difference=maximum,source_policy_mismatches=sum(not r['same_lambda'] for r in policies),
        policies_compared=len(policies),material_accuracy_or_new_shared_increment_established=False,
        grade='saved result readback under explicit reference fallback;no new data/fits/forward solves',
        goal_complete=False,code_hashes=p['code_hashes'],input_hashes=dict(p['input_hashes'],
            **{str((BASE/'prediction/predictions.csv').relative_to(ROOT)):e.sha(BASE/'prediction/predictions.csv'),
               str((BASE/'prediction/selections.json').relative_to(ROOT)):e.sha(BASE/'prediction/selections.json')}))
    e.write(OUT/'audit.json',result)
    print('Actual reference-fallback readback',result['passed'],len(rows),'points;max',maximum,
        'policy differences',result['source_policy_mismatches'],flush=True)
    if not result['passed']:raise RuntimeError('Preserve reference-equivalence discrepancy before diagnosis or versioned repair')


if __name__=='__main__':main()
