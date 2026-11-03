"""Actual saved-state/scalar-carrier/finite-difference gates for local charge."""
from pathlib import Path
from types import SimpleNamespace
import sys,numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase30_core as c
from spatial_local_charge_v1 import FrozenChargeKernel
from sandbox.joint_saved_model import SavedJointModel

OUT=c.BASE/'spatial_predictive_candidate/local_charge_gates_v2'


def main():
    assert not (OUT/'protocol_before.json').exists()
    plans=c.document(c.OLD/'prediction/plan.json')['plans'];table=c.cases();selected=[]
    for variant in ['Reference205_internal','JointProcess_nHonly_internal']:
        for element in ['Si','Sn']:
            row=next(table[table.element.eq(element)&table.mobility_eligible.eq(True)].itertuples())
            p=next(p for p in plans if p['variant']==variant and p['outer_held']==row.related_cluster)
            selected.append((p,row))
        hydrogen=table[table.H_known.eq(True)]
        if len(hydrogen):
            row=next(hydrogen.itertuples());p=next(p for p in plans if p['variant']==variant and p['outer_held']==row.related_cluster)
            selected.append((p,row))
    code=[Path(__file__),Path(__file__).with_name('spatial_local_charge_v1.py')]+list((c.ROOT/'src/sandbox').glob('*.py'))
    inputs=[c.OLD/'prediction/plan.json',c.OLD/'data/scenarios49.csv',c.BASE/'spatial_predictive_candidate/local_charge_gates_v1/gates.json']+[c.ROOT/p['fit_path'] for p,r in selected]
    c.write(OUT/'protocol_before.json',dict(created_utc=c.e.now(),code_hashes=c.file_hashes(code),
        input_hashes=c.file_hashes(inputs),selected_cases=[dict(variant=p['variant'],case_id=r.case_id) for p,r in selected],
        labels_not_used_for_charge_tests=True,
        FD_revision='fourth-order symmetric h3e-4;reduce cancellation in netQ;core unchanged,Qu threshold tightened',new_models_fitted=0,goal_complete=False))
    checks=[];details=[]
    for plan,row in selected:
        mapper=SavedJointModel(plan['minimal_fit']);point=mapper.point(SimpleNamespace(**c.design.inputs(row)),require_observation=False)
        engine=(mapper.engine.engine_for(row) if hasattr(mapper.engine,'engine_for') else mapper.engine.engines[row.scenario_pO2_atm])
        layer,solver=engine.layer_for(row);inv=solver.last_inventory;old=solver.last_prediction
        Nd=old.diagnostics['Nd_cm3']
        pars=layer.parameters(T=point.diagnostics['Tf_K'],Nd=Nd,weights=[0.],
            doses=(row.incorporated_cm3,row.H_cm3) if row.H_known else (row.incorporated_cm3,),
            pO2=row.scenario_pO2_atm,Tobs=row.Tobs_K,binding_eV=.0366,screen_s=0.)
        kernel=FrozenChargeKernel(layer,pars)
        u=np.array([old.ef_eV]);stocks=inv.totals[:,None]
        a=kernel.evaluate(u,stocks,background_cm3=Nd)
        probabilities=a.probabilities[layer.parent,kernel.slot,0]
        pe=float(max(abs(probabilities-old.probabilities)))
        values=np.log10([a.carriers_cm3[0,0],a.Nabs_cm3[0],a.Nq2_cm3[0]])
        ve=float(max(abs(values-old.values)))
        qe=float(abs(a.charge_EF_partial[0]-old.diagnostics['cold_Q_EF_cm3_per_eV'])/(1+abs(old.diagnostics['cold_Q_EF_cm3_per_eV'])))
        checks.extend([pe<2e-12,ve<2e-11,qe<2e-11])
        positions=old.ef_eV+np.linspace(-.35,.35,9)
        stocks=np.broadcast_to(inv.totals[:,None],(len(inv.totals),len(positions))).copy()
        a=kernel.evaluate(positions,stocks,background_cm3=Nd)
        expected=np.array([kernel.profile.carriers(float(v),kernel.T)[0] for v in positions])
        expected_j=np.array([kernel.profile.carriers(float(v),kernel.T)[1] for v in positions])
        ce=float(np.max(abs(a.carriers_cm3-expected)/(1+abs(expected))))
        de=float(np.max(abs(a.carrier_EF_partial-expected_j)/(1+abs(expected_j))))
        checks.extend([ce<2e-12,de<2e-12,np.all(a.charge_EF_partial<0),a.diagnostics['max_parent_probability_error']<2e-12])
        h=3e-4
        plus=kernel.evaluate(positions+h,stocks,background_cm3=Nd)
        minus=kernel.evaluate(positions-h,stocks,background_cm3=Nd)
        plus2=kernel.evaluate(positions+2*h,stocks,background_cm3=Nd)
        minus2=kernel.evaluate(positions-2*h,stocks,background_cm3=Nd)
        fd=(-plus2.charge_cm3+8*plus.charge_cm3-8*minus.charge_cm3+minus2.charge_cm3)/(12*h)
        qerr=float(max(abs(fd-a.charge_EF_partial)/(1+abs(fd)+abs(a.charge_EF_partial))))
        fdni=(-plus2.Nq2_cm3+8*plus.Nq2_cm3-8*minus.Nq2_cm3+minus2.Nq2_cm3)/(12*h)
        nierr=float(max(abs(fdni-a.Nq2_EF_partial)/(1+a.Nq2_cm3+abs(fdni)+abs(a.Nq2_EF_partial))))
        fdna=(-plus2.Nabs_cm3+8*plus.Nabs_cm3-8*minus.Nabs_cm3+minus2.Nabs_cm3)/(12*h)
        naerr=float(max(abs(fdna-a.Nabs_EF_partial)/(1+a.Nabs_cm3+abs(fdna)+abs(a.Nabs_EF_partial))))
        checks.extend([qerr<1e-8,nierr<2e-7,naerr<2e-7])
        parent=int(np.argmax(inv.totals));stocks_p=stocks.copy();stocks_m=stocks.copy()
        stocks_p[parent]*=np.exp(2e-5);stocks_m[parent]*=np.exp(-2e-5)
        pp=kernel.evaluate(positions,stocks_p,background_cm3=Nd)
        mm=kernel.evaluate(positions,stocks_m,background_cm3=Nd)
        fd=(pp.charge_cm3-mm.charge_cm3)/(4e-5)
        direct=stocks[parent]*a.mean_q[parent]
        eparent=float(max(abs(fd-direct)/(1+abs(fd)+abs(direct)+a.Nabs_cm3)))
        checks.append(eparent<2e-7)
        shift=float(max(a.Nq2_cm3)*.1)
        shifted=kernel.evaluate(positions,stocks,background_cm3=Nd+shift)
        be=float(max(abs(shifted.charge_cm3-a.charge_cm3-shift)/(shift+a.Nabs_cm3)))
        bni=float(max(abs(shifted.Nq2_cm3-a.Nq2_cm3-shift)/(shift+a.Nq2_cm3)))
        checks.extend([be<2e-12,bni<2e-12])
        details.append(dict(variant=plan['variant'],case_id=row.case_id,H_known=bool(row.H_known),
            probability_error=pe,old_cold_value_error=ve,old_charge_slope_error=qe,
            scalar_carrier_error=ce,scalar_carrier_derivative_error=de,
            charge_EF_FD_error=qerr,Nq2_EF_FD_error=nierr,Nabs_EF_FD_error=naerr,
            stock_FD_error=eparent,background_charge_error=be,background_q2_error=bni))
    c.verify_hashes(c.document(OUT/'protocol_before.json'))
    c.write(OUT/'details.json',details)
    c.write(OUT/'gates.json',dict(passed=all(checks),checks=len(checks),failures=sum(not bool(v) for v in checks),
        actual_saved_states=len(details),hydrogen_positive_states=sum(r['H_known'] for r in details),
        max_old_cold_value_error=max(r['old_cold_value_error'] for r in details),
        max_charge_EF_FD_error=max(r['charge_EF_FD_error'] for r in details),
        max_Nq2_EF_FD_error=max(r['Nq2_EF_FD_error'] for r in details),
        no_complete_Poisson_or_spatial_prediction_model_yet=True,new_models_fitted=0,goal_complete=False))
    print('Local vector charge gates',len(checks),'states',len(details),'failed',sum(not bool(v) for v in checks),flush=True)
    assert all(checks),'Preserve first local-charge gate before versioned repair'


if __name__=='__main__':main()
