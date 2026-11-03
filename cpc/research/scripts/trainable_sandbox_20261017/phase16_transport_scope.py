"""Energy-resolved semiclassical scope diagnostic, without filtering scores."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy.special import expit
from sandbox.coherent_fermi import published_coherent_profile,reference
from sandbox.ionized_transport import E_CHARGE,EPSILON0,K_B_J,HBAR,M_E,coulomb_log
import phase16_prediction as study
import phase2_energy as energy

OUT=study.OUT/'transport_scope'


def moments(n,NI,T,order=512):
    profile=published_coherent_profile();mass=profile.electron_mass_ratio*M_E;eps=10.2*EPSILON0;kT=K_B_J*T
    Nc=profile.Nc_ref*(T/profile.T_ref)**1.5;eta=reference().eta_from_logF(np.log(n/Nc));_,g,_=reference().log(eta)
    nodes,weight=np.polynomial.legendre.leggauss(order);lo=np.sqrt(max(0.,eta-50));hi=np.sqrt(max(60.,eta+50))
    t=(nodes+1)*(hi-lo)/2+lo;x=t*t;weight*=((hi-lo)/2)
    p=weight*2*t**4*expit(eta-x)*expit(x-eta)/expit(eta);p/=p.sum()
    qs2=E_CHARGE**2*n*g*1e6/(eps*kT)
    b=8*mass*kT*x/(HBAR**2*qs2)
    pref=NI*1e6*E_CHARGE**4/(16*np.sqrt(2)*np.pi*eps**2*np.sqrt(mass)*kT**1.5)
    rate=pref*x**(-1.5)*coulomb_log(b)+E_CHARGE/(mass*205e-4)
    tau=1/rate
    # k=sqrt(2mE)/hbar, l=v*tau=sqrt(2E/m)*tau, hence kl=2E*tau/hbar.
    kl=2*x*kT*tau/HBAR;sigma_weights=p*tau;sigma_weights/=sigma_weights.sum()
    hall_weights=p*tau**2;hall_weights/=hall_weights.sum()
    mud=E_CHARGE/mass*(p@tau)*1e4;rh=(p@(tau**2))/(p@tau)**2
    return dict(eta=float(eta),reconstructed_log_muH=float(np.log10(mud*rh)),
        fraction_sigma_kell_lt1=float(sigma_weights[kl<1].sum()),fraction_Hall_kell_lt1=float(hall_weights[kl<1].sum()),
        sigma_weighted_kell=float(sigma_weights@kl))


def main():
    runtime=study.OUT/'research_runtime';assert json.loads((runtime/'verification.json').read_text())['passed']
    protocol=dict(code_sha256=energy.sha(Path(__file__)),
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (runtime/'verification.json',runtime/'independent_outputs.json',runtime/'index.json')},
        reason='newmu supervision may reduceerror by pushing an elastic Born/RTA band model into strongscattering;check actual predicted currentmu states',
        diagnostic='energyresolved kl=2E tau/hbar;fractions of sigma and Hall momentweights below1',
        scope='semiclassical selfconsistency diagnostic,not rigorous universal transport bound or proofBornvalidwhenkl>1',
        controls='no fit,filter,modelchoice,threshold tuning or score change',
        background_source=dict(url='https://arxiv.org/abs/cond-mat/0305412',doi='10.1103/RevModPhys.75.1085',
            reading_depth='abstract retrieved;no fullpaper review claimed;our kl identity directly derived'),goal_complete=False)
    pp=OUT/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==energy.clean(protocol)
    else:energy.write(pp,protocol)
    index={r['task_id']:r for r in json.loads((runtime/'index.json').read_text())}
    targets=study.TARGETS[study.TARGETS.endpoint.eq('muH')].set_index('obs_id');rows=[];cache={}
    for r in json.loads((runtime/'independent_outputs.json').read_text())['rows']:
        plan=index[r['task_id']]
        if plan['budget']!=0 or r['obs_id'] not in targets.index:continue
        o=targets.loc[r['obs_id']];n=r['free_n_cm3'];NI=10**r['values'][2];T=float(o.Tobs_K)
        key=(float(n),float(NI),T)
        if key not in cache:cache[key]=moments(n,NI,T)
        value=cache[key];error=abs(value['reconstructed_log_muH']-r['transport_values'][0])
        assert error<1e-6,(plan['method'],r['obs_id'],error)
        rows.append(dict(method=plan['method'],obs_id=r['obs_id'],sample_id=o.sample_id,family=o.family,kind=o.kind,T_K=T,
            predicted_log_muH=r['transport_values'][0],moment_replay_error=error,**value))
    data=pd.DataFrame(rows);assert len(data)==708
    data.to_csv(OUT/'diagnostics.csv',index=False)
    summary=data.groupby(['method','kind']).agg(states=('obs_id','count'),
        sigma_fraction_lt1_mean=('fraction_sigma_kell_lt1','mean'),sigma_fraction_lt1_max=('fraction_sigma_kell_lt1','max'),
        states_majority_sigma_below1=('fraction_sigma_kell_lt1',lambda s:int((s>.5).sum()))).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    energy.write(OUT/'contract_audit.json',dict(passed=True,states=len(data),RT_states=int(data.kind.eq('RT').sum()),
        temperature_states=int(data.kind.eq('curve').sum()),maximum_moment_replay_error=float(data.moment_replay_error.max()),
        states_majority_sigma_kell_below1=int(data.fraction_sigma_kell_lt1.gt(.5).sum()),
        removed_rows=0,new_fits=0,all_prediction_metrics_unchanged=True,material_Born_validity_established=False,goal_complete=False))
    print('Scope diagnostic',len(data),'states,majority_sigma_kl<1',int(data.fraction_sigma_kell_lt1.gt(.5).sum()))


if __name__=='__main__':main()
