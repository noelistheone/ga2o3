"""Positive parent inventories, foreign-atom-preserving allocation and cold IFT.

Growth inventories are conditional endpoints. No microscopic reaction rate or
closed Ga/O exchange path is inferred. Frozen earlier layers remain unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from . import thermo
from .donor_mixture import DonorMixtureLayer
from .implicit_defects import _charge_log,ImplicitDomainError,LN10


@dataclass
class ParentReference:
    totals:np.ndarray
    jacobian:np.ndarray
    diagnostics:dict


@dataclass
class InventoryState:
    totals:np.ndarray
    jacobian:np.ndarray
    diagnostics:dict


@dataclass
class FixedChargePrediction:
    values:np.ndarray
    jacobian:np.ndarray
    diagnostics:dict
    populations:np.ndarray
    probabilities:np.ndarray
    ef_eV:float


class FixedInventoryCharge:
    def __init__(self,template:DonorMixtureLayer):
        if not isinstance(template,DonorMixtureLayer):raise TypeError('Audited donor-mixture template required')
        self.layer=template;self.cold_calls=0;self.cold_evaluations=0;self.reference_calls=0;self._reference_cache=None
        dm=template.db.dm[template.first];self.native=np.all(dm[:,2:]==0,axis=1)
        X=template.donor_element;ix=template.db.col(X);k=dm[:,ix]
        other=[j for j in range(2,len(template.db.elements)) if j!=ix]
        self.complex=(k>0)&(dm[:,0]<-k)&(dm[:,1]==0)&np.all(dm[:,other]==0,axis=1)
        ids=template.original_parent_ids[template.first]
        self.receivers=np.isin(ids,template.donor_original_ids)
        self.dopant_count=k.astype(float);self.parent_atoms=template.atoms[template.first]
        if not self.complex.any() or self.receivers.sum()!=2:raise ValueError('Audited vacancy complexes and two isolated receivers required')
        expected=np.zeros(len(template.db.elements));expected[0]=-1;expected[ix]=1
        if not np.all(dm[self.receivers]==expected):raise ValueError('Receiver stoichiometry changed')
        if np.any(self.native&self.complex) or np.any(self.native&self.receivers):raise ValueError('Disjoint allocation classes required')

    def reference(self,pars):
        l=self.layer;p,T,_,doses=l._unpack(pars);key=tuple(np.delete(p,l.iobs));self.reference_calls+=1
        if self._reference_cache is not None and self._reference_cache[0]==key:return self._reference_cache[1]
        hot,block=l._hot_solution(p);initial_residual=float(max(abs(block[0])));polish_steps=0
        # The frozen generic hot solver can accept 2e-9 log residual. That is
        # insufficient when the cold free charge is a tiny difference of stocks.
        # Polish the same equations locally; never edit or relax the old solver.
        for _ in range(6):
            residual=float(max(abs(block[0])))
            if residual<=5e-14:break
            step=np.linalg.solve(block[1],block[0]);best=(residual,hot,block)
            for scale in (1.,.5,.25,.125):
                candidate=hot-scale*step;trial=l._hot(candidate,p);error=float(max(abs(trial[0])))
                if error<best[0]:best=(error,candidate,trial)
            if best[0]>=residual:break
            _,hot,block=best;polish_steps+=1
        F,J,P,N,NU,NP,meta=block;l.last_hot=hot.copy()
        up=-np.linalg.solve(J,P);dN=NP+NU@up
        c=np.bincount(l.parent,weights=N,minlength=l.db.nD)
        dc=np.array([np.bincount(l.parent,weights=v,minlength=l.db.nD) for v in dN.T]).T
        foreign=self.parent_atoms.T@c;condition=float(np.linalg.cond(J))
        occupancy=c/l.db.prefactor[l.first]
        reasons=[]
        if max(abs(F))>5e-12:reasons.append('reference_hot_precision_residual')
        if not np.isfinite(condition) or condition>1e12:reasons.append('reference_hot_condition')
        if occupancy.max()>1:reasons.append('reference_site_population')
        if any(d['clipped'] for d in meta['donors']):reasons.append('reference_population_clip')
        if max(abs(foreign/doses-1))>3e-8:reasons.append('reference_foreign_atom_budget')
        if abs(T-900)<1e-6:reasons.append('reference_thermochemistry_kink')
        if reasons:raise ImplicitDomainError(';'.join(reasons))
        ref=ParentReference(c,dc,dict(hot_root_residual=float(max(abs(F))),hot_jacobian_condition=condition,
            initial_hot_root_residual=initial_residual,hot_polish_steps=polish_steps,
            reference_max_parent_fraction=float(occupancy.max()),Tf_K=float(T),reference_hot_solve='original_donor_mixture'))
        self._reference_cache=(key,ref);return ref

    def allocate(self,pars,parameter_chain,*,native_s=0.,complex_s=0.,native_s_jacobian=None,complex_s_jacobian=None):
        l=self.layer;p,_,_,doses=l._unpack(pars);chain=np.asarray(parameter_chain,float)
        if chain.ndim!=2 or chain.shape[0]!=l.np or not np.isfinite(chain).all():raise ValueError('Explicit raw-to-input derivative matrix required')
        nv=chain.shape[1];sn=float(native_s);sc=float(complex_s)
        if not np.isfinite([sn,sc]).all() or min(sn,sc)<0 or max(sn,sc)>120:raise ValueError('Suppression coordinates must lie in[0,120]')
        dn=np.zeros(nv) if native_s_jacobian is None else np.asarray(native_s_jacobian,float)
        ds=np.zeros(nv) if complex_s_jacobian is None else np.asarray(complex_s_jacobian,float)
        if dn.shape!=(nv,) or ds.shape!=(nv,) or not np.isfinite(np.r_[dn,ds]).all():raise ValueError('Finite allocation derivatives required')
        ref=self.reference(p);c0=ref.totals;dc0=ref.jacobian@chain;c=c0.copy();dc=dc0.copy()
        yn=np.exp(-sn);yc=np.exp(-sc);alpha=-np.expm1(-sc)
        c[self.native]=yn*c0[self.native]
        dc[self.native]=yn*(dc0[self.native]-c0[self.native,None]*dn)
        # Lost complex dopant atoms enter the same two existing isolated sites.
        # Their reference ratio and its derivative are both retained.
        ix=self.receivers;pool=float(c0[ix].sum())
        if not np.isfinite(pool) or pool<=0:raise ImplicitDomainError('Positive isolated reference pool required')
        r=c0[ix]/pool;dr=(dc0[ix]-r[:,None]*dc0[ix].sum(axis=0))/pool
        amount=float(self.dopant_count[self.complex]@c0[self.complex])
        damount=self.dopant_count[self.complex]@dc0[self.complex]
        c[self.complex]=yc*c0[self.complex]
        dc[self.complex]=yc*(dc0[self.complex]-c0[self.complex,None]*ds)
        c[ix]+=alpha*amount*r
        dc[ix]+=alpha*(amount*dr+r[:,None]*damount)+yc*amount*r[:,None]*ds
        achieved=self.parent_atoms.T@c
        expected_jac=doses[:,None]*LN10*chain[2+l.b:2+l.b+l.k]
        budget_error=float(max(abs(achieved/doses-1)))
        derivative_error=float(np.max(abs(self.parent_atoms.T@dc-expected_jac)/(1+doses[:,None]*LN10)))
        if np.any(c<0) or not np.isfinite(c).all() or not np.isfinite(dc).all():raise ImplicitDomainError('Invalid parent inventory')
        if budget_error>3e-8 or derivative_error>3e-7:raise ImplicitDomainError('Foreign-atom inventory/derivative constraint failed')
        occupancy=c/l.db.prefactor[l.first]
        foreign=np.any(self.parent_atoms>0,axis=1)
        if occupancy.max()>1 or occupancy[foreign].max()>=.1*(1-1e-6):raise ImplicitDomainError('Outside parent dilution domain')
        host_change=l.db.dm[l.first,:2].T@(c-c0)
        state=InventoryState(c,dc,dict(**ref.diagnostics,native_survival=float(yn),complex_survival=float(yc),
            complex_reallocation_fraction=float(alpha),complex_dopant_pool_cm3=amount,receiver_fractions=r.tolist(),
            foreign_atom_relative_error=budget_error,foreign_atom_derivative_error=derivative_error,
            host_Ga_O_inventory_change_cm3=host_change.tolist(),host_atom_exchange_scope='open growth endpoint;not closed Ga/O conservation or measured pathway',
            max_parent_fraction=float(occupancy.max()),allocation_interpretation='shared conditional endpoint speciation;not a microscopic rate'))
        self.last_inventory=state;return state

    def solve(self,pars,inventory:InventoryState,parameter_chain):
        l=self.layer;p,_,Nd,doses=l._unpack(pars);chain=np.asarray(parameter_chain,float);c=inventory.totals;dc=inventory.jacobian
        if chain.shape!=(l.np,dc.shape[1]):raise ValueError('Parameter and inventory chains must agree')
        self.cold_calls+=1;T=10**p[l.iobs];beta=1/(thermo.KB_EV*T)
        H,HP,HT,profile=l._reference(T,p);s,ds=l._entropy(T)
        A=H-3*thermo.KB_EV*T*s*l.db.sum_dm;AT=HT-3*thermo.KB_EV*(s+T*ds)*l.db.sum_dm
        ell,ellp=l.fraction(p)
        def populations(ef):
            z=-(A+l.q*ef)*beta;logw=z+l.logg
            maxima=np.full(l.db.nD,-np.inf);np.maximum.at(maxima,l.parent,logw)
            w=np.exp(logw-maxima[l.parent]);pi=w/np.bincount(l.parent,weights=w,minlength=l.db.nD)[l.parent]
            pairs=[]
            for plus,neutral in l.pairs:
                d=logw[neutral]-logw[plus];f=float(expit(d));fp=float(expit(-d))
                pi[neutral]=ell*f;pi[plus]=(1-ell)+ell*fp;pairs.append((f,fp))
            return c[l.parent]*pi,pi,z,pairs
        def charge(ef):
            self.cold_evaluations+=1;N,_,_,_=populations(ef);car=profile.carriers(ef,T)[0]
            pos=np.maximum(l.q,0)@N+car[1:].sum()+Nd;neg=np.maximum(-l.q,0)@N+car[0]
            return (pos-neg)/max(pos,neg,1.)
        ef=brentq(charge,-5.,12.,xtol=2e-14,rtol=1e-15)
        N,pi,z,pairs=populations(ef);mq=np.bincount(l.parent,weights=pi*l.q,minlength=l.db.nD)
        NE=-beta*N*(l.q-mq[l.parent])
        LP=-beta*HP;LP[:,l.iobs]+=(-beta*AT-z/T)*T*LN10
        mLP=np.array([np.bincount(l.parent,weights=pi*a,minlength=l.db.nD) for a in LP.T]).T
        NP=pi[:,None]*dc[l.parent]+N[:,None]*((LP-mLP[l.parent])@chain)
        susceptibility=np.bincount(l.parent,weights=pi*(l.q-mq[l.parent])**2,minlength=l.db.nD)
        donor=[]
        for (plus,neutral),(f,fp) in zip(l.pairs,pairs):
            parent=l.parent[plus];chi=ell*f*fp;dp0=(f*ellp+chi*(LP[neutral]-LP[plus]))@chain
            NE[neutral]=beta*c[parent]*chi;NE[plus]=-beta*c[parent]*chi
            NP[neutral]=pi[neutral]*dc[parent]+c[parent]*dp0
            NP[plus]=pi[plus]*dc[parent]-c[parent]*dp0
            susceptibility[parent]=chi
            donor.append(dict(parent=int(l.original_parent_ids[plus]),total=float(c[parent]),localized_fraction=ell,
                local_neutral_probability=f,mean_charge=float(pi[plus]),actual_q_squared=float(pi[plus]),conditional_susceptibility=chi))
        car,ce,ct,cm=profile.carriers(ef,T)
        if car[0]<=0:raise ImplicitDomainError('Free electron density underflows')
        raw_car=np.zeros((4,l.np));raw_car[:,l.iobs]=ct*T*LN10;raw_car[:,l.imat]=cm
        CP=raw_car@chain;dNd=Nd*LN10*chain[1]
        residual,J,P,M=_charge_log(N,np.c_[NE,NP],car,np.c_[ce,CP],l.q,Nd,np.r_[0.,dNd])
        if J[0]>=0 or not np.isfinite(J).all():raise ImplicitDomainError('Nonregular cold charge root')
        ef_jac=-J[1:]/J[0];dN=NP+NE[:,None]*ef_jac;dcar=CP+ce[:,None]*ef_jac
        QE=float(ce[1:].sum()-ce[0]-beta*(c@susceptibility))
        if QE>=0:raise ImplicitDomainError('Nondecreasing cold charge')
        values=[np.log10(car[0])];jac=[dcar[0]/(car[0]*LN10)];budgets=[]
        for weights in (abs(l.q),l.q**2):
            value=float(weights@N+Nd);budgets.append(value);values.append(np.log10(value));jac.append((weights@dN+dNd)/(value*LN10))
        prob_error=float(max(abs(np.bincount(l.parent,weights=pi,minlength=l.db.nD)-1)))
        cold_budget=l.atoms.T@N
        if abs(residual)>2e-8 or prob_error>2e-12 or max(abs(cold_budget/doses-1))>3e-8:raise ImplicitDomainError('Cold root/probability/atom contract failed')
        result=FixedChargePrediction(np.array(values),np.array(jac),dict(**inventory.diagnostics,
            root_residual_log=float(max(abs(residual),inventory.diagnostics['hot_root_residual'])),
            jacobian_condition=inventory.diagnostics['hot_jacobian_condition'],cold_logcharge_EF_slope=float(J[0]),
            cold_Q_EF_cm3_per_eV=QE,free_n_cm3=float(car[0]),free_p_cm3=float(car[1]),measurement_T_K=float(T),Nd_cm3=float(Nd),
            charge_relative_residual=float(abs(P-M)/max(P,M)),parent_probability_error=prob_error,
            single_electron_Hall_valid=bool(car[0]>car[1]),n_gt_p_does_not_establish_single_band=True,
            ionized_absolute_charge_cm3=budgets[0],charge_squared_scatter_budget_cm3=budgets[1],
            achieved_concentrations_cm3=dict(zip(l.elements,(self.parent_atoms.T@c).tolist())),
            cold_achieved_concentrations_cm3=dict(zip(l.elements,cold_budget.tolist())),
            profile_sha256=profile.fingerprint,localized_fraction=ell,donor_subgroups=donor,
            gradient_valid=True,material_Born_validity_established=False),N,pi,float(ef))
        self.last_prediction=result;return result
