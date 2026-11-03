"""Constrained localized/extended donor subgroups in the full defect layer.

The fraction depends only on supplied chemical dose. This is a conditional
statistical closure, not an impurity-band Hamiltonian or new DFT charge label.
Both hot inventory and cold charge response derive from the same free energy.
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit
from scipy.optimize import brentq
from . import thermo,transport
from .temperature_defects import TemperatureDefectLayer
from .implicit_defects import ImplicitPrediction,ImplicitDomainError,_charge_log,_thermo_derivatives,LN10


class DonorMixtureLayer(TemperatureDefectLayer):
    def __init__(self,db,elements=("Si",),*,fixed_fraction=None,**kwargs):
        if "donor_extension" in kwargs:raise ValueError("The donor pair is always explicit in this layer")
        super().__init__(db,elements,donor_extension=True,**kwargs)
        if fixed_fraction is not None and not (np.isfinite(fixed_fraction) and 0<=fixed_fraction<=1):
            raise ValueError("Fixed localized fraction must be in[0,1]")
        self._fixed_fraction=None if fixed_fraction is None else float(fixed_fraction)
        self.iloc=self.np;self.np+=2
        self.parameter_names+=("localized_logit_at_1e18","localized_slope_per_log10_dose")
        self.pairs=[]
        for parent in self.donor_original_ids:
            ix=np.flatnonzero(self.original_parent_ids==parent)
            plus=ix[self.q[ix]==1];neutral=ix[self.neutral[ix]]
            if len(plus)!=1 or len(neutral)!=1:raise ValueError("Expected audited +/0 pair")
            self.pairs.append((int(plus[0]),int(neutral[0])))

    @property
    def fixed_fraction(self):return self._fixed_fraction

    def parameters(self,*,localized_logit=0.,localized_slope=2.,**kwargs):
        return np.r_[super().parameters(**kwargs),localized_logit,localized_slope]

    def _unpack(self,pars):
        p,T,Nd,doses=super()._unpack(pars)
        if not (-30<=p[self.iloc]<=30 and 0<=p[self.iloc+1]<=20):
            raise ImplicitDomainError("Outside declared localization relation domain")
        return p,T,Nd,doses

    def fraction(self,pars):
        p=np.asarray(pars,float);derivative=np.zeros(self.np)
        if self.fixed_fraction is not None:return self.fixed_fraction,derivative
        x=p[self.idose]-18.;a,b=p[self.iloc:self.iloc+2]
        ell=float(expit(a-b*x));variance=ell*float(expit(-a+b*x))
        derivative[self.iloc]=variance;derivative[self.iloc+1]=-x*variance
        derivative[self.idose]=-b*variance
        return ell,derivative

    def _hot(self,u,pars):
        self.root_evaluations+=1;p,T,Nd,doses=self._unpack(pars)
        H,HP,HT,profile=self._reference(T,p);beta=1/(thermo.KB_EV*T)
        mu=np.full(len(self.db.elements),-30.);mu[:2]=thermo.mu_Ga_O(T,10**p[self.ipress]);mu[self.cols]=u[1:]
        s,ds=self._entropy(T)
        A=H-3*thermo.KB_EV*T*s*self.db.sum_dm-self.db.dm@mu
        x=-(A+self.q*u[0])*beta
        xu=np.column_stack([-self.q*beta,self.atoms*beta]);xp=-beta*HP
        dga,do,_=_thermo_derivatives(T,p[self.ipress])
        AT=HT-3*thermo.KB_EV*(s+T*ds)*self.db.sum_dm-self.db.dm[:,0]*dga-self.db.dm[:,1]*do
        xp[:,0]+=(-AT*beta-x/T)*T*LN10
        xp[:,self.ipress]+=(self.db.dm[:,1]-1.5*self.db.dm[:,0])*.5*LN10
        dx=np.c_[xu,xp];active=(x>-700)&(x<300)
        N=self.db.prefactor*np.exp(np.clip(x,-700,300));dn=N[:,None]*dx*active[:,None]
        ell,ellp=self.fraction(p);dell=np.r_[np.zeros(self.nhot),ellp]
        donor=[]
        for plus,neutral in self.pairs:
            logit=np.log(self.db.prefactor[neutral]/self.db.prefactor[plus])+x[neutral]-x[plus]
            h=np.logaddexp(0,logit);f=float(expit(logit));fp=float(expit(-logit));dlogit=dx[neutral]-dx[plus]
            xc=x[plus]+ell*h;ca=self.db.prefactor[plus]*np.exp(np.clip(xc,-700,300))
            dc=ca*(dx[plus]+ell*f*dlogit+h*dell)*bool(-700<xc<300)
            p0=ell*f;pp=(1-ell)+ell*fp;dp0=f*dell+ell*f*fp*dlogit
            N[plus]=ca*pp;N[neutral]=ca*p0
            dn[plus]=pp*dc-ca*dp0;dn[neutral]=p0*dc+ca*dp0
            donor.append(dict(parent=int(self.original_parent_ids[plus]),total=ca,localized_fraction=ell,
                neutral_fraction=p0,local_neutral_probability=f,mean_charge=pp,
                conditional_susceptibility=ell*f*fp,actual_q_squared=pp,
                clipped=not(-700<xc<300)))
        dU,dP=dn[:,:self.nhot],dn[:,self.nhot:]
        car,ce,ct,cm=profile.carriers(u[0],T)
        cU=np.zeros((4,self.nhot));cU[:,0]=ce
        cP=np.zeros((4,self.np));cP[:,0]=ct*T*LN10;cP[:,self.imat]=cm
        ndP=np.zeros(self.np);ndP[1]=Nd*LN10
        F0,J0,_,_=_charge_log(N,dn,car,np.c_[cU,cP],self.q,Nd,np.r_[np.zeros(self.nhot),ndP])
        C=self.atoms.T@N
        if np.any(C<=0):raise ImplicitDomainError("Nonpositive incorporated budget")
        F=np.r_[F0,np.log(C)-np.log(doses)]
        J=np.vstack([J0,self.atoms.T@dn/C[:,None]])
        for j in range(self.k):J[1+j,self.nhot+2+self.b+j]-=LN10
        qU=self.q@dU+cU[1:].sum(axis=0)-cU[0]
        raw_hessian=np.vstack([-qU,self.atoms.T@dU])
        return F,J[:,:self.nhot],J[:,self.nhot:],N,dU,dP,dict(
            donors=donor,raw_minus_charge_atom_jacobian=raw_hessian)

    def predict(self,pars,*,require_regular=True):
        self.root_calls+=1;p,T,Nd,doses=self._unpack(pars);To=10**p[self.iobs]
        hot,(Fh,Jh,Ph,Nh,NhU,NhP,hot_meta)=self._hot_solution(p)
        totals=np.bincount(self.parent,weights=Nh,minlength=self.db.nD)
        tU=np.array([np.bincount(self.parent,weights=a,minlength=self.db.nD) for a in NhU.T]).T
        tP=np.array([np.bincount(self.parent,weights=a,minlength=self.db.nD) for a in NhP.T]).T
        H,HP,HT,profile=self._reference(To,p);beta=1/(thermo.KB_EV*To);s,ds=self._entropy(To)
        A=H-3*thermo.KB_EV*To*s*self.db.sum_dm
        AT=HT-3*thermo.KB_EV*(s+To*ds)*self.db.sum_dm
        ell,ellp=self.fraction(p)
        def populations(ef):
            z=-(A+self.q*ef)*beta;logw=z+self.logg
            mx=np.full(self.db.nD,-np.inf);np.maximum.at(mx,self.parent,logw)
            w=np.exp(logw-mx[self.parent]);den=np.bincount(self.parent,weights=w,minlength=self.db.nD)
            pi=w/den[self.parent];f0=[]
            for plus,neutral in self.pairs:
                logit=logw[neutral]-logw[plus];f=float(expit(logit));fp=float(expit(-logit));p0=ell*f
                pi[neutral]=p0;pi[plus]=(1-ell)+ell*fp;f0.append((f,fp))
            return totals[self.parent]*pi,pi,z,f0
        def cold(ef):
            N,_,_,_=populations(ef);c=profile.carriers(ef,To)[0]
            pos=np.maximum(self.q,0)@N+c[1:].sum()+Nd
            neg=np.maximum(-self.q,0)@N+c[0]
            return (pos-neg)/max(pos,neg,1.)
        ef=brentq(cold,-5.,12.,xtol=2e-14,rtol=1e-15)
        N,pi,z,f0=populations(ef)
        mq=np.bincount(self.parent,weights=pi*self.q,minlength=self.db.nD)
        dNU=np.zeros((self.db.nCS,self.nu));dNU[:,:self.nhot]=pi[:,None]*tU[self.parent]
        dNU[:,-1]=-beta*N*(self.q-mq[self.parent])
        LP=-beta*HP;LP[:,self.iobs]+=(-beta*AT-z/To)*To*LN10
        mLP=np.array([np.bincount(self.parent,weights=pi*a,minlength=self.db.nD) for a in LP.T]).T
        dNP=pi[:,None]*tP[self.parent]+N[:,None]*(LP-mLP[self.parent])
        susceptibility=np.bincount(self.parent,weights=pi*(self.q-mq[self.parent])**2,minlength=self.db.nD)
        donor=[]
        for (plus,neutral),(f,fp) in zip(self.pairs,f0):
            parent=self.parent[plus];c=totals[parent];chi=ell*f*fp
            dp0=f*ellp+chi*(LP[neutral]-LP[plus])
            dNU[neutral,-1]=beta*c*chi;dNU[plus,-1]=-beta*c*chi
            dNP[neutral]=pi[neutral]*tP[parent]+c*dp0
            dNP[plus]=pi[plus]*tP[parent]-c*dp0
            susceptibility[parent]=chi
            donor.append(dict(parent=int(self.original_parent_ids[plus]),total=c,localized_fraction=ell,
                local_neutral_probability=f,mean_charge=pi[plus],actual_q_squared=pi[plus],
                mixture_variance=ell*f*(1-ell*f),conditional_susceptibility=chi))
        car,ce,ct,cm=profile.carriers(ef,To)
        if car[0]<=0:raise ImplicitDomainError("Free-electron density underflows in this observation state")
        cU=np.zeros((4,self.nu));cU[:,-1]=ce
        cP=np.zeros((4,self.np));cP[:,self.iobs]=ct*To*LN10;cP[:,self.imat]=cm
        ndP=np.zeros(self.np);ndP[1]=Nd*LN10
        Fc,Jc,P,M=_charge_log(N,np.c_[dNU,dNP],car,np.c_[cU,cP],self.q,Nd,np.r_[np.zeros(self.nu),ndP])
        Fu=np.zeros((self.nu,self.nu));Fu[:self.nhot,:self.nhot]=Jh;Fu[-1]=Jc[:self.nu]
        Fp=np.vstack([Ph,Jc[self.nu:]])
        root_deriv=-np.linalg.solve(Fu,Fp);condition=float(np.linalg.cond(Fu))
        dN=np.c_[dNU,dNP];dc=np.c_[cU,cP];n=car[0];logn=np.log10(n)
        rh=float(transport.hall_factor(To,n)) if self.hall=="heuristic" else 1.
        drh=-.5/2.7 if self.hall=="heuristic" and 17<logn<19.7 else 0.
        deriv=np.zeros((3,self.nu+self.np));deriv[0]=dc[0]/(n*LN10)*(1-drh/(rh*LN10))
        budgets=[]
        for i,charge in enumerate((abs(self.q),self.q**2)):
            value=float(charge@N+Nd);dv=charge@dN;dv[self.nu+1]+=Nd*LN10
            budgets.append(value);deriv[i+1]=dv/(value*LN10)
        jac=deriv[:,self.nu:]+deriv[:,:self.nu]@root_deriv
        values=np.r_[logn-np.log10(rh),np.log10(budgets)]
        residual=float(max(abs(Fh).max(),abs(Fc)));occ=Nh/self.db.prefactor
        parent_occ=totals/self.db.prefactor[self.first]
        dopant_occ={e:float(max(occ[self.atoms[:,j]>0].max(),
            parent_occ[self.db.dm[self.first,self.cols[j]]>0].max())) for j,e in enumerate(self.elements)}
        reasons=[]
        if residual>2e-8:reasons.append("root_residual")
        if not np.isfinite(condition) or condition>1e12:reasons.append("ill_conditioned_root")
        if any(v>=.1*(1-1e-6) for v in dopant_occ.values()):reasons.append("dopant_dilute_ceiling")
        if max(occ.max(),parent_occ.max())>1:reasons.append("unphysical_site_population")
        if any(r["clipped"] for r in hot_meta["donors"]):reasons.append("donor_population_exponential_clip")
        if abs(T-900)<1e-6:reasons.append("hot_thermochemistry_kink")
        if self.hall=="heuristic" and min(abs(logn-17),abs(logn-19.7))<1e-8:reasons.append("Hall_interpolation_kink")
        if not np.isfinite(jac).all():reasons.append("nonfinite_gradient")
        QE=float(ce[1:].sum()-ce[0]-beta*(totals@susceptibility))
        if QE>=0:reasons.append("nondecreasing_cold_charge")
        diagnostics=dict(root_residual_log=residual,jacobian_condition=condition,gradient_valid=not reasons,
            invalid_reasons=reasons,free_n_cm3=float(n),free_p_cm3=float(car[1]),hall_factor=rh,hall_convention=self.hall,
            single_electron_Hall_valid=bool(n>car[1]),above_carrier_detection_floor=bool(n/rh>=1e13),
            Hall_model_assumption="one itinerant electron channel;rH convention explicit;impurity-band transport not included",
            n_gt_p_does_not_establish_single_band=True,
            max_site_occupancy=float(max(occ.max(),parent_occ.max())),dopant_site_occupancy=dopant_occ,
            achieved_concentrations_cm3={e:float(self.atoms[:,j]@Nh) for j,e in enumerate(self.elements)},
            cold_achieved_concentrations_cm3={e:float(self.atoms[:,j]@N) for j,e in enumerate(self.elements)},
            ionized_absolute_charge_cm3=budgets[0],charge_squared_scatter_budget_cm3=budgets[1],
            charge_relative_residual=float(abs(P-M)/max(P,M)),cold_Q_EF_cm3_per_eV=QE,
            parent_probability_error=float(abs(np.bincount(self.parent,weights=pi,minlength=self.db.nD)-1).max()),
            Tf_K=float(T),measurement_T_K=float(To),Nd_cm3=float(Nd),pO2_atm=float(10**p[self.ipress]),
            profile=self.profile.to_dict(),profile_sha256=self.profile.fingerprint,material_deltas=p[self.imat].tolist(),
            active_charge_states=self.db.nCS,active_parents=self.db.nD,excluded_absent_elements=self.exclude_absent,
            localized_fraction=ell,localized_logit=float(p[self.iloc]),localized_slope=float(p[self.iloc+1]),
            fixed_fraction=self.fixed_fraction,donor_subgroups=donor,
            interpretation="fixed-input constrained subgroups,not microscopic spectrum,Motttransition or newDFTlabels")
        self.last_state=dict(hot=hot,hot_charge=Nh,cold_charge=N,totals=totals,pi=pi,ef_cold=ef,
            parent=self.parent,original_parent_ids=self.original_parent_ids,hot_mixture=hot_meta,
            cold_susceptibility=susceptibility,donor_subgroups=donor)
        if require_regular and reasons:raise ImplicitDomainError(";".join(reasons))
        return ImplicitPrediction(values,jac,self.output_names,self.parameter_names,np.r_[hot,ef],diagnostics)
