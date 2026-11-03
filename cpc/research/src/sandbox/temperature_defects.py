"""Explicit-profile, variable-temperature, implicitly differentiable defect layer.

Uses the complete supplied energetic catalogue, optionally removes exactly absent
elements, and can add effective neutral donor states consistently in BOTH hot
equilibrium and the frozen-parent cold partition. These are not new HSE labels.
Outputs are Hall density, absolute charge budget, q-squared budget. Old RT CT
mobility is available only through an explicit300K compatibility head.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from scipy.optimize import root,brentq
from . import thermo,transport
from .implicit_defects import (ImplicitDefectLayer,ImplicitPrediction,ImplicitDomainError,
                              _charge_log,_thermo_derivatives,LN10)
from .material_profile import MaterialProfile,REPOSITORY_PROFILE

class TemperatureDefectLayer(ImplicitDefectLayer):
    def __init__(self,db,elements=("Si",),*,profile=REPOSITORY_PROFILE,basis=None,
                 hall="heuristic",donor_extension=False,donor_element=None,exclude_absent=True):
        if not isinstance(profile,MaterialProfile):raise TypeError("Explicit immutable MaterialProfile required")
        elements=tuple(elements);original=db
        B=np.zeros((db.nCS,0)) if basis is None else np.asarray(basis,float)
        if B.ndim!=2 or B.shape[0]!=db.nCS:raise ValueError("Energy basis shape")
        other=[j for j in range(2,len(db.elements)) if db.elements[j] not in elements]
        keep=(np.flatnonzero(np.all(db.dm[:,other]==0,axis=1)) if exclude_absent
              else np.arange(db.nCS))
        parents=np.unique(db.cs_ID[keep]);pmap={int(p):i+1 for i,p in enumerate(parents)}
        db=replace(db,dHo=db.dHo[keep].copy(),charge=db.charge[keep].copy(),dm=db.dm[keep].copy(),
            cs_ID=np.array([pmap[int(p)] for p in db.cs_ID[keep]]),
            prefactor=db.prefactor[keep].copy(),sum_dm=db.sum_dm[keep].copy(),nCS=len(keep),nD=len(parents))
        B=B[keep].copy();original_ids=original.cs_ID[keep].copy();nbase=db.nCS
        self.neutral=np.zeros(nbase,bool);self.donor_original_ids=[]
        self.donor_element=donor_element or elements[0]
        if donor_extension:
            if self.donor_element not in ("Si","Sn") or self.donor_element not in elements:
                raise ValueError("Audited donor extension supports incorporated Si or Sn")
            wanted=(169,170) if self.donor_element=="Si" else (151,152)
            refs=[]
            for parent in wanted:
                ix=np.flatnonzero(original_ids==parent)
                if len(ix)!=1 or db.charge[ix[0]]!=1:raise ValueError("Audited isolated donor catalogue changed")
                dm=db.dm[ix[0]];expected=np.zeros(len(db.elements),int)
                expected[0]=-1;expected[db.col(self.donor_element)]=1
                if not np.array_equal(dm,expected):raise ValueError("Donor composition changed")
                refs.append(int(ix[0]))
            refs=np.asarray(refs)
            db=replace(db,dHo=np.r_[db.dHo,db.dHo[refs]],charge=np.r_[db.charge,np.zeros(len(refs),int)],
                dm=np.r_[db.dm,db.dm[refs]],cs_ID=np.r_[db.cs_ID,db.cs_ID[refs]],
                prefactor=np.r_[db.prefactor,2*db.prefactor[refs]],sum_dm=np.r_[db.sum_dm,db.sum_dm[refs]],
                nCS=nbase+len(refs))
            B=np.r_[B,B[refs]];original_ids=np.r_[original_ids,original_ids[refs]]
            self.neutral=np.r_[self.neutral,np.ones(len(refs),bool)]
            self.donor_original_ids=list(wanted)
        super().__init__(db,elements,B,hall)
        self._profile=profile;self.donor_extension=bool(donor_extension)
        self.exclude_absent=bool(exclude_absent);self.original_parent_ids=original_ids
        self.base_np=self.np;self.iobs=self.base_np
        self.imat=slice(self.base_np+1,self.base_np+5)
        self.ie=self.base_np+5;self.ipress=2+self.b+self.k
        self.np+=5+(2 if donor_extension else 0)
        self.parameter_names+=(("log10_measurement_T_K","delta_log10_NcRef","delta_log10_NvRef",
            "delta_Eg0_eV","delta_Ec_fraction")+
            (("log10_donor_binding_E0_eV","donor_screen_s") if donor_extension else ()))
        self.output_names=("log10_nH_cm3","log10_Nabs_cm3","log10_Nq2_cm3")
        self.idose=2+self.b+(elements.index(self.donor_element) if donor_extension else 0)
        self.first=np.array([np.flatnonzero(self.parent==i)[0] for i in range(db.nD)])
        for i,j in enumerate(self.first):
            if not np.all(db.dm[self.parent==i]==db.dm[j]):raise ValueError("Parent atom compositions differ")
        # Consistent hot/cold conditional weights, including new neutral spin2.
        self.logg=np.log(db.prefactor/db.prefactor[self.first[self.parent]])
        for x in (db.dHo,db.charge,db.dm,db.cs_ID,db.prefactor,db.sum_dm,self.basis):
            x.setflags(write=False)
        self._hot_cache=None;self.hot_cache_hits=0;self.hot_solve_calls=0

    @property
    def profile(self):return self._profile

    def parameters(self,T=1325.,Nd=9.1e15,weights=None,doses=(1e18,),pO2=1e-5,
                   Tobs=300.,material_deltas=(0.,0.,0.,0.),binding_eV=.0366,screen_s=0.):
        w=np.zeros(self.b) if weights is None else np.asarray(weights,float)
        p=np.r_[np.log10(T),np.log10(Nd),w,np.log10(doses),np.log10(pO2),
                 np.log10(Tobs),np.asarray(material_deltas,float)]
        return np.r_[p,np.log10(binding_eV),screen_s] if self.donor_extension else p

    def _unpack(self,pars):
        p=np.asarray(pars,float)
        if p.shape!=(self.np,) or not np.isfinite(p).all():raise ValueError("Finite full parameter vector required")
        T,Nd=10**p[0],10**p[1];doses=10**p[2+self.b:2+self.b+self.k]
        To=10**p[self.iobs]
        if not (300<T<=2200 and 10<=To<=T*(1+1e-12) and Nd>0 and np.all(doses>0)):
            raise ImplicitDomainError("Outside hot/frozen observation domain")
        self.profile.shifted(p[self.imat])
        if self.donor_extension and not (-4<=p[self.ie]<=np.log10(.5) and 0<=p[self.ie+1]<=10):
            raise ImplicitDomainError("Outside declared effective binding relation domain")
        return p,T,Nd,doses

    def _reference(self,T,p):
        profile=self.profile.shifted(p[self.imat])
        H=self.db.dHo+self.basis@p[2:2+self.b]
        HP=np.zeros((self.db.nCS,self.np));HP[:,2:2+self.b]=self.basis
        HT=np.zeros(self.db.nCS)
        if self.donor_extension:
            _,Ec,_,dEc,_,mEc=profile.bands(T)
            z=(10**p[self.idose]/1e18)**(1/3)
            E=10**p[self.ie]*np.exp(-p[self.ie+1]*z)
            H=H.copy();H[self.neutral]+=Ec-E
            HT[self.neutral]=dEc;HP[self.neutral,self.imat]=mEc
            HP[self.neutral,self.ie]=-E*LN10
            HP[self.neutral,self.ie+1]=z*E
            HP[self.neutral,self.idose]+=p[self.ie+1]*E*z*LN10/3
        return H,HP,HT,profile

    def _entropy(self,T):
        s=float(thermo.dSvib_norm(T,self.profile.vib_T0))
        n=1/np.expm1(self.profile.vib_T0/T)
        ds=self.profile.vib_T0**2/T**3*n*(1+n)
        return s,ds

    def _hot(self,u,pars):
        self.root_evaluations+=1;p,T,Nd,doses=self._unpack(pars)
        H,HP,HT,profile=self._reference(T,p);beta=1/(thermo.KB_EV*T)
        mu=np.full(len(self.db.elements),-30.);mu[:2]=thermo.mu_Ga_O(T,10**p[self.ipress]);mu[self.cols]=u[1:]
        s,ds=self._entropy(T)
        A=H-3*thermo.KB_EV*T*s*self.db.sum_dm-self.db.dm@mu
        x=-(A+self.q*u[0])*beta;active=(x>-700)&(x<300)
        N=self.db.prefactor*np.exp(np.clip(x,-700,300))
        dU=N[:,None]*np.column_stack([-self.q*beta,self.atoms*beta])*active[:,None]
        dP=-N[:,None]*beta*HP
        dga,do,_=_thermo_derivatives(T,p[self.ipress])
        AT=HT-3*thermo.KB_EV*(s+T*ds)*self.db.sum_dm-self.db.dm[:,0]*dga-self.db.dm[:,1]*do
        dP[:,0]+=N*(-AT*beta-x/T)*T*LN10
        dP[:,self.ipress]+=N*beta*(self.db.dm[:,1]-1.5*self.db.dm[:,0])*.5/beta*LN10
        dP*=active[:,None]
        car,ce,ct,cm=profile.carriers(u[0],T)
        cU=np.zeros((4,self.nhot));cU[:,0]=ce
        cP=np.zeros((4,self.np));cP[:,0]=ct*T*LN10;cP[:,self.imat]=cm
        ndP=np.zeros(self.np);ndP[1]=Nd*LN10
        F0,J0,_,_=_charge_log(N,np.c_[dU,dP],car,np.c_[cU,cP],self.q,Nd,np.r_[np.zeros(self.nhot),ndP])
        C=self.atoms.T@N
        F=np.r_[F0,np.log(C)-np.log(doses)]
        J=np.vstack([J0,self.atoms.T@np.c_[dU,dP]/C[:,None]])
        for j in range(self.k):J[1+j,self.nhot+2+self.b+j]-=LN10
        return F,J[:,:self.nhot],J[:,self.nhot:],N,dU,dP

    def _hot_solution(self,p):
        key=(self.profile.fingerprint,tuple(np.delete(p,self.iobs)))
        if self._hot_cache is not None and self._hot_cache[0]==key:
            self.hot_cache_hits+=1;return self._hot_cache[1]
        self.hot_solve_calls+=1
        _,T,_,_=self._unpack(p);profile=self.profile.shifted(p[self.imat])
        Ec=profile.bands(T)[1];seed=np.r_[Ec-.15,np.full(self.k,-5.)]
        for _ in range(6):
            F,J,*_=self._hot(seed,p)
            for j in range(self.k):
                if abs(J[1+j,1+j])>1e-12:seed[1+j]-=np.clip(F[1+j]/J[1+j,1+j],-2,2)
        seeds=([self.last_hot.copy()] if self.last_hot is not None else [])+[seed]
        best=None
        for initial in seeds:
            sol=root(lambda u:self._hot(u,p)[0],initial,jac=lambda u:self._hot(u,p)[1],
                     method="hybr",options={"xtol":2e-11,"maxfev":160})
            res=float(abs(self._hot(sol.x,p)[0]).max())
            if best is None or res<best[0]:best=(res,sol.x)
            if res<2e-9:break
        if best[0]>2e-8:raise ImplicitDomainError("Hot root residual "+str(best[0]))
        if np.any(best[1][1:]<=-30) or np.any(best[1][1:]>=5):raise ImplicitDomainError("Hot chemical potential bounds")
        u=best[1];self.last_hot=u.copy();block=self._hot(u,p)
        self._hot_cache=(key,(u.copy(),block));return self._hot_cache[1]

    def predict(self,pars,*,require_regular=True):
        self.root_calls+=1;p,T,Nd,doses=self._unpack(pars);To=10**p[self.iobs]
        hot,(Fh,Jh,Ph,Nh,NhU,NhP)=self._hot_solution(p)
        totals=np.bincount(self.parent,weights=Nh,minlength=self.db.nD)
        tU=np.array([np.bincount(self.parent,weights=a,minlength=self.db.nD) for a in NhU.T]).T
        tP=np.array([np.bincount(self.parent,weights=a,minlength=self.db.nD) for a in NhP.T]).T
        H,HP,HT,profile=self._reference(To,p);beta=1/(thermo.KB_EV*To);s,ds=self._entropy(To)
        A=H-3*thermo.KB_EV*To*s*self.db.sum_dm
        AT=HT-3*thermo.KB_EV*(s+To*ds)*self.db.sum_dm
        def populations(ef):
            z=-(A+self.q*ef)*beta;logw=z+self.logg
            mx=np.full(self.db.nD,-np.inf);np.maximum.at(mx,self.parent,logw)
            w=np.exp(logw-mx[self.parent]);den=np.bincount(self.parent,weights=w,minlength=self.db.nD)
            pi=w/den[self.parent]
            return totals[self.parent]*pi,pi,z
        def cold(ef):
            N,_,_=populations(ef);c=profile.carriers(ef,To)[0]
            P=np.maximum(self.q,0)@N+c[1:].sum()+Nd
            M=np.maximum(-self.q,0)@N+c[0]
            # At verylowT, one sign can underflow at a remote bracket endpoint.
            return (P-M)/max(P,M,1.)
        ef=brentq(cold,-5.,12.,xtol=2e-14,rtol=1e-15)
        N,pi,z=populations(ef)
        mq=np.bincount(self.parent,weights=pi*self.q,minlength=self.db.nD)
        dNU=np.zeros((self.db.nCS,self.nu));dNU[:,:self.nhot]=pi[:,None]*tU[self.parent]
        dNU[:,-1]=-beta*N*(self.q-mq[self.parent])
        LP=-beta*HP;LP[:,self.iobs]+=(-beta*AT-z/To)*To*LN10
        mLP=np.array([np.bincount(self.parent,weights=pi*a,minlength=self.db.nD) for a in LP.T]).T
        dNP=pi[:,None]*tP[self.parent]+N[:,None]*(LP-mLP[self.parent])
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
        deriv=np.zeros((3,self.nu+self.np))
        deriv[0]=dc[0]/(n*LN10)*(1-drh/(rh*LN10))
        budgets=[];budget_grads=[]
        for charge in (abs(self.q),self.q**2):
            v=float(charge@N+Nd);dv=charge@dN;dv[self.nu+1]+=Nd*LN10
            budgets.append(v);budget_grads.append(dv)
        deriv[1]=budget_grads[0]/(budgets[0]*LN10);deriv[2]=budget_grads[1]/(budgets[1]*LN10)
        jac=deriv[:,self.nu:]+deriv[:,:self.nu]@root_deriv
        values=np.r_[logn-np.log10(rh),np.log10(budgets)]
        residual=float(max(abs(Fh).max(),abs(Fc)));occ=Nh/self.db.prefactor
        parent_pref=self.db.prefactor[self.first];parent_occ=totals/parent_pref
        dopant_occ={e:float(max(occ[self.atoms[:,j]>0].max(),
            parent_occ[self.db.dm[self.first,self.cols[j]]>0].max())) for j,e in enumerate(self.elements)}
        reasons=[]
        if residual>2e-8:reasons.append("root_residual")
        if not np.isfinite(condition) or condition>1e12:reasons.append("ill_conditioned_root")
        if any(v>=.1*(1-1e-6) for v in dopant_occ.values()):reasons.append("dopant_dilute_ceiling")
        if max(occ.max(),parent_occ.max())>1:reasons.append("unphysical_site_population")
        if abs(T-900)<1e-6:reasons.append("hot_thermochemistry_kink")
        if self.hall=="heuristic" and min(abs(logn-17),abs(logn-19.7))<1e-8:reasons.append("Hall_interpolation_kink")
        if not np.isfinite(jac).all():reasons.append("nonfinite_gradient")
        variance=np.bincount(self.parent,weights=pi*(self.q-mq[self.parent])**2,minlength=self.db.nD)
        QE=float(ce[1:].sum()-ce[0]-beta*(totals@variance))
        if QE>=0:reasons.append("nondecreasing_cold_charge")
        diagnostics=dict(root_residual_log=residual,jacobian_condition=condition,
            gradient_valid=not reasons,invalid_reasons=reasons,
            free_n_cm3=float(n),free_p_cm3=float(car[1]),hall_factor=rh,hall_convention=self.hall,
            single_electron_Hall_valid=bool(n>car[1]),above_carrier_detection_floor=bool(n/rh>=1e13),
            max_site_occupancy=float(max(occ.max(),parent_occ.max())),dopant_site_occupancy=dopant_occ,
            achieved_concentrations_cm3={e:float(self.atoms[:,j]@Nh) for j,e in enumerate(self.elements)},
            cold_achieved_concentrations_cm3={e:float(self.atoms[:,j]@N) for j,e in enumerate(self.elements)},
            ionized_absolute_charge_cm3=budgets[0],charge_squared_scatter_budget_cm3=budgets[1],
            charge_relative_residual=float(abs(P-M)/max(P,M)),cold_Q_EF_cm3_per_eV=QE,
            parent_probability_error=float(abs(np.bincount(self.parent,weights=pi,minlength=self.db.nD)-1).max()),
            Tf_K=float(T),measurement_T_K=float(To),Nd_cm3=float(Nd),pO2_atm=float(10**p[self.ipress]),
            profile=self.profile.to_dict(),profile_sha256=self.profile.fingerprint,
            material_deltas=p[self.imat].tolist(),donor_extension=self.donor_extension,
            active_charge_states=self.db.nCS,active_parents=self.db.nD,excluded_absent_elements=self.exclude_absent,
            donor_binding_interpretation="effective localized statistical states,not new HSE charge labels",
            kinetic_inventory_interpretation="specified constrained equilibrium then frozen atoms;no cooling-rate claim")
        self.last_state=dict(hot=hot,hot_charge=Nh,cold_charge=N,totals=totals,pi=pi,ef_cold=ef,
                             parent=self.parent,original_parent_ids=self.original_parent_ids)
        if require_regular and reasons:raise ImplicitDomainError(";".join(reasons))
        return ImplicitPrediction(values,jac,self.output_names,self.parameter_names,np.r_[hot,ef],diagnostics)

    def legacy_300_outputs(self,pred):
        if abs(pred.diagnostics["measurement_T_K"]-300)>1e-7:
            raise ValueError("RT empirical compatibility head is restricted to300K")
        NI=10**pred.values[1];mu=float(transport.mu_caughey_thomas(NI,300))
        zz=(NI/transport.CT_N_REF)**transport.CT_ALPHA
        dlogmu_dlogNI=-(transport.CT_MU_MAX-transport.CT_MU_MIN)*transport.CT_ALPHA*zz/((1+zz)**2*mu)
        jmu=dlogmu_dlogNI*pred.jacobian[1]
        vals=np.array([pred.values[0],np.log10(mu),np.log10(thermo.Q)+pred.values[0]+np.log10(mu)])
        return vals,np.vstack([pred.jacobian[0],jmu,pred.jacobian[0]+jmu])
