"""Catalogue/gauge/stock/IFT gates before fitting the new allocation relation."""
from pathlib import Path
import os,sys,json
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ.setdefault(k,"1")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from scipy.special import softmax
from sandbox.kroger_db import load,NPZ
from sandbox.parent_allocation import ParentAllocationLayer,isolated_parent_contrast
from sandbox.temperature_defects import TemperatureDefectLayer
from sandbox.material_profile import PUBLISHED_PROFILE
from sandbox.thermo import KB_EV
import phase2_energy as energy

OUT=energy.BASE/"phase11_parent_allocation"


def freeze():
    p=dict(created_utc=energy.now(),version=1,
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (Path(__file__),ROOT/"src/sandbox/parent_allocation.py",
            ROOT/"src/sandbox/temperature_defects.py",ROOT/"src/sandbox/material_profile.py")},
        input_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in (NPZ,energy.BASE/"artifact_manifest_phase10.json")},
        theory="dimensionless bias b:u withHshift=kBTf*b*u,free-element-potential projection only,not hostGa/O",
        stage="catalogue feasibility and numerical gates before new predictive learning;no material accuracy claim")
    path=OUT/"numerical_protocol.json"
    if path.exists():
        old=json.loads(path.read_text());assert old["code_hashes"]==p["code_hashes"] and old["input_hashes"]==p["input_hashes"]
    else:energy.write(path,p)


def main():
    freeze();db=load();checks=[];scans=[];gradients=[];catalogue=[]
    for elements,doses in ((("Si",),(1e18,)),(("Sn",),(1e19,)),(("Si","H"),(5.4e17,1.8e17))):
        layer=ParentAllocationLayer(db,elements);core=layer.core;c=layer.contrast
        checks.append(dict(name="nontrivial_gauge_free_basis",elements=elements,norm=float(np.linalg.norm(c.projected_parent)),
            orthogonality=c.orthogonality_error,passed=np.linalg.norm(c.projected_parent)>1e-8 and c.orthogonality_error<1e-10))
        for parent in c.parent_ids:
            ix=np.flatnonzero(db.cs_ID==parent);j=ix[0]
            if db.dm[j,db.col(elements[0])]>0:
                catalogue.append(dict(elements=elements,parent_id=int(parent),
                    composition={e:int(v) for e,v in zip(db.elements,db.dm[j]) if v},
                    charge_states=db.charge[ix],projected_weight=float(c.projected_parent[np.flatnonzero(c.parent_ids==parent)[0]])))
        args=dict(Tf_K=1500.,background_cm3=1e15,doses=doses,E0_eV=.06,s=.5)
        # Raw and projected bases differ only by a free-potential gauge.
        raw=TemperatureDefectLayer(db,elements,profile=PUBLISHED_PROFILE,basis=c.raw_charge_basis[:,None],donor_extension=True,hall="rH1")
        for b in (-3.,0.,3.):
            a,J,p=layer.point(**args,bias=b)
            q=raw.parameters(T=1500,Nd=1e15,doses=doses,binding_eV=.06,screen_s=.5,weights=[KB_EV*1500*b])
            r=raw.predict(q)
            err=float(abs(a.values-r.values).max())
            # new projected potential = raw potential - kBT*b*v.
            muerr=float(abs(core.last_state["hot"][1:]-(raw.last_state["hot"][1:]-KB_EV*1500*b*c.removed_potential_coordinates)).max())
            checks.append(dict(name="raw_projected_prediction_equivalence",elements=elements,bias=b,error=err,
                potential_error=muerr,passed=err<2e-8 and muerr<2e-8))
        # A pure composition direction is exactly canceled by potential solving.
        v=np.arange(1,len(elements)+1)*.3
        gauge=db.dm[:,[db.col(e) for e in elements]]@v
        g=TemperatureDefectLayer(db,elements,profile=PUBLISHED_PROFILE,basis=gauge[:,None],donor_extension=True,hall="rH1")
        q=g.parameters(T=1500,Nd=1e15,doses=doses,binding_eV=.06,screen_s=.5,weights=[0.])
        baseline=g.predict(q);u0=g.last_state["hot"].copy()
        q[2]=KB_EV*1500*2.;changed=g.predict(q)
        err=float(abs(changed.values-baseline.values).max())
        shift=float(abs(g.last_state["hot"][1:]-u0[1:]-KB_EV*1500*2*v).max())
        checks.append(dict(name="pure_gauge_no_prediction_signal",elements=elements,error=err,potential_error=shift,
            passed=err<2e-8 and shift<2e-8))
        # Common parent bias cancels in a fixed-EF conditional partition.
        a,J,p=layer.point(**args,bias=2.)
        beta=1/(KB_EV*300);p0=p.copy();p0[2]=0.
        H1=core._reference(300,p)[0];H0=core._reference(300,p0)[0]
        error=0.
        for parent in range(core.db.nD):
            ix=np.flatnonzero(core.parent==parent)
            pi1=softmax(-(H1[ix]+core.q[ix]*4.5)*beta+core.logg[ix])
            pi0=softmax(-(H0[ix]+core.q[ix]*4.5)*beta+core.logg[ix])
            error=max(error,float(abs(pi1-pi0).max()))
        checks.append(dict(name="fixed_inventory_conditional_invariance",elements=elements,error=error,passed=error<1e-11))
        # Explicit stock and atom preservation between hot and cold states.
        for To in (40.,300.,1500.):
            a,J,p=layer.point(**args,bias=2.,Tobs_K=To);s=core.last_state
            atomhot=core.db.dm.T@s["hot_charge"];atomcold=core.db.dm.T@s["cold_charge"]
            ae=float(abs(atomhot-atomcold).max()/max(abs(atomhot).max(),1))
            checks.append(dict(name="stock_charge_probability",elements=elements,Tobs=To,
                passed=ae<1e-11 and a.diagnostics["charge_relative_residual"]<1e-8 and a.diagnostics["parent_probability_error"]<1e-12))
            if To==1500:
                e=float(abs(s["ef_cold"]-s["hot"][0]))
                checks.append(dict(name="no_quench_identity",elements=elements,error=e,passed=e<1e-9))
        # All5 new coordinates, including chain fromTf to the energy weight.
        for To in (40.,300.):
            a,J,p=layer.point(**args,bias=2.,Tobs_K=To)
            x=np.array([np.log10(1500),15.,np.log10(.06),.5,2.])
            def call(v):
                return layer.point(Tf_K=10**v[0],background_cm3=10**v[1],doses=doses,Tobs_K=To,E0_eV=10**v[2],s=v[3],bias=v[4])[0].values
            for j,name in enumerate(layer.parameter_names):
                h=1e-5;plus=x.copy();minus=x.copy();plus[j]+=h;minus[j]-=h
                fd=(call(plus)-call(minus))/(2*h);err=float(np.max(abs(fd-J[:,j])/(1+abs(fd))))
                gradients.append(dict(elements=elements,Tobs=To,coordinate=name,scaled_error=err))
                checks.append(dict(name="allocation_full_IFT",elements=elements,Tobs=To,coordinate=name,error=err,passed=err<3e-5))
        for C in (1e17,1e19):
            for b in (-6.,-3.,0.,3.,6.,12.):
                dose=(C,doses[1]) if len(doses)>1 else (C,)
                try:
                    a,J,p=layer.point(**{**args,"doses":dose},bias=b)
                    state=core.last_state;Xp=core.db.dm[core.first,core.cols[0]]
                    ids=core.original_parent_ids[core.first];isol=np.isin(ids,c.isolated_parent_ids)
                    f=float((state["totals"]*Xp*isol).sum()/C)
                    scans.append(dict(elements=elements,C_cm3=C,bias=b,numeric=True,values=a.values,
                        dnH_db=float(J[0,4]),isolated_dopant_fraction=f,diagnostics=a.diagnostics))
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                    scans.append(dict(elements=elements,C_cm3=C,bias=b,numeric=False,error=repr(exc)))
    result=dict(passed=all(r["passed"] for r in checks),checks=checks,gradients=gradients,
        maximum_gradient_scaled_error=max(r["scaled_error"] for r in gradients),catalogue=catalogue,scans=scans,
        scope="catalogue/gauge/IFT/stock checks and hypothetical input sensitivity,not measured prediction accuracy")
    energy.write(OUT/"numerical_gates.json",result)
    print(json.dumps(dict(passed=result["passed"],checks=len(checks),
        gradient_error=result["maximum_gradient_scaled_error"],scan_failures=sum(not r["numeric"] for r in scans)),indent=2))
    for e in (("Si",),("Sn",),("Si","H")):
        print("SCAN",e,[(r["C_cm3"],r["bias"],round(r["values"][0],4),round(r["isolated_dopant_fraction"],5)) for r in scans if tuple(r["elements"])==e and r["numeric"]])
    assert result["passed"]


if __name__=="__main__":main()
