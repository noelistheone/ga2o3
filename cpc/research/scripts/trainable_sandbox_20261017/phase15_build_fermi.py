"""Generate a mathematical Fermi integral reference, without material labels."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from sandbox.coherent_fermi import TABLE,low_tail
import phase2_energy as energy


def reference(eta):
    amplitude=float(expit(eta));upper=np.sqrt(max(80.,eta+60.))
    def pair(power):
        def function(t):
            f=float(expit(eta-t*t));return 2*t*t*f*(1-f if power else 1)/amplitude
        return quad(function,0,upper,epsabs=3e-13,epsrel=3e-13,limit=180,points=[np.sqrt(eta)] if eta>0 else None)[0]
    F=pair(0);g=pair(1)/F
    return np.log(amplitude*2/np.sqrt(np.pi)*F),g


def main():
    if TABLE.exists():print('Existing mathematical reference',TABLE,energy.sha(TABLE));return
    start=time.perf_counter();eta=np.linspace(-20,60,12801);values=np.array([reference(x) for x in eta])
    low=low_tail(-20.);values[0]=low[:2]
    TABLE.parent.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(TABLE,eta=eta,logF=values[:,0],slope=values[:,1])
    out=energy.BASE/'phase15_coherent_hall'
    energy.write(out/'fd_reference_build.json',dict(created_utc=energy.now(),code_sha256=energy.sha(Path(__file__)),path=str(TABLE.relative_to(ROOT)),
        sha256=energy.sha(TABLE),nodes=len(eta),seconds=time.perf_counter()-start,labels=0,
        convention='normalizedFhalf;independentadaptive x=t² quadrature;logC2 spline/endpoints;tails gated separately'))
    print('Built',len(eta),'nodes',round(time.perf_counter()-start,3),'seconds',energy.sha(TABLE))


if __name__=='__main__':main()
