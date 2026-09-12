"""Crystal-chemistry validation of the 18-dopant MLIP structure library:
computed mean dopant-O bond length vs Shannon ionic-radius sum r_M(IV)+r_O(1.38 A).
Writes paper/npj/figures/shannon_parity.json."""
import json
from pathlib import Path
from pymatgen.core import Species
P=Path(__file__).resolve().parents[1]
db=json.load(open(P/"dft/mace_relax/structure_db.json"))["dopants"]
OX={"Al":3,"B":3,"Bi":3,"Cu":2,"Ca":2,"Cr":3,"Fe":3,"Ge":4,"Hf":4,"In":3,"Mg":2,
    "Nb":5,"Sb":5,"Si":4,"Sn":4,"Ta":5,"Ti":4,"Zn":2,"Zr":4,"W":6,"Mn":2,"Ni":2,"Co":2,"Ir":4,"Pt":4,"Rh":3,"H":1}
R_O=1.38  # Shannon r(O2-, CN IV)
out={"r_O":R_O,"points":{}}
for el,v in db.items():
    if el not in OX: continue
    sp=Species(el,OX[el])
    r=None; cn_used=None
    for cn in ("IV","VI"):
        try:
            r=sp.get_shannon_radius(cn); cn_used=cn; break
        except Exception: continue
    if r is None:
        try: r=float(sp.ionic_radius); cn_used="default"
        except Exception: continue
    out["points"][el]={"computed_A":v["mean_dopant_O_bond_A"],"shannon_sum_A":round(float(r)+R_O,3),
                       "ox":OX[el],"cn":cn_used}
import numpy as np
xs=np.array([p["shannon_sum_A"] for p in out["points"].values()])
ys=np.array([p["computed_A"] for p in out["points"].values()])
A=np.vstack([xs,np.ones_like(xs)]).T
m,b=np.linalg.lstsq(A,ys,rcond=None)[0]
r2=1-((ys-(m*xs+b))**2).sum()/((ys-ys.mean())**2).sum()
out["fit"]={"slope":round(float(m),3),"intercept_A":round(float(b),3),"R2":round(float(r2),3),
            "rms_A":round(float(np.sqrt(((ys-(m*xs+b))**2).mean())),3),"n":len(xs)}
(P/"paper/npj/figures/shannon_parity.json").write_text(json.dumps(out,indent=1))
print(json.dumps(out["fit"],indent=1)); print("n dopants:",len(out["points"]))
