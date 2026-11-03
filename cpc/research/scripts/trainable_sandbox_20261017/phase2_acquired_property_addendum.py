"""Real-data controls prompted by the completed virtual acquisition ablation.

All new controls and their exploratory timing are explicit. Prior protocols,
predictions and source code remain frozen and unchanged.
"""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from sandbox.energy_learning import reconstruct
import phase2_energy as energy
import phase2_acquired_property as runner

OUT=energy.OUT/"acquired_property_addendum"
SPECS={}
for model,short,cov in (
        ("ridge","Ridge","diagonal_xi"),
        ("extra_trees","ET","diagonal_A"),
        ("extra_trees","ET","diagonal_xi"),
        ("rbf","RBF","diagonal_xi")):
    for k in (2,4):
        SPECS[f"{short}_{cov}_k{k}"]=dict(model=model,covariance=cov,budget=k)

def freeze():
    energy.frozen()
    path=OUT/"protocol.json"
    payload=dict(version=1,created_utc=energy.now(),after_virtual_ablation=True,
        validation="retrospective additional controls, not blind",
        code_hashes={str(p.relative_to(ROOT)):energy.sha(p) for p in
            (Path(__file__),Path(runner.__file__),Path(runner.property_fit.__file__))},
        source_protocol_sha256=energy.sha(energy.OUT/"acquisition_ablation/protocol.json"),
        specs=SPECS,
        purpose="carry the stronger simple Ridge and independent covariance controls into the real-source carrier endpoint",
        workers=4,per_fit_budget="same two starts, bounds, max_nfev60 and analytic derivatives",
        data="same29/5 sources, same 0/1/2 anchors; no query carrier labels used for energy acquisition",
        covariance="diagonal_xi or diagonal_A as declared in the previous ablation",
        baseline="copy unchanged oracle_full/global_mean/extra_trees zero-acquisition references")
    if path.exists():
        assert json.loads(path.read_text())["code_hashes"]==payload["code_hashes"]
    else:energy.write(path,payload)

def database(element,method):
    spec=SPECS[method]
    db,rr,_=energy.records();targets=[r for r in rr if r["element"]==element]
    file=energy.OUT/"acquisition_ablation/states"/f'{element}__{spec["model"]}__{spec["covariance"]}.npz'
    states=np.load(file)
    return reconstruct(db,targets,states["means"][spec["budget"]])

if __name__=="__main__":
    runner.OUT=OUT;runner.SPECS=SPECS
    runner.freeze=freeze;runner.acquired_database=database
    runner.run(4)
