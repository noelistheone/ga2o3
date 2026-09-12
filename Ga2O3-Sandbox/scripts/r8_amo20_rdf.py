"""R8 AMO-20: structural characterization of the amorphous ensembles for the Methods protocol:
Ga-O partial RDF first-shell position and mean Ga coordination (2.4 A cutoff) for the three
MatterSim cells + the low-density cells + the crystal reference. Writes
results/tier3/r8_amo20_rdf.json."""
import json, glob
from pathlib import Path
import numpy as np
from ase.io import read
from ase.geometry import get_distances

PROJ = Path(__file__).resolve().parents[1]
out = {"quench_protocol": {
    "cell": "80 atoms, fixed cell", "engine": "MatterSim (and MACE-MP-0 for the 20-cell arm)",
    "melt_K": 3000.0, "melt_steps": 3000, "quench_ladder_K": [2200, 1400, 700, 300],
    "quench_steps_total": 3000, "timestep_fs": 2.0,
    "thermostat": "Langevin, friction 0.02/fs", "seeds": "independent per cell"}}


def rdf_stats(path):
    at = read(path)
    ga = [i for i, s in enumerate(at.get_chemical_symbols()) if s == "Ga"]
    ox = [i for i, s in enumerate(at.get_chemical_symbols()) if s == "O"]
    D = at.get_all_distances(mic=True)
    d_gao = D[np.ix_(ga, ox)].ravel()
    d_gao = d_gao[(d_gao > 0.5) & (d_gao < 6.0)]
    hist, edges = np.histogram(d_gao, bins=110, range=(1.0, 6.0))
    centers = 0.5 * (edges[1:] + edges[:-1])
    first_peak = float(centers[np.argmax(hist[centers < 2.6])])
    coord = float(np.mean([(D[g, ox] < 2.4).sum() for g in ga]))
    return {"first_shell_GaO_A": round(first_peak, 3), "mean_Ga_coordination_2.4A": round(coord, 2)}


cells = {}
for f in sorted(glob.glob(str(PROJ / "dft/disorder_mattersim/amorph_*.xyz"))):
    cells[Path(f).stem] = rdf_stats(f)
    print(Path(f).stem, cells[Path(f).stem], flush=True)
# crystal reference from the ordered supercell
try:
    ref = rdf_stats(str(PROJ.parent / "Ga2O3-Net/data/structures/ordered/undoped.cif"))
    cells["crystal_reference"] = ref
    print("crystal", ref, flush=True)
except Exception as e:
    print("crystal ref skipped:", e)
out["cells"] = cells
(PROJ / "results/tier3/r8_amo20_rdf.json").write_text(json.dumps(out, indent=1))
print("wrote results/tier3/r8_amo20_rdf.json")
