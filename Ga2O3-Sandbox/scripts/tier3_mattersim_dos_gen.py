import sys; sys.path.insert(0, "scripts")
from pathlib import Path
import ase.io
import importlib.util
spec = importlib.util.spec_from_file_location("dg", "scripts/tier3_disorder_dos_gen.py")
dg = importlib.util.module_from_spec(spec); spec.loader.exec_module(dg)
MS = Path("dft/disorder_mattersim")
n = 0
for xyz in sorted(MS.glob("amorph_cell_*.xyz")):
    name = "ms_" + xyz.stem.split("_")[-1]   # ms_0, ms_1, ms_2
    dg.write_dos(ase.io.read(str(xyz)), name)
    n += 1
print(f"wrote {n} MatterSim DOS inputs: dft/qe_cc/dos_ms_*.in")
