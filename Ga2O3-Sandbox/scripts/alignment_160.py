"""160-atom planar-average potential alignment for V_O q2 (the convergence upgrade of the 80-atom
attempt that had no plateau). Writes results/tier3/alignment_160.json."""
import json
from pathlib import Path
import numpy as np
import importlib.util
PROJ=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("al", PROJ/"scripts/charged_defect_alignment.py")
al=importlib.util.module_from_spec(spec)
import sys; sys.modules['al']=al; spec.loader.exec_module.__self__ if False else spec.loader.exec_module(al)
PP=PROJ/"dft/qe_cc/ppot"
per=al.parse_cube(PP/"fs160_perfect_vtot.cube")
q2 =al.parse_cube(PP/"fs160_VO_q2_vtot.cube")
res=al.alignment(per,q2,None)
# apply with the same sign convention used at 80 atoms (+1, fixed by the V_O->database direction)
raw=2.223
out={"cell":"160-atom (2x1x1)","C_align_eV":res["C_align_eV"],"plateau_spread_eV":res["plateau_spread_eV"],
     "dV_range_eV":res["dV_range_eV"],"profile":res["dV_profile_sample"],
     "eps_raw_80":raw,"eps_aligned_160":round(raw+res["C_align_eV"],3),
     "note":"alignment constant from the 160-atom pair applied to the 80-atom raw level (monopole "
            "part shown converged to 0.019 eV by the D+MP scaling check)."}
(PROJ/"results/tier3/alignment_160.json").write_text(json.dumps(out,indent=1))
print(json.dumps(out,indent=1))
