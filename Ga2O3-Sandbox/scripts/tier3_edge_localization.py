"""R6 reviewer #2 item 9b: are the amorphous band-edge states extended or tail-localized?

For the crystal reference and amorphous cells (ms_0 + low-density lr54/lr50), extract the
VBM (band 352) and CBM (band 353) |psi|^2 with pp.x (plot_num=7) at the k-point hosting the
band extremum, and compute the participation ratio
    PR = (sum rho)^2 / (N * sum rho^2)   in [0, 1]  (1 = fully extended, <<1 = localized).
The amorphous-vs-crystal PR ratio classifies the states that define the occupation-based gap.
Writes results/tier3/edge_localization.json."""
import json, re, subprocess, os
from pathlib import Path
import numpy as np
import importlib.util

PROJ = Path(__file__).resolve().parents[1]
CC = PROJ / "dft/qe_cc"
spec = importlib.util.spec_from_file_location("edge", PROJ / "scripts/tier3_disorder_edge.py")
edge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(edge)
NOCC = edge.NOCC
PP = "/home/lawrence/qe-gpu-build/q-e-qe-7.5/bin/pp.x"

CELLS = ["crystal", "ms_0", "lr50"]


def kpt_of_extremum(name):
    ks = edge.per_k_eigs(name)
    vk = int(np.argmax([k[NOCC - 1] for k in ks])) + 1     # pp.x kpoint is 1-based
    ck = int(np.argmin([k[NOCC] for k in ks])) + 1
    return vk, ck


def run_pp(name, band, kpt, tag):
    fplot = CC / f"psi2_{name}_{tag}.cube"
    inp = CC / f"pp_{name}_{tag}.in"
    inp.write_text(f"""&INPUTPP
    prefix = 'dos_{name}_wf'
    outdir = '{CC}/outdir/dos_{name}_wf'
    plot_num = 7
    kpoint(1) = {kpt}
    kband(1) = {band}
/
&PLOT
    iflag = 3
    output_format = 6
    fileout = '{fplot}'
/
""")
    env = dict(os.environ)
    r = subprocess.run(f"source /home/lawrence/qe-gpu-build/setup_env.sh && "
                       f"mpirun -np 1 {PP} -in {inp} > {CC}/pp_{name}_{tag}.log 2>&1",
                       shell=True, executable="/bin/bash")
    return fplot if fplot.exists() else None


def participation(cube):
    lines = open(cube).read().splitlines()
    natoms = int(lines[2].split()[0])
    nx, ny, nz = (int(lines[i].split()[0]) for i in (3, 4, 5))
    data = np.fromstring(" ".join(lines[6 + natoms:]), sep=" ")
    rho = np.abs(data[: nx * ny * nz])
    return float((rho.sum() ** 2) / (rho.size * (rho ** 2).sum()))


out = {}
for name in CELLS:
    try:
        vk, ck = kpt_of_extremum(name)
    except Exception as e:
        out[name] = {"status": f"eigs unavailable: {e}"}
        continue
    entry = {}
    for tag, band, kpt in (("VBM", NOCC, vk), ("CBM", NOCC + 1, ck)):
        cube = run_pp(name, band, kpt, tag)
        entry[tag] = round(participation(cube), 4) if cube else None
    out[name] = entry
    print(name, entry, flush=True)

if "crystal" in out and isinstance(out["crystal"], dict) and out["crystal"].get("VBM"):
    for name in CELLS[1:]:
        if isinstance(out.get(name), dict) and out[name].get("VBM"):
            out[f"{name}_over_crystal"] = {
                t: round(out[name][t] / out["crystal"][t], 3) for t in ("VBM", "CBM")}

(PROJ / "results/tier3/edge_localization.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
