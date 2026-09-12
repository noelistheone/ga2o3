"""Small square atomic-structure insets for the Fig 1 workflow (module III & VI thumbnails).
Transparent background, no titles/cell-box clutter — just the ball-and-stick motif."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib.util

PROJ = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rs", PROJ / "scripts/render_structures.py")
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
import ase.io

OUT = PROJ / "paper/npj/src"

def inset(atoms, fname, highlight_Z=None):
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    rs.render(ax, atoms, "", highlight_Z=highlight_Z, radii_scale=0.55)
    ax.set_title("")
    fig.savefig(OUT / fname, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", fname)

sb = ase.io.read(str(PROJ / "dft/qe_cc/dop_Sb_Ga_II_oct_mace.xyz"))
amorph = ase.io.read(str(PROJ / "dft/disorder_mattersim/amorph_cell_0.xyz"))
inset(sb, "inset_sb.png", highlight_Z=51)
inset(amorph, "inset_amorph.png")
