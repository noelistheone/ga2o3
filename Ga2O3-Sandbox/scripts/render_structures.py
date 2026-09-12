"""Publication-quality 3D atomic-structure figures of the sandbox's OWN computed structures.

Fully-manual ball-and-stick renderer (single coordinate frame: rotation + painter's-algorithm depth
sort so bonds/atoms/highlight/cell-box are all consistent) → 300-dpi PNGs for a paper. All structures
are genuinely computed by the sandbox: pristine β-Ga2O3, neutral O-vacancy, the new-dopant Sb/Bi
octahedral substitutionals, and a crystalline-vs-amorphous pair (the disorder-dEg mechanism).
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import ase.io
from ase.data import covalent_radii, atomic_numbers, chemical_symbols
from ase.data.colors import jmol_colors

PROJ = Path(__file__).resolve().parents[1]
CC = PROJ / "dft/qe_cc"
FIG = PROJ / "figures"; FIG.mkdir(exist_ok=True)
ROT = [(-80, "x"), (12, "y"), (0, "z")]     # view angles


def rotate(atoms):
    a = atoms.copy()
    for ang, ax in ROT:
        a.rotate(ang, ax, rotate_cell=True)
    return a


def cell_edges(cell, origin):
    """12 edges of the parallelepiped as pairs of 3D corner points."""
    import itertools
    corners = [origin + i * cell[0] + j * cell[1] + k * cell[2]
               for i, j, k in itertools.product([0, 1], repeat=3)]
    corners = np.array(corners)
    edges = []
    for a in range(8):
        for b in range(a + 1, 8):
            # connected if they differ in exactly one bit
            da = [(a >> t) & 1 for t in range(3)]
            db = [(b >> t) & 1 for t in range(3)]
            if sum(x != y for x, y in zip(da, db)) == 1:
                edges.append((corners[a], corners[b]))
    return edges


def render(ax, atoms, title, highlight_Z=None, radii_scale=0.52, bond_cut=2.5):
    r = rotate(atoms)
    P = r.get_positions()
    Z = atoms.get_atomic_numbers()
    sym = atoms.get_chemical_symbols()
    ctr = P.mean(axis=0)
    P = P - ctr
    cell = np.array(r.get_cell()); org = -ctr
    # --- cell box (behind everything) ---
    for p, q in cell_edges(cell, org):
        ax.plot([p[0], q[0]], [p[1], q[1]], "--", color="0.35", lw=0.9, zorder=0)
    # --- bonds (Ga-O, direct distance < cut; skip PBC wrap) : depth-sorted ---
    bonds = []
    for a in range(len(P)):
        for b in range(a + 1, len(P)):
            if {sym[a], sym[b]} == {"Ga", "O"}:
                d = np.linalg.norm(P[a] - P[b])
                if d < bond_cut:
                    bonds.append((0.5 * (P[a, 2] + P[b, 2]), a, b))
    for zmid, a, b in sorted(bonds):
        ax.plot([P[a, 0], P[b, 0]], [P[a, 1], P[b, 1]], "-", color="0.5", lw=1.4,
                zorder=1 + 0.001 * zmid, alpha=0.85, solid_capstyle="round")
    # --- atoms: painter's algorithm (back -> front) ---
    order = np.argsort(P[:, 2])
    zspan = (P[:, 2].min(), P[:, 2].max())
    for idx in order:
        z = Z[idx]
        rad = covalent_radii[z] * radii_scale
        # slight depth shading
        shade = 0.55 + 0.45 * (P[idx, 2] - zspan[0]) / (zspan[1] - zspan[0] + 1e-9)
        col = np.array(jmol_colors[z]) * shade + (1 - shade) * 0.15
        big = (highlight_Z is not None and z == highlight_Z)
        ax.add_patch(Circle((P[idx, 0], P[idx, 1]), rad * (1.7 if big else 1.0),
                            facecolor=np.clip(col, 0, 1),
                            edgecolor="k", linewidth=2.0 if big else 0.7,
                            zorder=5 + P[idx, 2]))
    m = np.abs(P[:, :2]).max() + 1.5
    ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title(title, fontsize=11)


def legend(fig, species, y=0.0):
    h = [Line2D([0], [0], marker="o", color="w", label=s,
                markerfacecolor=jmol_colors[atomic_numbers[s]], markeredgecolor="k", markersize=12)
         for s in species]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(species),
               frameon=False, fontsize=11)


def main():
    cryst = ase.io.read(str(CC / "gate2_perfect.in"), format="espresso-in")
    vo = ase.io.read(str(CC / "vo_q0_mace_relaxed.xyz"))
    sb = ase.io.read(str(CC / "dop_Sb_Ga_II_oct_mace.xyz"))
    bi = ase.io.read(str(CC / "dop_Bi_Ga_II_oct_mace.xyz"))

    import json as _json
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(9.5, 12.3))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.62], hspace=0.25, wspace=0.08)
    axs = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)]
    render(axs[0][0], cryst, "(a) β-Ga$_2$O$_3$ pristine (80-atom supercell)")
    render(axs[0][1], vo, "(b) neutral O-vacancy V$_O$ (MACE-MPA-0 relaxed)")
    render(axs[1][0], sb, "(c) Sb$_{Ga}$ octahedral (new dopant)", highlight_Z=51)
    render(axs[1][1], bi, "(d) Bi$_{Ga}$ octahedral (new dopant)", highlight_Z=83)
    legend(fig, ["Ga", "O", "Sb", "Bi"], y=0.295)
    # (e) crystal-chemistry validation strip: computed M-O bonds vs Shannon-radius sum
    axp = fig.add_subplot(gs[2, :])
    sp = _json.load(open(PROJ / "paper/npj/figures/shannon_parity.json"))
    HOST = (1.838, 2.079)
    axp.axhspan(HOST[0], HOST[1], color="0.85", alpha=0.6, zorder=0)
    axp.text(2.32, 1.86, "host Ga–O cage\n(1.84–2.08 Å)", fontsize=9.5, color="0.35")
    for el, p in sp["points"].items():
        if el == "W":
            continue
        small = p["shannon_sum_A"] < HOST[0]
        axp.scatter(p["shannon_sum_A"], p["computed_A"], s=42,
                    color="#D55E00" if small else "#0072B2", zorder=3)
        axp.annotate(el, (p["shannon_sum_A"], p["computed_A"]),
                     xytext=(3, 4), textcoords="offset points", fontsize=9)
    lo, hi = 1.42, 2.5
    axp.plot([lo, hi], [lo, hi], ls=":", color="0.4", lw=1)
    axp.set_xlim(lo, hi); axp.set_ylim(1.42, 2.3)
    axp.set_xlabel("Shannon ionic-radius sum  $r_M + r_O$  (Å)", fontsize=9)
    axp.set_ylabel("computed M–O bond (Å)", fontsize=9)
    axp.set_title("(e) geometry library vs crystal chemistry, 13 dopants with tabulated radii "
                  f"(Spearman ρ={sp['spearman_excl_W']['rho']}, p={sp['spearman_excl_W']['p']:.3f})", fontsize=10)
    # no in-figure title (caption carries it)
    fig.subplots_adjust(bottom=0.075, top=0.97)
    fig.savefig(FIG / "fig_structures_newdopant.png", dpi=300)
    fig.savefig(FIG / "fig_structures_newdopant.pdf")          # vector for the paper
    plt.close(fig)
    print("wrote figures/fig_structures_newdopant.{png,pdf}")

    amorph = ase.io.read(str(PROJ / "dft/disorder_mattersim/amorph_cell_0.xyz"))
    fig2, axs2 = plt.subplots(1, 2, figsize=(11, 5.6))
    render(axs2[0], cryst, "(a) crystalline β-Ga$_2$O$_3$   (PBE gap 2.34 eV)")
    render(axs2[1], amorph, "(b) amorphous a-Ga$_2$O$_3$ (MatterSim, 5.95 g/cc)\n3-cell mean gap 1.76 eV  →  −0.58 eV disorder narrowing")
    legend(fig2, ["Ga", "O"])
    fig2.suptitle("",
                  fontsize=12.5, y=0.98)
    fig2.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig2.savefig(FIG / "fig_crystal_vs_amorphous.png", dpi=300)
    fig2.savefig(FIG / "fig_crystal_vs_amorphous.pdf")         # vector for the paper
    plt.close(fig2)
    print("wrote figures/fig_crystal_vs_amorphous.{png,pdf}")


if __name__ == "__main__":
    main()
