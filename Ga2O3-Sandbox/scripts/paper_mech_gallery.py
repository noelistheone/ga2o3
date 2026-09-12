"""Mechanism-gallery figure (intermediate products of the pipeline, visualized).
(a) configuration-coordinate diagrams for V_O vs V_Ga (parabolas reconstructed from the computed
    lambda/DeltaQ/DeltaE parameters; capture barriers annotated) — the PPC mechanism.
(b) defect-level diagram: our V_O (2+/0) vs the database band, and the Sb/Bi PBE->HSE arrows.
(c) the two absolute-n levers/cliffs: n(T_f) for Si (the 4-decade freeze-in knob) and n(pO2) for
    Ta (the exponential cliff), from solver runs.
(d) disorder-ensemble gap distribution: 20 MACE cells (bimodal: recrystallized vs amorphous) + 3
    density-corrected MatterSim cells — two independent engines, one mechanism.
All numbers from files of record. Vector PDF, Okabe-Ito.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path("/home/lawrence/Physics/Ga2O3-Sandbox")
OUT = P / "paper/npj/src"
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00",
      "sky": "#56B4E9", "purple": "#CC79A7", "grey": "#7f7f7f"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.8,
                     "lines.linewidth": 1.5, "savefig.bbox": "tight", "pdf.fonttype": 42})

def panel_label(ax, s):
    ax.text(-0.14, 1.06, s, transform=ax.transAxes, fontweight="bold", fontsize=11, va="top")

fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.9))
axs = axs.ravel()

# ---------- (a) configuration-coordinate diagrams ----------
ax = axs[0]
vo = json.load(open(P / "results/tier2/tau_VO_result.json"))
vga = json.load(open(P / "results/tier2/tau_VGa_result.json"))
for d, c, name, style in [(vo, OI["blue"], r"V$_\mathrm{O}$ (2+/0)", "-"),
                          (vga, OI["verm"], r"V$_\mathrm{Ga}$ (0/3$-$)", "--")]:
    dQ = d["dQ_amu_half_A"]; dE = d["driving_force_dE_eV"]
    lamA = d.get("reorg_energy_charged_lam2_eV", d.get("lambda_qA_eV"))
    lamB = d.get("reorg_energy_neutral_lam0_eV", d.get("lambda_qB_eV"))
    kA = 2 * lamA / dQ**2; kB = 2 * lamB / dQ**2
    Q = np.linspace(-0.35 * dQ, 1.45 * dQ, 300)
    ax.plot(Q / dQ, 0.5 * kA * Q**2, style, color=c, label=name)
    ax.plot(Q / dQ, dE + 0.5 * kB * (Q - dQ)**2, style, color=c, alpha=0.55)
    ytxt = 0.43 if "O}" in name else 0.355
    ax.annotate(f"$E_b$={d['capture_barrier_eV']:.2f} eV, $\\Delta Q$={dQ:.1f}",
                xy=(0.27, ytxt), xycoords="axes fraction", ha="left", fontsize=7, color=c)
ax.set_xlabel(r"configuration coordinate $Q/\Delta Q$")
ax.set_ylabel("energy (eV)")
ax.set_ylim(-0.3, 4.2); ax.legend(frameon=False, loc="upper left")
ax.text(0.60, 0.97, "sequential steps:\nV$_\\mathrm{O}$ (+/0) $\\lambda$=0.51 (rate-limiting)\nV$_\\mathrm{Ga}$ per-step $\\lambda$=0.20–0.39\n(weak-coupling boundary)", fontsize=6.6, color="k",
        transform=ax.transAxes, ha="center", va="top")
ax.set_title("carrier capture: PPC mechanism", fontsize=8)
panel_label(ax, "a")

# ---------- (b) defect-level diagram (best-estimate chain + budget bands) ----------
ax = axs[1]
Eg = 4.9
BUD = 0.25  # the paper's +-0.2-0.3 eV absolute-level budget, drawn at its midpoint
lv = json.load(open(P / "results/tier3/levels_v2.json"))
vo_best = lv["VO_best_estimate_vs_DB"]["eps_eV"]                      # 2.586
sb_best = lv["Sb_a033"]["aniso_exact"]["eps_2+/0"]                    # 3.651
bi_best = lv["Bi_a033"]["aniso_exact"]["eps_2+/0"]                    # 2.067
ax.axhspan(-0.4, 0, color=OI["grey"], alpha=0.35); ax.text(2.7, -0.24, "VBM", fontsize=7)
ax.axhspan(Eg, Eg + 0.4, color=OI["grey"], alpha=0.35); ax.text(2.38, Eg + 0.12, "CBM (exp)", fontsize=7)
ax.axhline(3.905, color="k", lw=0.8, ls="--")
ax.text(0.06, 3.96, "CBM (HSE06 $\\alpha$=0.25)", fontsize=6.6)
ax.axhspan(2.625, 3.49, color=OI["blue"], alpha=0.12)
ax.text(0.06, 3.06, "database\nV$_\\mathrm{O}$(2+/0)\nrange", fontsize=6.0, color=OI["blue"])
for x, a025, best, name, c, lbl, txy in [
        (1.0, 2.223, vo_best, "V$_\\mathrm{O}$", OI["blue"],
         f"best est. {vo_best:.2f}\n(DB 2.63)", (1.24, 2.30)),
        (1.75, 3.106, sb_best, "Sb$_\\mathrm{Ga}$", OI["purple"],
         f"CBM$-${Eg - sb_best:.1f}", (1.98, 3.28)),
        (2.45, 1.656, bi_best, "Bi$_\\mathrm{Ga}$", OI["verm"],
         f"CBM$-${Eg - bi_best:.1f}", (2.26, 2.42))]:
    ax.fill_between([x - 0.2, x + 0.2], best - BUD, best + BUD, color=c, alpha=0.18, lw=0)
    ax.hlines(a025, x - 0.2, x + 0.2, color=c, lw=1.1, ls=":")
    ax.hlines(best, x - 0.2, x + 0.2, color=c, lw=2.2)
    ax.annotate("", xy=(x, best), xytext=(x, a025),
                arrowprops=dict(arrowstyle="->", color=c, lw=0.9))
    ax.text(*txy, f"{name}\n{lbl}", fontsize=6.0, color=c)
ax.text(0.06, 0.45, "dotted: $\\alpha$=0.25 MP\nsolid: $\\alpha$=0.33 + exact aniso.\n"
                    "point charge (+ elastic, V$_\\mathrm{O}$)\nband: $\\pm$0.25 eV budget",
        fontsize=6.6, color="k")
ax.set_xlim(0, 3.1); ax.set_ylim(-0.4, Eg + 0.5)
ax.set_ylabel(r"$\varepsilon$(2+/0) above VBM (eV)")
ax.set_xticks([])
ax.set_title("charged levels: best estimate vs database", fontsize=8)
panel_label(ax, "b")

# ---------- (c) the two levers: freeze-in knob + Ta cliff ----------
ax = axs[2]
md = json.load(open(P / "paper/npj/figures/mech_data.json"))
fz = md["freezein_Si"]; ta = md["cliff_Ta"]
ax.plot(fz["Tf"], fz["log_n"], "o-", color=OI["green"], label="Si 1 at.%: $n(T_\\mathrm{f})$")
ax.set_xlabel(r"freeze-in temperature $T_\mathrm{f}$ (K)", color=OI["green"])
ax.set_ylabel(r"$\log_{10} n$ (cm$^{-3}$)")
ax2 = ax.twiny()
bb = json.load(open(P / "paper/npj/figures/budget_bands.json"))["cliff_Ta"]
lo = np.minimum(bb["deeper"], bb["shallower"]); hi = np.maximum(bb["deeper"], bb["shallower"])
ax2.fill_between(bb["log_pO2"], lo, hi, color=OI["verm"], alpha=0.14, lw=0,
                 label=r"$\pm$0.25 eV on Ta levels")
ax2.plot(ta["log_pO2"], ta["log_n"], "s--", color=OI["verm"], label=r"Ta 0.5 at.%: $n(p_{\mathrm{O}_2})$")
ax2.set_xlabel(r"$\log_{10} p_{\mathrm{O}_2}$ (atm)", color=OI["verm"])
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, loc="lower left", fontsize=6.6)
ax.set_title("freeze-in lever and compensation boundary", fontsize=8)
panel_label(ax, "c")

# ---------- (d) disorder-ensemble gap distribution ----------
ax = axs[3]
dd = json.load(open(P / "results/tier3/disorder_deg.json"))
gaps = [c["gap"] for c in dd["per_cell"]]
ms = [c["gap_eV"] for c in json.load(open(P / "results/tier3/disorder_deg_mattersim.json"))["cells"]]
bins = np.linspace(1.5, 2.45, 15)
cen = 0.5 * (bins[1:] + bins[:-1])
bw = bins[1] - bins[0]
h_mace, _ = np.histogram(gaps, bins=bins)
h_ms, _ = np.histogram(ms, bins=bins)
ax.bar(cen - bw / 4, h_mace, width=bw / 2 * 0.92, color=OI["sky"], edgecolor="white",
       label="MACE, 20 cells\n(12 recrystallized)")
ax.bar(cen + bw / 4, h_ms, width=bw / 2 * 0.92, color=OI["purple"], edgecolor="white",
       label="MatterSim, 3 cells\n(no recrystallization)")
ax.axvline(dd["crystal_gap_PBE_eV"], color="k", lw=1.2, ls=":")
ax.text(dd["crystal_gap_PBE_eV"] + 0.008, 5.4, "crystal", rotation=90, fontsize=7)
ax.set_xlabel("occupation-based gap (eV)"); ax.set_ylabel("cells")
ax.legend(frameon=False, loc="upper left", fontsize=6.6)
ax.set_title("disorder ensemble: gap distribution", fontsize=8)
panel_label(ax, "d")

fig.tight_layout(w_pad=2.0, h_pad=2.2)
fig.savefig(OUT / "fig_mechanisms.pdf")
plt.close(fig)
print("wrote fig_mechanisms.pdf")
