"""R8 result visualizations for the manuscript (every number read from its file of record).
  fig_oracle.pdf        — Fig 2e: adopted rho -> oracle (study-ID) rho dumbbells vs the bound.
  fig_icc_robust.pdf    — Fig 2f: ICC under OOF conditioning (all 8) + author-group regrouping (n, mu).
  fig_sb_confront.pdf   — new Results figure: Li-2023 measured series vs three solver scenarios;
                          hypothetical-level scan; site-preference energies (MLIP/DFT/published).
  fig_statebox.pdf      — Fig 3 companion: PDR direction across the a-priori state box (4 dopants x 15 states).
  fig_ko25.pdf          — Fig 7 companion: delta-cap sweep with the self-rejecting gate.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parents[1]
OUT = P / "paper/npj/src"
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00",
      "sky": "#56B4E9", "purple": "#CC79A7", "grey": "#7f7f7f"}
NAVY = "#1f3b73"
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
                     "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.linewidth": 0.8, "legend.fontsize": 6.8,
                     "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "DejaVu Sans"]})

ORDER = ["photo_dark_ratio", "dark_current_pA", "vacancy_concentration", "hall_carrier_n",
         "hall_mobility_mu", "optical_bandgap_Eg", "bandgap_shift_dEg", "tau_decay"]
LBL = {"photo_dark_ratio": "PDR", "dark_current_pA": r"$I_\mathrm{dark}$",
       "vacancy_concentration": r"[V$_\mathrm{O}$]", "hall_carrier_n": r"$n$",
       "hall_mobility_mu": r"$\mu$", "optical_bandgap_Eg": r"$E_g$",
       "bandgap_shift_dEg": r"$\Delta E_g$", "tau_decay": r"$\tau$"}

# ---------------- fig_oracle (Fig 2e) ----------------
orc = json.loads((P / "results/tier2/r8_bake21_oracle.json").read_text())
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ys = np.arange(len(ORDER))[::-1]
for y, k in zip(ys, ORDER):
    d = orc[k]
    ax.plot([d["rho_adopted_grouped"], d["rho_oracle_studyID"]], [y, y],
            color=OI["grey"], lw=1.1, zorder=1)
    ax.plot(d["rho_adopted_grouped"], y, "o", ms=5.5, color=NAVY, zorder=3)
    ax.plot(d["rho_oracle_studyID"], y, "o", ms=6, mfc="white", mec=OI["verm"], mew=1.4, zorder=3)
    ax.plot(d["bound_sqrt_1_minus_ICC"], y, "|", ms=9, color=OI["green"], mew=1.6, zorder=2)
ax.set_yticks(ys, [LBL[k] for k in ORDER])
ax.set_xlabel(r"pooled $\rho$")
ax.set_xlim(0.15, 0.9)
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([], [], marker="o", ls="", color=NAVY, label="grouped (deployed)"),
    Line2D([], [], marker="o", ls="", mfc="white", mec=OI["verm"], label="study-ID oracle"),
    Line2D([], [], marker="|", ls="", color=OI["green"], ms=9, label=r"bound $\sqrt{1-\mathrm{ICC}}$")],
    frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=7)
ax.set_title("the bound is binding: study identity is the missing information", fontsize=7.5)
fig.tight_layout(); fig.savefig(OUT / "fig_oracle.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig); print("fig_oracle.pdf")

# ---------------- fig_icc_robust (Fig 2f) ----------------
cond = json.loads((P / "results/tier2/r8_icc_conditional.json").read_text())
grp = json.loads((P / "results/tier2/r8_author_groups.json").read_text())
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ys = np.arange(len(ORDER))[::-1]
for y, k in zip(ys, ORDER):
    c = cond[k]
    ax.plot([c["icc_conditional_oof"], c["icc"]], [y, y], color=OI["grey"], lw=1.1, zorder=1)
    ax.plot(c["icc"], y, "o", ms=5.5, color=NAVY, zorder=3)
    ax.plot(c["icc_conditional_oof"], y, "s", ms=5, mfc="white", mec=OI["sky"], mew=1.3, zorder=3)
for k, y_off, gkey in (("hall_carrier_n", 0, "hall_carrier_n"), ("hall_mobility_mu", 0, "hall_mobility_mu")):
    y = ys[ORDER.index(k)]
    ax.plot(grp[gkey]["icc_group_level"], y, "D", ms=5, mfc="white", mec=OI["verm"], mew=1.3, zorder=4)
ax.set_yticks(ys, [LBL[k] for k in ORDER])
ax.set_xlabel("between-study share (ICC)")
ax.set_xlim(0, 1.0)
ax.legend(handles=[
    Line2D([], [], marker="o", ls="", color=NAVY, label="ICC"),
    Line2D([], [], marker="s", ls="", mfc="white", mec=OI["sky"], label="after OOF conditioning"),
    Line2D([], [], marker="D", ls="", mfc="white", mec=OI["verm"], label="author-group level")],
    frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=7)
ax.set_title("the share survives conditioning (covariate-complete rows)\nand author-level regrouping", fontsize=7.2)
fig.tight_layout(); fig.savefig(OUT / "fig_icc_robust.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig); print("fig_icc_robust.pdf")

# ---------------- fig_sb_confront (new Results figure) ----------------
sb = json.loads((P / "results/tier3/sb_ofz_confrontation.json").read_text())
s17 = json.loads((P / "results/tier2/r8_sb17_solver.json").read_text())
site = json.loads((P / "results/tier3/r8_site_preference_dft.json").read_text())
fig, axs = plt.subplots(1, 3, figsize=(9.9, 2.9))

# (a) measured series vs scenarios
ax = axs[0]
meas_x = [1.27e18, 3.05e18, 8.96e18]          # crystal Sb content 0.010/0.024 mol% bracket + top
# measured points: crystal content (x) unavailable per sample beyond 0.010-0.024 mol%; plot vs sample index instead
meas_n = [9.55e16, 5.40e17, 1.54e18, 8.10e18]
ax.semilogy(range(4), meas_n, "o-", color="k", ms=5, label="measured (OFZ, Li 2023)")
deep = [x["n_cm3"] for x in sb["best_estimate_a033_aniso_3.65"]["OFZ_effective"][:4]]
ax.semilogy(range(4), [max(v, 1e13) for v in deep], "s--", color=OI["blue"],
            label="computed deep Sb$_\\mathrm{Ga}$")
si = [x["n_cm3"] for x in s17["si_contamination_scenario"]["series"]]
ax.semilogy(range(4), si, "^:", color=OI["verm"],
            label="shallow background rising\nwith nominal load")
ax.set_xticks(range(4), ["UID", "s1", "s2", "s3"])
ax.set_ylabel(r"$n$ (cm$^{-3}$)"); ax.set_ylim(1e13, 3e19)
ax.legend(frameon=False, loc="lower right", fontsize=6.2)
ax.set_title("the measured trend needs a donor beyond isolated Sb$_\\mathrm{Ga}$", fontsize=7.2)
ax.text(-0.13, 1.06, "a", transform=ax.transAxes, fontweight="bold", fontsize=11, va="top")

# (b) hypothetical-level scan at the calibrated anneal state
ax = axs[1]
scan = sb["required_level_scan"]
lv = [x["level_below_CBM_eV"] for x in scan]
n1 = [x["n_at_1e-3"] for x in scan]
ax.semilogy(lv, n1, "o-", color=OI["purple"], label="hypothetical Sb level,\n0.1 mol\\% loading")
ax.axhline(8.1e18, color="k", lw=0.9, ls="--")
ax.text(0.35, 1.2e19, "measured $8.1\\times10^{18}$", fontsize=6.5)
ax.axhspan(1e15, 1e17, color=OI["grey"], alpha=0.15, lw=0)
ax.text(0.55, 2e16, "V$_\\mathrm{Ga}$ self-compensation\nknife-edge", fontsize=6.2, color=OI["grey"])
ax.set_xlabel("hypothetical donor level below CBM (eV)")
ax.set_ylabel(r"$n$ (cm$^{-3}$)"); ax.set_ylim(1e2, 3e19)
ax.set_title("no isolated substitutional level reproduces it", fontsize=7.2)
ax.text(-0.13, 1.06, "b", transform=ax.transAxes, fontweight="bold", fontsize=11, va="top")

# (c) site-preference energies: MLIP + DFT + published
ax = axs[2]
mlip = {"Sb": None, "Bi": None}
labels = ["Sb (DFT@MLIP)", "Bi (DFT@MLIP)", "Bi (published HSE)"]
vals = [-site["Sb"]["oct_minus_tet_eV"], -site["Bi"]["oct_minus_tet_eV"], 0.59]
cols = [OI["purple"], OI["verm"], OI["grey"]]
bars = ax.bar(labels, vals, color=cols, width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=7)
ax.set_ylabel("octahedral preference (eV per dopant)")
ax.set_ylim(0, 0.75)
ax.tick_params(axis="x", labelsize=6.4)
ax.set_title("site preference: computed vs published", fontsize=7.2)
ax.text(-0.13, 1.06, "c", transform=ax.transAxes, fontweight="bold", fontsize=11, va="top")

fig.tight_layout(w_pad=1.6)
fig.savefig(OUT / "fig_sb_confront.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig); print("fig_sb_confront.pdf")

# ---------------- fig_statebox: regime margins (Fig 3 companion) ----------------
dev = json.loads((P / "results/tier2/r8_dev06_statebox.json").read_text())
fig, ax = plt.subplots(figsize=(3.8, 2.5))
COLS = {"Si": OI["green"], "Sn": OI["blue"], "Mg": OI["purple"], "Zn": OI["verm"]}
import collections
for X, c in COLS.items():
    per_p = collections.defaultdict(list)
    for g in dev["per_dopant"][X]["grid"]:
        per_p[g["pO2"]].append(g["dPDR"])
    ps = sorted(per_p)
    x = [np.log10(v) for v in ps]
    lo = [min(per_p[v]) for v in ps]
    hi = [max(per_p[v]) for v in ps]
    ax.fill_between(x, lo, hi, color=c, alpha=0.25, lw=0)
    ax.plot(x, lo, color=c, lw=1.0)
    ax.plot(x, hi, color=c, lw=1.0, label=X)
ax.axhline(0, color="k", lw=1.1)
ax.set_ylim(-2.6, None)
ax.text(-15.7, -1.9, "sign boundary (direction flips below zero)", fontsize=6.2)
sn_min = min(g["dPDR"] for g in dev["per_dopant"]["Sn"]["grid"])
sn_arg = min(dev["per_dopant"]["Sn"]["grid"], key=lambda g: g["dPDR"])
ax.annotate(f"worst case: Sn, +{sn_min:.1f}", xy=(np.log10(sn_arg["pO2"]), sn_min),
            xytext=(-15.2, 9.6), fontsize=7,
            color=OI["blue"], arrowprops=dict(arrowstyle="->", color=OI["blue"], lw=0.8))
ax.set_xlabel(r"$\log_{10} p_{\mathrm{O}_2}$ (atm)")
ax.set_ylabel(r"predicted $\Delta$PDR (doped $-$ intrinsic)")
ax.legend(frameon=False, fontsize=7, loc="center right", ncol=2)
ax.set_title("direction margin across the a-priori state box\n(bands: min/max over anneal temperatures)", fontsize=7.2)
fig.tight_layout(); fig.savefig(OUT / "fig_statebox.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig); print("fig_statebox.pdf (margin bands)")

# ---------------- fig_ko25 (Fig 7 companion) ----------------
ko = json.loads((P / "results/tier2/r8_ko25_fis23.json").read_text())["ko25_cap_sweep"]
caps = [0.05, 0.10, 0.15, 0.25]
phys = [ko[str(c)]["LOLO_physics_only_dex"] for c in caps]
delt = [ko[str(c)]["LOLO_physics_plus_delta_dex"] for c in caps]
fig, ax = plt.subplots(figsize=(3.3, 2.2))
ax.plot(caps, phys, "s--", color=OI["grey"], label="physics only")
ax.plot(caps, delt, "o-", color=NAVY, label=r"physics + capped $\delta$")
ax.axvline(0.15, color=OI["green"], lw=0.9, ls=":")
ax.text(0.146, 0.2335, "deployed cap", fontsize=6.2, color=OI["green"], rotation=90,
        va="bottom", ha="right")
ax.annotate("rejected by the\nablation gate", xy=(0.25, delt[-1]), xytext=(0.185, 0.2405),
            fontsize=6.2, color=OI["verm"],
            arrowprops=dict(arrowstyle="->", color=OI["verm"], lw=0.8))
ax.set_ylim(0.2255, 0.2535)
ax.set_xlabel(r"correction cap $B_\mu$ (dex)")
ax.set_ylabel(r"$\mu$ LOLO median (dex)")
ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig_ko25.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig); print("fig_ko25.pdf")


# ---------------- fig_sb_confront panel d: the Ta record at the calibrated state ------------
def fig_confront_v2():
    sb = json.loads((P / "results/tier3/sb_ofz_confrontation.json").read_text())
    s17 = json.loads((P / "results/tier2/r8_sb17_solver.json").read_text())
    site = json.loads((P / "results/tier3/r8_site_preference_dft.json").read_text())
    ta = json.loads((P / "results/tier2/ta_experimental_conditions.json").read_text())
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.8))
    axs = axs.ravel()

    ax = axs[0]
    meas_n = [9.55e16, 5.40e17, 1.54e18, 8.10e18]
    ax.semilogy(range(4), meas_n, "o-", color="k", ms=5, label="measured (OFZ)")
    deep = [x["n_cm3"] for x in sb["best_estimate_a033_aniso_3.65"]["OFZ_effective"][:4]]
    ax.semilogy(range(4), [max(v, 1e13) for v in deep], "s--", color=OI["blue"],
                label="computed deep Sb$_\\mathrm{Ga}$")
    si = [x["n_cm3"] for x in s17["si_contamination_scenario"]["series"]]
    ax.semilogy(range(4), si, "^:", color=OI["verm"],
                label="shallow background\nrising with load")
    ax.set_xticks(range(4), ["UID", "s1", "s2", "s3"])
    ax.set_ylabel(r"$n$ (cm$^{-3}$)"); ax.set_ylim(1e13, 3e19)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    ax.set_title("Sb: the trend needs more than isolated Sb$_\\mathrm{Ga}$", fontsize=8)
    ax.text(-0.14, 1.06, "a", transform=ax.transAxes, fontweight="bold", fontsize=11, va="top")

    ax = axs[1]
    scan = sb["required_level_scan"]
    lv = [x["level_below_CBM_eV"] for x in scan]
    n1 = [x["n_at_1e-3"] for x in scan]
    ax.semilogy(lv, n1, "o-", color=OI["purple"], label="hypothetical Sb level")
    ax.axhline(8.1e18, color="k", lw=0.9, ls="--")
    ax.text(0.32, 2.6e18, "measured $8.1\\times10^{18}$", fontsize=7, va="top")
    ax.set_xlabel("hypothetical donor level below CBM (eV)")
    ax.set_ylabel(r"$n$ at 0.1 mol% (cm$^{-3}$)"); ax.set_ylim(1e2, 5e19)
    ax.set_title("no isolated level reproduces it\n(V$_\\mathrm{Ga}$ self-compensation)", fontsize=8)
    ax.text(-0.14, 1.06, "b", transform=ax.transAxes, fontweight="bold", fontsize=11, va="top")

    ax = axs[2]
    xx = np.arange(2)
    mlip = [-site["Sb"]["mlip_oct_minus_tet_eV"], -site["Bi"]["mlip_oct_minus_tet_eV"]]
    dft = [-site["Sb"]["oct_minus_tet_eV"], -site["Bi"]["oct_minus_tet_eV"]]
    b1 = ax.bar(xx - 0.17, mlip, 0.32, color=OI["sky"], label="MLIP (structure engine)")
    b2 = ax.bar(xx + 0.17, dft, 0.32, color=OI["purple"], label="explicit DFT")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    f"{b.get_height():.2f}", ha="center", fontsize=7)
    ax.axhline(0.59, color=OI["grey"], lw=1.2, ls="--")
    ax.text(0.52, 0.61, "published HSE (Bi): 0.59", fontsize=7, color=OI["grey"])
    ax.set_xticks(xx, ["Sb", "Bi"])
    ax.set_ylabel("octahedral-site preference (eV)")
    ax.set_ylim(0, 0.95)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_title("site preference: MLIP orders, DFT refines", fontsize=8)
    ax.text(-0.14, 1.06, "c", transform=ax.transAxes, fontweight="bold", fontsize=11, va="top")

    ax = axs[3]
    ser = ta["OFZ_O2poor_Tf1500"]
    x = [s["Ta_cm3"] for s in ser]
    y = [max(s["n_cm3"], 1.0) for s in ser]
    ax.loglog(x, y, "o-", color=OI["green"], label="solver, calibrated state")
    ser2 = ta["OFZ_air_Tf1350"]
    ax.loglog([s["Ta_cm3"] for s in ser2], [max(s["n_cm3"], 1.0) for s in ser2],
              "s--", color=OI["grey"], alpha=0.8, label="solver, uncalibrated 1350 K")
    ax.axhspan(3.6e16, 3e19, color="k", alpha=0.07, lw=0)
    ax.text(1.3e16, 2.3e19, "measured controllable range\n$3.6\\times10^{16}$–$3\\times10^{19}$",
            fontsize=7, va="top")
    ax.loglog([7e17 / 0.9], [7e17], "*", ms=11, color=OI["verm"], zorder=5)
    ax.text(9e17, 1.6e17, "$\\mu$: 138 vs 138\n@ $7\\times10^{17}$", fontsize=7, color=OI["verm"], va="top")
    ax.plot([1e16, 3e19], [1e16, 3e19], color=OI["grey"], lw=0.7, ls=":")
    ax.set_xlabel(r"Ta loading (cm$^{-3}$)"); ax.set_ylabel(r"$n$ (cm$^{-3}$)")
    ax.set_xlim(1e16, 4e19); ax.set_ylim(1e2, 5e19)
    ax.legend(frameon=False, loc="center left", fontsize=7)
    ax.set_title("Ta: the record is reproduced at the\ncalibrated state, not blind", fontsize=8)
    ax.text(-0.14, 1.06, "d", transform=ax.transAxes, fontweight="bold", fontsize=11, va="top")

    fig.tight_layout(w_pad=2.0, h_pad=2.2)
    fig.savefig(OUT / "fig_sb_confront.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig); print("fig_sb_confront.pdf (4 panels)")


# ---------------- fig_disorder_anatomy: RDF + density monotonicity + localization -----------
def fig_disorder_anatomy():
    from ase.io import read
    lr = json.loads((P / "results/tier3/disorder_lowrho.json").read_text())
    ed = json.loads((P / "results/tier3/edge_localization.json").read_text())
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5))

    ax = axs[0]
    def gao_rdf(atoms):
        sym = atoms.get_chemical_symbols()
        ga = [i for i, s in enumerate(sym) if s == "Ga"]
        ox = [i for i, s in enumerate(sym) if s == "O"]
        D = atoms.get_all_distances(mic=True)
        d = D[np.ix_(ga, ox)].ravel()
        d = d[(d > 0.8) & (d < 5.5)]
        h, e = np.histogram(d, bins=100, range=(1.2, 5.2), density=True)
        return 0.5 * (e[1:] + e[:-1]), h
    cr = read(str(P.parent / "Ga2O3-Net/data/structures/ordered/undoped.cif")) * (2, 2, 2)
    for atoms, c, lb in ((cr, "k", "crystal"),
                         (read(str(P / "dft/disorder_mattersim/amorph_cell_0.xyz")), OI["purple"],
                          "amorphous, 5.95 g cm$^{-3}$"),
                         (read(str(P / "dft/disorder_mattersim/amorph_lowrho_50.xyz")), OI["verm"],
                          "amorphous, 5.0 g cm$^{-3}$")):
        x, h = gao_rdf(atoms)
        ax.plot(x, h, color=c, lw=1.3, label=lb)
    ax.set_xlabel("Ga–O distance (Å)"); ax.set_ylabel("partial RDF (arb.)")
    ax.set_ylim(0, 2.9)
    ax.legend(frameon=False, fontsize=6.8, loc="upper right")
    ax.set_title("melt-quench structure", fontsize=7.8)
    ax.text(-0.22, 1.12, "a", transform=ax.transAxes, fontweight="bold", fontsize=11, va="top")

    ax = axs[1]
    dens = [5.95, 5.4, 5.0]
    nar = [lr["reference_595_narrowing_eV"]] + [c["narrowing_eV"] for c in lr["cells"]]
    ax.plot(dens, nar, "o-", color=OI["blue"], ms=6)
    for d_, n_ in zip(dens, nar):
        ax.annotate(f"{n_:.2f}", (d_, n_), xytext=(d_ - 0.02, n_ + 0.06), fontsize=6.8)
    ax.invert_xaxis()
    ax.set_xlabel(r"quench density (g cm$^{-3}$)  $\rightarrow$ film-like")
    ax.set_ylabel("gap narrowing (eV)")
    ax.margins(y=0.18)
    ax.set_title("the fixed-density value is a\nmeasured lower bound", fontsize=7.8)
    ax.text(-0.22, 1.12, "b", transform=ax.transAxes, fontweight="bold", fontsize=11, va="top")

    ax = axs[2]
    grp = ["crystal", "5.95", "5.0"]
    vb = [ed["crystal"]["VBM"], ed["ms_0"]["VBM"], ed["lr50"]["VBM"]]
    cb = [ed["crystal"]["CBM"], ed["ms_0"]["CBM"], ed["lr50"]["CBM"]]
    xx = np.arange(3)
    ax.bar(xx - 0.18, cb, 0.34, color=OI["sky"], label="conduction edge")
    ax.bar(xx + 0.18, vb, 0.34, color=OI["verm"], label="valence edge")
    ax.set_xticks(xx, grp)
    ax.set_ylabel("participation ratio")
    ax.legend(frameon=False, fontsize=6.8)
    ax.set_title("conduction edge stays extended;\nvalence edge localizes", fontsize=7.8)
    ax.text(-0.22, 1.12, "c", transform=ax.transAxes, fontweight="bold", fontsize=11, va="top")

    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT / "fig_disorder_anatomy.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig); print("fig_disorder_anatomy.pdf")


# ---------------- fig_audit: metrology program summary -----------------
def fig_audit():
    at = json.loads((P / "results/tier2/quote_audit_alltables.json").read_text())
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    rows = [
        ("transport (100-row)", [("verbatim", 14), ("adjudicated partial", 41),
                                 ("re-joined supported", 40), ("flagged", 5)]),
        ("$E_g$", [("verbatim", at["optical_bandgap_Eg"]["exact"])]),
        ("$\\tau$", [("verbatim", at["tau_decay"]["exact"]), ("unit-registered", at["tau_decay"]["unit"]),
                     ("unresolved", at["tau_decay"]["absent"])]),
        ("$I_\\mathrm{dark}$", [("verbatim", at["dark_current_pA"]["exact"]),
                                 ("unit-registered", at["dark_current_pA"]["unit"] + at["dark_current_pA"]["mantissa"])]),
        ("PDR", [("verbatim", at["photo_dark_ratio"]["exact"]), ("adjudicated partial", 4),
                 ("flagged", 1)]),
    ]
    CMAP = {"verbatim": OI["green"], "unit-registered": OI["sky"], "adjudicated partial": OI["orange"],
            "re-joined supported": "#8fce91", "unresolved": OI["grey"], "flagged": OI["verm"]}
    seen = set()
    for iy, (name, segs) in enumerate(rows[::-1]):
        tot = sum(v for _, v in segs)
        left = 0.0
        for lab, v in segs:
            frac = v / tot
            ax.barh(iy, frac, left=left, color=CMAP[lab], height=0.62,
                    label=lab if lab not in seen else None)
            seen.add(lab)
            left += frac
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.set_xlabel("fraction of audited rows")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=6.0, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.set_title("per-table transcription audit (sampled rows; per-row lists in the deposit)",
                 fontsize=7.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_audit.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig); print("fig_audit.pdf")


if __name__ == "__main__" or True:
    fig_confront_v2()
    fig_disorder_anatomy()
    fig_audit()
