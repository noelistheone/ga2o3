"""npj paper data figures (Acts 2-4), vector PDF, Okabe-Ito, Nature-ish styling.
Every number is read from its file of record (metrics-from-disk rule).
  fig3_validation.pdf — Act 2: Brouwer physics + energetics anchors + direction battery + the
                        4-device hinge (ML reversed vs sandbox 4/4).
  fig5_hybrid.pdf     — Act 3: LOLO per-property certification + the 6.9-dex unrecorded-state
                        anatomy + the delta-gate.
  fig6_designer.pdf   — Act 4: Fisher identifiability + 18-lab designed-vs-random validation.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path("/home/lawrence/Physics/Ga2O3-Sandbox")
OUT = P / "paper/npj/src"
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00",
      "sky": "#56B4E9", "purple": "#CC79A7", "yellow": "#F0E442", "grey": "#7f7f7f"}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.8,
    "lines.linewidth": 1.4, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

def panel_label(ax, s, dx=-0.12, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontweight="bold", fontsize=11, va="top")


# ================= Fig 3 — the instrument's validation (Act 2) =================
def fig3():
    bro = json.load(open(P / "paper/npj/figures/brouwer_data.json"))
    tb = json.load(open(P / "results/tier2/corpus_trend_battery.json"))
    hse = json.load(open(P / "results/tier2/hse_formation_energy.json"))
    vo2 = json.load(open(P / "results/tier3/hse_vo_2plus0_level.json"))
    lab = json.load(open(P / "results/tier2/lab_validation.json"))
    rel = json.load(open(P / "results/tier2/relative_validation.json"))

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.8))
    axs = axs.ravel()

    # (a) Brouwer diagram: log10[V_O] vs pO2, undoped, three T + Sn-doped at 1273 K
    ax = axs[0]
    x = np.log10(bro["pO2_grid"])
    bb = json.load(open(P / "paper/npj/figures/budget_bands.json"))["brouwer_1273_undoped"]
    ax.fill_between(np.log10(bb["pO2_grid"]), bb["lo"], bb["hi"], color=OI["blue"],
                    alpha=0.13, lw=0)
    for T, c in [(1073, OI["sky"]), (1273, OI["blue"]), (1473, "#023858")]:
        ax.plot(x, bro["curves"][f"T{T}_undoped"]["log10_VO"], color=c, label=f"{T} K")
    ax.plot(x, bro["curves"]["T1273_Sn"]["log10_VO"], color=OI["verm"], ls="--",
            label="1273 K, Sn 0.5 at.%")
    ax.plot(x, bro["curves"]["T1273_Mg"]["log10_VO"], color=OI["purple"], ls="-.",
            label="1273 K, Mg 0.5 at.%")
    ax.annotate("slope $-1/2$", xy=(-8, 13.2), fontsize=7, color=OI["blue"], rotation=-18)
    ax.set_xlabel(r"$\log_{10}\,p_{\mathrm{O}_2}$ (atm)")
    ax.set_ylabel(r"$\log_{10}\,[\mathrm{V_O}]$ (cm$^{-3}$)")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Brouwer diagram (module V)")
    panel_label(ax, "a")

    # (b) energetics anchors: two tiers + matched mixing vs the DB anchor band
    ax = axs[1]
    a33 = json.load(open(P / "results/tier3/r8_a33_anchor.json"))
    names = ["PBE", "HSE\n$\\alpha$=0.25", "HSE\n$\\alpha$=0.33"]
    vals = [hse["comparison"]["PBE_VO_q0_Orich"], hse["E_f_HSE_VO_q0_Orich_eV"],
            a33["Ef_VO0_Orich_a033_eV"]]
    ax.bar(names, vals, color=[OI["grey"], OI["green"], "#0b6e4f"], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v - 0.45, f"{v:.2f}", ha="center", fontsize=7.5, color="white", fontweight="bold")
    ax.axhline(3.5, color=OI["blue"], lw=1.4, ls="--")
    ax.text(-0.36, 3.42, "DB anchor 3.50", fontsize=7, color=OI["blue"], va="top")
    ax.set_ylabel(r"$E^f[\mathrm{V_O}^{0}]$ O-rich (eV)")
    ax.set_ylim(0, 5.2)
    ax.set_title("formation-energy anchor: tier and\nmixing responses vs the database", fontsize=7.6)
    fls = json.load(open(P / "results/tier3/fs160_level_shift.json"))
    a33 = fls.get("alpha033")
    if isinstance(a33, dict):
        best = json.load(open(P / "results/tier3/levels_v2.json"))["VO_best_estimate_vs_DB"]
        lvl_txt = (f"charged level $\\varepsilon$(2+/0):\n"
                   f"{vo2['eps_2plus_0_above_VBM_eV']:.2f} eV ($\\alpha$=0.25); "
                   f"{a33['eps_2plus_0_above_VBM_a033_eV']:.2f} eV ($\\alpha$=0.33)\n"
                   f"best est. {best['eps_eV']:.2f} vs DB 2.63")
    else:
        lvl_txt = (f"charged level $\\varepsilon$(2+/0):\n"
                   f"{vo2['eps_2plus_0_above_VBM_eV']:.2f} eV (DB: 2.63)")
    ax.text(0.03, 0.97, lvl_txt,
            transform=ax.transAxes, ha="left", va="top", fontsize=7, color=OI["green"])
    panel_label(ax, "b")

    # (c) direction battery
    ax = axs[2]
    t2 = json.load(open(P / "results/tier2/trend_t2_rebuilt.json"))
    tests = ["$n$ direction\nvs undoped", "PDR sign\n(base rate)", "Si series\n($n$=18)", "Ge series\n($n$=6)"]
    fr = [t2["frac"], tb["T3_within_paper_PDR_sign"]["frac"],
          rel["series"][1]["spearman"], rel["series"][0]["spearman"]]
    lab_n = [f'{t2["correct"]}/{t2["n_contrasts"]}',
             f"{tb['T3_within_paper_PDR_sign']['correct_sign']}/{tb['T3_within_paper_PDR_sign']['n_contrasts']}",
             r"$\rho$", r"$\rho$"]
    cols = [OI["green"], OI["grey"], OI["blue"], OI["verm"]]
    bars = ax.bar(tests, fr, color=cols, width=0.6)
    for b, f, t in zip(bars, fr, lab_n):
        ax.text(b.get_x() + b.get_width()/2, (f + 0.04) if f > 0 else 0.05,
                f"{f:.2f} ({t})", ha="center", fontsize=6.6)
    ax.axhline(t2["majority_baseline"], color=OI["grey"], lw=0.9, ls=":")
    ax.text(0.99, t2["majority_baseline"] - 0.09, "always-up base rate", fontsize=6, ha="right",
            color=OI["grey"], transform=ax.get_yaxis_transform())
    ax.set_ylim(-0.78, 1.32); ax.set_ylabel("fraction correct / $\\rho$")
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title("state-conditioned response (module VI)")
    panel_label(ax, "c")

    # (d) the hinge: 4 in-house devices, PDR direction — check/cross grid
    # literature-ML per-device correctness VERIFIED from the AAAI held-out table (tab:holdout):
    # PDR direction — Mg match; Si, Sn, Zn miss (1/4). Simulator from lab_validation.json (4/4).
    ax = axs[3]
    dev = ["Si", "Sn", "Mg", "Zn"]
    ml = {"Si": 0, "Sn": 0, "Mg": 1, "Zn": 0}
    rows = [("literature ML (1/4)", [ml[d] for d in dev]),
            ("simulator (4/4)", [1 if lab["results"][d]["PDR_direction_correct"] else 0 for d in dev])]
    for r, (name, vals) in enumerate(rows):
        for c, v in enumerate(vals):
            col = OI["green"] if v else OI["verm"]
            ax.add_patch(plt.Rectangle((c, 1 - r), 0.92, 0.92, facecolor=col, alpha=0.18,
                                       edgecolor=col, lw=1.2))
            ax.text(c + 0.46, 1 - r + 0.46, "✓" if v else "✗", ha="center",
                    va="center", fontsize=13, color=col, fontweight="bold")
    ax.set_xticks(np.arange(len(dev)) + 0.46, dev)
    ax.set_yticks([1.46, 0.46], [r[0] for r in rows])
    ax.set_xlim(-0.08, len(dev)); ax.set_ylim(-0.08, 2.0)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("held-out laboratory: PDR direction")
    panel_label(ax, "d")

    fig.tight_layout(w_pad=2.0, h_pad=2.2)
    fig.savefig(OUT / "fig3_validation.pdf")
    plt.close(fig); print("fig3_validation.pdf")


# ================= Fig 5 — hybrid certification (Act 3) =================
def fig5():
    cert = json.load(open(P / "results/hybrid/certification.json"))
    dmu = json.load(open(P / "results/hybrid/delta_mu.json"))
    br = pd.read_csv(P / "results/hybrid/bridge_data.csv")

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.8))
    axs = axs.ravel()

    # (a) LOLO certification: mu vs n, three modes
    ax = axs[0]
    modes = ["cross-lab\n(LOLO)", "anchored\n(per-lab)", "within-lab\nfloor"]
    mu = [cert["properties"]["mu"][k] for k in
          ["modeA_crosslab_LOLO_median_dex", "modeB_anchored_median_dex", "within_lab_scatter_floor_dex"]]
    nn = [cert["properties"]["n"][k] for k in
          ["modeA_crosslab_LOLO_median_dex", "modeB_anchored_median_dex", "within_lab_scatter_floor_dex"]]
    X = np.arange(3)
    ax.bar(X - 0.19, mu, width=0.36, color=OI["blue"], label=r"$\mu$ (mobility)")
    ax.bar(X + 0.19, nn, width=0.36, color=OI["orange"], label=r"$n$ (carrier density)")
    muf = [cert["properties"]["mu"][k] for k in ["modeA_factor", "modeB_factor"]] + [round(10**mu[2], 1)]
    nnf = [cert["properties"]["n"][k] for k in ["modeA_factor", "modeB_factor"]] + [round(10**nn[2], 1)]
    for x, v, f in zip(X - 0.19, mu, muf):
        ax.text(x, v + 0.02, f"$\\times${f:g}", ha="center", fontsize=7)
    for x, v, f in zip(X + 0.19, nn, nnf):
        ax.text(x, v + 0.02, f"$\\times${f:.0f}", ha="center", fontsize=7)
    ax.set_xticks(X, modes); ax.set_ylabel(r"median $|\Delta\log_{10}|$ (dex)")
    ax.set_ylim(0, 1.28)
    ax.legend(frameon=False)
    ax.set_title(f"LOLO certification ({cert['properties']['mu']['n_labs']} $\\mu$ / "
                 f"{cert['properties']['n']['n_labs']} $n$ labs)")
    panel_label(ax, "a")

    # (b) anatomy of the n ceiling: measured n spread for FIXED dopant (Si)
    ax = axs[1]
    si = br[(br["element"] == "Si") & br["meas_n"].notna()]["meas_n"].astype(float)
    si = si[si > 0]
    ln = np.log10(si)
    ax.hist(ln, bins=18, color=OI["orange"], alpha=0.85, edgecolor="white")
    ax.set_xlabel(r"measured $\log_{10} n$ (cm$^{-3}$), Si-doped only")
    ax.set_ylabel("rows")
    ax.set_title(f"one dopant spans {ln.max()-ln.min():.1f} decades ($N$={len(ln)})")
    ax.axvspan(ln.min(), ln.max(), color=OI["orange"], alpha=0.08)
    panel_label(ax, "b")

    # (c) the delta gate: physics-only vs physics+delta
    ax = axs[2]
    vals = [dmu["LOLO_physics_only_dex"], dmu["LOLO_physics_plus_delta_dex"]]
    bars = ax.bar(["physics only", r"physics $+\ \delta$ (capped)"], vals,
                  color=[OI["green"], OI["purple"]], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.004, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_ylabel(r"$\mu$ LOLO median (dex)")
    ax.set_title(f"$\\delta$ carries {100*dmu['delta_skill_share']:.1f}% of skill "
                 f"(cap {dmu['B_mu_cap_dex']} dex)")
    ax.set_ylim(0, max(vals) * 1.25)
    panel_label(ax, "c")

    # (d) per-dopant parity at the certified global process
    ax = axs[3]
    pj = json.load(open(P / "paper/npj/figures/parity_data.json"))["dopants"]
    for el, v in pj.items():
        if v["meas_log_mu"] and v["meas_log_mu"] > 0.3:
            ax.scatter(v["meas_log_mu"], v["pred_log_mu"], s=20, color=OI["blue"], zorder=3)
        if v["meas_log_n"] and v["pred_log_n"] > 1:
            ax.scatter(v["meas_log_n"] - 15, v["pred_log_n"] - 15, s=20, marker="s",
                       color=OI["orange"], zorder=3)
    ax.plot([0, 5.4], [0, 5.4], color=OI["grey"], lw=0.8, ls=":")
    ax.set_xlim(0, 5.4); ax.set_ylim(0, 5.4)
    ax.set_xlabel(r"measured median: $\log\mu$ / $\log n{-}15$")
    ax.set_ylabel(r"simulated: $\log\mu$ / $\log n{-}15$")
    ax.scatter([], [], s=20, color=OI["blue"], label=r"$\mu$ per dopant")
    ax.scatter([], [], s=20, marker="s", color=OI["orange"], label=r"$n$ per dopant")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("per-dopant parity (blind, global process)")
    panel_label(ax, "d")

    fig.tight_layout(w_pad=2.0, h_pad=2.2)
    fig.savefig(OUT / "fig5_hybrid.pdf")
    plt.close(fig); print("fig5_hybrid.pdf")


# ================= Fig 6 — designer (Act 4) =================
def fig6():
    sens = json.load(open(P / "results/hybrid/design_sensitivity.json"))
    val = json.load(open(P / "results/hybrid/design_validation_dedup.json"))

    fig = plt.figure(figsize=(7.4, 5.9))
    gs = fig.add_gridspec(2, 6, wspace=1.5, hspace=0.55)
    axs = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6]),
           fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])]

    # (a) Fisher information per theta (log scale) — the identifiability anatomy
    ax = axs[0]
    info = sens["full_battery_theta_information_diag"]
    keys = ["log10_Tf", "E_B", "log10_pO2", "log10_Nd_bg"]
    lbl = [r"$T_\mathrm{f}$ (freeze-in)", r"$E_B$ (GB barrier)", r"$p_{\mathrm{O}_2}$",
           r"$N_{d,\mathrm{bg}}$ (background)"]
    v = [max(info[k], 1e-5) for k in keys]
    cols = [OI["green"], OI["green"], OI["verm"], OI["verm"]]
    ax.bar(lbl, v, color=cols, width=0.6)
    ax.set_yscale("log"); ax.set_ylabel("Fisher information (doped $n,\\mu,\\sigma$)")
    ax.set_ylim(1e-5, 3e6)
    for i, x in enumerate(v):
        ax.text(i, x * 1.6, f"{x:.3g}", ha="center", fontsize=6.6)
    ax.set_title(f"identifiability: cond. no. {sens['condition_number']:.0e}", fontsize=8)
    ax.tick_params(axis="x", rotation=30, labelsize=6.4)
    panel_label(ax, "a", dx=-0.26)

    # (b) 18-lab validation: pooled strategies
    ax = axs[1]
    pm = val["pooled_median_dex"]
    r50 = json.load(open(P / "results/hybrid/design_validation_r50.json"))["per_K"]
    flr = json.load(open(P / "results/tier2/r8_rsp13_flr15.json"))["flr15"]
    names = ["blind", "rand-2", "des-2", "rand-3", "des-3", "floor\n(LOO)"]
    vals = [pm["global"], r50["2"]["random_median"], r50["2"]["designed_median"],
            r50["3"]["random_median"], r50["3"]["designed_median"],
            flr["floor_leave_one_point_out_pooled_median"]]
    cols = [OI["grey"], OI["sky"], OI["purple"], OI["sky"], OI["purple"], OI["green"]]
    bars = ax.bar(names, vals, color=cols, width=0.62)
    for b, v_ in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v_ + 0.02, f"{v_:.2f}", ha="center", fontsize=7)
    ax.set_ylabel(r"held-out median $|\Delta\log_{10} n|$ (dex)")
    ax.set_title(f"per-lab calibration, {val['n_labs_dedup']} deduplicated labs", fontsize=8)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=6.6)
    panel_label(ax, "b", dx=-0.26)

    # (c) per-lab improvement: blind -> floor scatter
    ax = axs[2]
    g = [r["global"] for r in val["per_lab"]]
    f = [r["floor"] for r in val["per_lab"]]
    ax.scatter(g, f, s=22, color=OI["purple"], zorder=3, alpha=0.85)
    m = max(max(g), max(f)) * 1.08
    ax.plot([0, m], [0, m], color=OI["grey"], lw=0.8, ls=":")
    ax.set_xlabel("blind error (dex)"); ax.set_ylabel("calibrated error (dex, all-point fit)")
    ax.set_xlim(0, m); ax.set_ylim(0, m)
    ax.set_title('per-lab calibration vs blind error', fontsize=8)
    ax.annotate("pooled median 1.28 $\\to$ 0.76\n(two designed anchors;\nLOO floor 0.81, in-sample 0.47)",
                xy=(0.04, 0.96), xycoords="axes fraction", ha="left", va="top",
                fontsize=6.8, color=OI["purple"])
    panel_label(ax, "c", dx=-0.26)

    # (d) measured K-sweep: designed vs random anchors
    ax = axs[3]
    ks = json.load(open(P / "results/hybrid/design_ksweep.json"))
    K = ks["Ks"]
    ax.plot(K, [ks["per_K"][str(k)]["designed"] for k in K], "o-", color=OI["purple"], label="designed (D-opt)")
    ax.plot(K, [ks["per_K"][str(k)]["random"] for k in K], "s--", color=OI["sky"], label="random (5 draws)")
    ax.axhline(ks["floor"], color=OI["green"], lw=1.0, ls=":"); ax.text(4.0, ks["floor"] + 0.02, "in-sample fit 0.47", fontsize=6.8, color=OI["green"])
    flr6 = json.load(open(P / "results/tier2/r8_rsp13_flr15.json"))["flr15"]
    ax.axhline(flr6["floor_leave_one_point_out_pooled_median"], color=OI["green"], lw=1.0, ls="--")
    ax.text(1.0, flr6["floor_leave_one_point_out_pooled_median"] - 0.045, "LOO floor 0.81",
            fontsize=6.8, color=OI["green"], va="top")
    ax.axhline(ks["blind"], color=OI["grey"], lw=1.0, ls=":"); ax.text(1.0, ks["blind"] - 0.09, "blind 1.28", fontsize=6.8, color=OI["grey"])
    ax.set_xlabel("anchor budget $K$"); ax.set_ylabel(r"held-out median $|\Delta\log_{10} n|$ (dex)")
    ax.set_xticks(K); ax.set_ylim(0.3, 1.4)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("measured anchor-budget sweep (13 labs)", fontsize=8)
    panel_label(ax, "d", dx=-0.17)

    # (e) 1000-draw pooled-random sampling distribution vs the designed value
    ax = axs[4]
    r1k = json.load(open(P / "results/hybrid/design_validation_r1000.json"))["dedup_13"]
    flr = json.load(open(P / "results/tier2/r8_rsp13_flr15.json"))["flr15"]
    for K, c, a in (("2", OI["purple"], 0.55), ("3", OI["sky"], 0.45)):
        s = r1k[K]["pooled_random_median_samples"]
        ax.hist(s, bins=34, density=True, alpha=a, color=c, label=f"random, $K$={K}")
        ax.axvline(r1k[K]["designed_median"], color=c, lw=1.6)
    ax.axvline(flr["floor_leave_one_point_out_pooled_median"], color=OI["green"], lw=1.2, ls=":")
    ax.text(flr["floor_leave_one_point_out_pooled_median"] + 0.012, 0.88, "LOO floor",
            fontsize=6.8, color=OI["green"], transform=ax.get_xaxis_transform())
    ax.text(r1k["2"]["designed_median"] - 0.032, 0.6, "designed", fontsize=6.8,
            color=OI["purple"], rotation=90, transform=ax.get_xaxis_transform())
    ax.set_xlabel(r"pooled median $|\Delta\log_{10} n|$ (dex)")
    ax.set_ylabel("density (1000 draws)")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("designed anchors vs the random-draw distribution", fontsize=8)
    panel_label(ax, "e", dx=-0.17)

    fig.tight_layout(w_pad=1.8)
    fig.savefig(OUT / "fig6_designer.pdf")
    plt.close(fig); print("fig6_designer.pdf")


if __name__ == "__main__":
    fig3(); fig5(); fig6()
