"""R7 reviewer #1 item 2: redraw the attenuation panel (paper Fig. 2b) with mobility's measured
relaxed bound (icc_orthogonality.json, ICC_perp) as an open marker. Sandbox-side variant of the
Ga2O3-Net panel (that repo is read-only); reads the same Net result files read-only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parents[1]
NET = P.parent / "Ga2O3-Net"
OUT = P / "paper/npj/src"
NAVY = "#1f3b73"; RED = "#c23b22"
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9,
                     "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "axes.linewidth": 0.8,
                     "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "DejaVu Sans"]})

FOM_KEYS = ["hall_mobility_mu", "optical_bandgap_Eg", "vacancy_concentration",
            "tau_decay", "hall_carrier_n", "photo_dark_ratio", "dark_current_pA"]
FOM_LBL = {"hall_mobility_mu": r"$\mu$", "optical_bandgap_Eg": r"$E_g$",
           "vacancy_concentration": r"[V$_\mathrm{O}$]", "tau_decay": r"$\tau$",
           "hall_carrier_n": r"$n$", "photo_dark_ratio": "PDR", "dark_current_pA": r"$I_\mathrm{dark}$"}

icc = json.loads((P / "results/tier2/r8_icc_lrt.json").read_text())
rho = json.loads((NET / "results/phase79/adopted_rho_phase79.json").read_text())
orth = json.loads((P / "results/tier2/icc_orthogonality.json").read_text())

fig, ax = plt.subplots(figsize=(3.4, 2.6))
for k in FOM_KEYS:
    b = float(np.sqrt(1 - icc[k]["icc_reml"]))
    o = rho[k] if isinstance(rho[k], float) else rho[k]["rho"]
    ax.plot(b, o, "o", ms=6, color=NAVY)
    dx, dy = 0.012, 0.012
    if k == "vacancy_concentration": dx, dy = -0.02, -0.045
    if k == "dark_current_pA": dy = -0.045
    if k == "tau_decay": dx, dy = 0.014, -0.038
    ax.annotate(FOM_LBL[k], (b, o), xytext=(b + dx, o + dy), fontsize=8)
    if k == "hall_mobility_mu":
        br = orth[k]["bound_icc_perp"]          # 0.76: measured relaxed bound
        ax.plot(br, o, "o", ms=6.5, mfc="none", mec=NAVY, mew=1.3)
        ax.annotate("", xy=(br - 0.012, o), xytext=(b + 0.012, o),
                    arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.8, ls=":",
                                    connectionstyle="arc3,rad=0.25"))
        ax.annotate(r"$\mu$ bound relaxed:" "\n" r"$R^2_{\rm cv}$=0.50 measured",
                    (br, o), xytext=(br - 0.16, o + 0.075), fontsize=6.8, color=NAVY)
lim = [0.2, 0.85]
ax.plot(lim, lim, "k--", lw=0.9)
ax.fill_between(lim, lim, [lim[1]] * 2, color=RED, alpha=0.07, lw=0)
ax.text(0.24, 0.76, "forbidden if lab effect\northogonal to features", fontsize=7.3, color=RED)
ax.set_xlabel("attenuation bound $\\sqrt{1-\\mathrm{ICC}}$\n(open marker: relaxed $\\sqrt{1-\\mathrm{ICC}_\\perp}$)")
ax.set_ylabel(r"adopted pooled $\rho$")
ax.set_xlim(*lim); ax.set_ylim(*lim)
fig.tight_layout()
fig.savefig(OUT / "fig_attenuation.pdf", bbox_inches="tight", pad_inches=0.02)
print("wrote fig_attenuation.pdf")
