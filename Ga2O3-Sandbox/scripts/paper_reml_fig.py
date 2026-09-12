"""R8: redraw paper Fig. 2a from the current-corpus REML results (r8_icc_lrt.json):
bars = ICC point estimates, whiskers = parametric-bootstrap 95% CIs, annotation = exact
restricted LRT verdict. Sandbox-side variant (the original panel lives in the read-only Net
repo). Writes paper/npj/src/fig_reml.pdf."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parents[1]
OUT = P / "paper/npj/src"
NAVY = "#1f3b73"
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9,
                     "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "axes.linewidth": 0.8,
                     "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "DejaVu Sans"]})

ORDER = ["photo_dark_ratio", "dark_current_pA", "vacancy_concentration", "hall_carrier_n",
         "hall_mobility_mu", "optical_bandgap_Eg", "bandgap_shift_dEg", "tau_decay"]
LBL = {"photo_dark_ratio": "PDR", "dark_current_pA": r"$I_\mathrm{dark}$",
       "vacancy_concentration": r"[V$_\mathrm{O}$]", "hall_carrier_n": r"$n$",
       "hall_mobility_mu": r"$\mu$", "optical_bandgap_Eg": r"$E_g$",
       "bandgap_shift_dEg": r"$\Delta E_g$", "tau_decay": r"$\tau$"}

d = json.loads((P / "results/tier2/r8_icc_lrt.json").read_text())
vals = [d[k]["icc_reml"] for k in ORDER]
lo = [d[k]["icc_reml"] - d[k]["icc_ci95_parametric_bootstrap"][0] for k in ORDER]
hi = [d[k]["icc_ci95_parametric_bootstrap"][1] - d[k]["icc_reml"] for k in ORDER]

fig, ax = plt.subplots(figsize=(3.4, 2.6))
x = np.arange(len(ORDER))
ax.bar(x, vals, color=NAVY, width=0.62, yerr=[lo, hi], ecolor="#777777", capsize=2.5,
       error_kw={"lw": 0.9})
ax.set_xticks(x)
ax.set_xticklabels([LBL[k] for k in ORDER], fontsize=8)
ax.set_ylabel("between-study share (ICC)")
ax.set_ylim(0, 1.0)
ax.axhline(0.5, color="#bbbbbb", lw=0.7, ls=":")
ax.text(0.02, 0.955, r"all: $\sigma_b^2>0$, exact restricted LRT $p\leq 5{\times}10^{-4}$",
        transform=ax.transAxes, fontsize=6.8)
fig.tight_layout()
fig.savefig(OUT / "fig_reml.pdf", bbox_inches="tight", pad_inches=0.02)
print("wrote fig_reml.pdf")
