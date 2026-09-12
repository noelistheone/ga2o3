"""fig_floors.pdf — Fig 2 (ceiling) panel c: adopted-model LOEO placement-C against the three
pre-registered floors (1-NN, random projection, permutation-null q95), per property. Rebuilt
from the file of record (Ga2O3-Net phase99 cr3_second_floor.json, read-only) at a size legible
in the 0.48-textwidth composite slot; the previous ad-hoc generator was not retained.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parents[1]
NET = P.parent / "Ga2O3-Net"
OUT = P / "paper/npj/src"
NAVY = "#1f3b73"
OI = {"grey": "#b8b8b8", "sky": "#a8c6e8", "sand": "#e8d8a0"}
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
                     "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.linewidth": 0.8, "legend.fontsize": 6.4,
                     "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "DejaVu Sans"]})

ORDER = ["bandgap_shift_dEg", "hall_mobility_mu", "optical_bandgap_Eg", "vacancy_concentration",
         "tau_decay", "hall_carrier_n", "photo_dark_ratio", "dark_current_pA"]
LBL = {"photo_dark_ratio": "PDR", "dark_current_pA": r"$I_\mathrm{dark}$",
       "vacancy_concentration": r"[V$_\mathrm{O}$]", "hall_carrier_n": r"$n$",
       "hall_mobility_mu": r"$\mu$", "optical_bandgap_Eg": r"$E_g$",
       "bandgap_shift_dEg": r"$\Delta E_g$", "tau_decay": r"$\tau$"}

cr3 = json.loads((NET / "results/phase99/cr3_second_floor.json").read_text())
x = np.arange(len(ORDER))
w = 0.21
fig, ax = plt.subplots(figsize=(3.6, 2.7))
ax.bar(x - 1.5 * w, [cr3[k]["adopted_loeo_C"] for k in ORDER], w, color=NAVY,
       label="adopted model")
ax.bar(x - 0.5 * w, [cr3[k]["floor_1nn"] for k in ORDER], w, color=OI["grey"],
       label="1-NN floor")
ax.bar(x + 0.5 * w, [cr3[k]["floor_rp"] for k in ORDER], w, color=OI["sky"],
       label="random-projection floor")
ax.bar(x + 1.5 * w, [cr3[k]["perm_null_q95"] for k in ORDER], w, color=OI["sand"],
       label="permutation null (q95)")
ax.axhline(0.5, color="k", lw=0.8, ls=":")
ax.text(0.985, 0.505, "chance", fontsize=6.4, ha="right", va="bottom",
        transform=ax.get_yaxis_transform())
ax.set_xticks(x, [LBL[k] for k in ORDER], fontsize=7.5)
ax.set_ylabel("LOEO placement $C$")
ax.set_ylim(0.4, 0.84)
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
fig.tight_layout()
fig.savefig(OUT / "fig_floors.pdf", bbox_inches="tight", pad_inches=0.02)
print("wrote fig_floors.pdf")
