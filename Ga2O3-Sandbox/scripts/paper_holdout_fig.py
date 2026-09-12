"""fig_holdout.pdf — Fig 2 (ceiling) panel d: literature-model transfer onto the four
in-house devices, observed vs predicted doped-minus-intrinsic shifts for PDR (1/4) and
dark current (3/4). Rebuilt from the file of record (Ga2O3-Net phase105 holdout_lab.json,
read-only); no internal panel letters (the composite figure supplies the letter d).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path("/home/lawrence/Physics/Ga2O3-Sandbox")
NET = Path("/home/lawrence/Physics/Ga2O3-Net")
OUT = P / "paper/npj/src"
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00",
      "grey": "#bbbbbb", "navy": "#1f3a6d"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.8,
                     "savefig.bbox": "tight", "pdf.fonttype": 42})

hl = json.load(open(NET / "results/phase105/holdout_lab.json"))
ORDER = ["Mg", "Si", "Sn", "Zn"]

fig, axs = plt.subplots(2, 1, figsize=(3.35, 4.3))
for ax, key, name in [(axs[0], "photo_dark_ratio", r"photo-to-dark ratio (log$_{10}$)"),
                      (axs[1], "dark_current_pA", r"dark current (log$_{10}$ pA)")]:
    pairs = {p["elem"]: p for p in hl[key]["per_pair_direction"]}
    obs = [pairs[e]["obs_delta"] for e in ORDER]
    pred = [pairs[e]["pred_delta"] for e in ORDER]
    agree = [pairs[e]["agree"] for e in ORDER]
    X = np.arange(len(ORDER))
    ax.bar(X - 0.17, obs, width=0.34, color=OI["navy"], label=r"observed $\Delta$ (doped $-$ intrinsic)")
    ax.bar(X + 0.17, pred, width=0.34, color=OI["grey"], label=r"predicted $\Delta$")
    span = max(obs + pred) - min(obs + pred + [0])
    for x, o, p, a in zip(X, obs, pred, agree):
        ax.text(x, max(o, p, 0) + 0.05 * span,
                "match" if a else "miss", ha="center", fontsize=7,
                color=OI["green"] if a else "#C0392B", fontweight="bold")
        # near-zero bars are invisible: print their values so the flags stay interpretable
        if abs(o) < 0.06 * span:
            ax.text(x - 0.17, -0.055 * span, f"{o:+.2g}", ha="center", va="top",
                    fontsize=5.8, color=OI["navy"])
        if abs(p) < 0.06 * span:
            ax.text(x + 0.17, -0.055 * span, f"{p:+.2g}", ha="center", va="top",
                    fontsize=5.8, color="#6d6d6d")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(X, ORDER)
    ax.set_ylabel("$\\Delta$ vs. paired intrinsic film")
    n_ok = sum(agree)
    ax.set_title(f"{name}   (direction {n_ok}/{len(ORDER)})", fontsize=8)
    ax.margins(y=0.28)
axs[0].legend(frameon=False, loc="upper left", fontsize=6.2)
fig.tight_layout(h_pad=2.0)
fig.savefig(OUT / "fig_holdout.pdf")
plt.close(fig)
print("wrote fig_holdout.pdf")
