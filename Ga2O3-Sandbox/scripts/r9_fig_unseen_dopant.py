"""Regenerate the unseen-dopant figure with ELEMENT-CLUSTERED statistics throughout."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
PROJ = Path(__file__).resolve().parents[1]
d = json.load(open(PROJ / "results/revision_r9/unseen_dopant_perfom.json"))["per_fom"]
pooled = json.load(open(PROJ / "results/tier2/r8_pair26_clustered.json"))
LBL = {"hall_mobility_mu": r"$\mu$", "hall_carrier_n": r"$n$", "optical_bandgap_Eg": r"$E_g$",
       "vacancy_concentration": r"[V$_\mathrm{O}$]", "dark_current_pA": r"$I_\mathrm{dark}$",
       "photo_dark_ratio": "PDR", "tau_decay": r"$\tau$", "bandgap_shift_dEg": r"$\Delta E_g$"}
order = [k for k in LBL if d[k]["chem5"]] + [k for k in LBL if not d[k]["chem5"]]
GREEN, GREY = "#2e7d5b", "#b6b6b6"
fig, ax = plt.subplots(figsize=(7.0, 2.9))
for i, k in enumerate(order):
    v = d[k]; lo, hi = v["element_bootstrap_ci95"]
    ax.bar(i, v["pairwise_acc"], 0.66, color=GREEN if v["chem5"] else GREY,
           edgecolor="none", zorder=2)
    ax.errorbar(i, v["pairwise_acc"], yerr=[[v["pairwise_acc"]-lo], [hi-v["pairwise_acc"]]],
                fmt="none", ecolor="0.25", elinewidth=0.9, capsize=3, zorder=3)
    if v["element_permutation_p"] < 0.05:
        ax.text(i, hi + 0.035, "*", ha="center", va="bottom", fontsize=13, zorder=4)
    ax.text(i, 0.045, f"{v['n_elements']}", ha="center", va="bottom", fontsize=7,
            color="white" if v["chem5"] else "0.25", zorder=4)
ax.axhline(0.5, ls=":", lw=1.0, color="0.35", zorder=1)
ax.text(len(order)-0.35, 0.512, "chance", fontsize=7.5, color="0.35", va="bottom", ha="right")
ax.set_xticks(range(len(order))); ax.set_xticklabels([LBL[k] for k in order])
ax.set_ylabel("unseen-dopant pairwise accuracy"); ax.set_ylim(0, 1.05)
ax.set_title(f"pooled over the five chemistry-driven properties: "
             f"{pooled['chem5']['pooled_pairwise_acc']:.3f} "
             f"({pooled['chem5']['n_pairs']} pairs, element-permutation "
             f"$p={pooled['chem5']['element_permutation_p']:.3f}$)", fontsize=8.5)
for s in ("top", "right"): ax.spines[s].set_visible(False)
h = [plt.Rectangle((0,0),1,1,color=GREEN), plt.Rectangle((0,0),1,1,color=GREY)]
ax.legend(h, ["chemistry-driven (the five-property subset)", "device / laboratory-noise dominated"],
          fontsize=7.5, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0))
fig.tight_layout()
out = PROJ / "paper/npj/src/fig_unseen_dopant.pdf"
fig.savefig(out, bbox_inches="tight"); print("wrote", out)
