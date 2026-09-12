"""Complete-data panel for the Si/Ge concentration-series claims: per-point measured n vs the
solver's prediction at the three freeze-in temperatures (uncalibrated 1073 K, donor-subset
1350 K, certified 1500 K). Writes results/tier2/r8_series_points.json + paper/npj/src/fig_series.pdf."""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sandbox import full_forward as ff, kroger_db

db = kroger_db.load()
OI = {"blue": "#0072B2", "green": "#009E73", "verm": "#D55E00", "purple": "#CC79A7",
      "sky": "#56B4E9", "grey": "#7f7f7f"}
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.4,
                     "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "DejaVu Sans"]})

tp = pd.read_csv(PROJ.parent / "Ga2O3-Net/results/phase65/transport_llm_extracted_v2.csv")
out = {}
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6))
PANEL = {"Si": "conc", "Ge": "parity"}
for ax, (doi, dop, film) in zip(axs, (("10.1063/5.0142107", "Si", True),
                                      ("10.1063/5.0058059", "Ge", True))):
    sub = tp[(tp.doi.astype(str).str.strip() == doi) & tp.carrier_cm3.notna()
             & tp.concentration_value.notna()].copy()
    sub = sub[sub.carrier_cm3 > 0]
    xs = sub.concentration_value.astype(float).to_numpy()
    ys = sub.carrier_cm3.astype(float).to_numpy()
    if dop == "Ge":
        keep = xs <= 100.0        # two rows carry impossible extracted concentrations (flagged)
        xs, ys = xs[keep], ys[keep]
    o = np.argsort(xs); xs, ys = xs[o], ys[o]
    if PANEL[dop] == "parity":
        pn = []
        for cv in xs:
            r = ff.forward(dop, float(cv) / 100.0, T_anneal=1500.0, pO2=1e-5, film=film, db=db)
            pn.append(max(r["hall_n_cm3"], 1.0))
        ax.loglog(pn, ys, "o", color=OI["verm"], ms=5)
        lim = [min(min(pn), min(ys)) * 0.5, max(max(pn), max(ys)) * 2]
        ax.plot(lim, lim, color=OI["grey"], lw=0.8, ls=":")
        ax.set_xlabel(r"solver $n$ at the certified state (cm$^{-3}$)")
        ax.set_ylabel(r"measured $n$ (cm$^{-3}$)")
        out[dop] = {"doi": doi, "conc_at_pct": xs.tolist(), "meas_n": ys.tolist(),
                    "pred_1500K": pn}
        continue
    ax.loglog(xs, ys, "o", color="k", ms=4.5, label="measured")
    rec = {"doi": doi, "dopant": dop, "conc_at_pct": xs.tolist(), "meas_n": ys.tolist(), "pred": {}}
    for Tf, c, ls in ((1073, OI["sky"], ":"), (1350, OI["blue"], "--"), (1500, OI["purple"], "-")):
        grid = np.unique(xs)
        pn = []
        for cv in grid:
            r = ff.forward(dop, float(cv) / 100.0, T_anneal=float(Tf), pO2=1e-5,
                           film=film, db=db)
            pn.append(max(r["hall_n_cm3"], 1.0))
        ax.loglog(grid, pn, ls, color=c, lw=1.3, label=f"solver, $T_\\mathrm{{f}}$={Tf} K")
        rec["pred"][str(Tf)] = {"conc": grid.tolist(), "n": pn}
    out[dop] = rec
    ax.set_xlabel(f"{dop} content (at.%)")
    ax.set_ylabel(r"$n$ (cm$^{-3}$)")
    ax.legend(frameon=False, loc="best")
ax0, ax1 = axs
ax0.set_title("Si series: rank-faithful ($\\rho=0.99$); slope rises\nmonotonically with the calibrable state", fontsize=7.2)
ax1.set_title("Ge series: sign fails; the solubility gate\nfires across the whole series (abstention)", fontsize=7.2)
ax0.text(-0.14, 1.09, "a", transform=ax0.transAxes, fontweight="bold", fontsize=11, va="top")
ax1.text(-0.14, 1.09, "b", transform=ax1.transAxes, fontweight="bold", fontsize=11, va="top")
fig.tight_layout(w_pad=1.8)
fig.savefig(PROJ / "paper/npj/src/fig_series.pdf", bbox_inches="tight", pad_inches=0.02)
(PROJ / "results/tier2/r8_series_points.json").write_text(json.dumps(out, indent=1))
print("wrote fig_series.pdf + r8_series_points.json;",
      {k: len(v["conc_at_pct"]) for k, v in out.items()}, "points")
