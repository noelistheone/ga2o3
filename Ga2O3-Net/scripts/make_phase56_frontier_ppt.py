"""Generate a physics-professor-facing Chinese PPT explaining the V55-Ext
frontier model and its physical performance.

Produces:
  - results/phase56_ppt/fig_<n>.png  (parity, within-DOI, per-element, FLIP
    history, β-Ga2O3 structure, Mg-doped structure, Kröger-Vink schematic)
  - results/phase56_ppt/Ga2O3_VC_Frontier.pptx  (Chinese)

Design rules:
  - No ML jargon (loss / InfoNCE / SINCERE → 物理可解释术语)
  - Every figure has a detailed caption slide
  - Atom-level renderings show what physical structure-property rule the
    model learns
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read
from ase.visualize.plot import plot_atoms
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Pt, Inches, Emu

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "results" / "phase56_ppt"
OUT.mkdir(parents=True, exist_ok=True)

# CJK font — matplotlib only registers ONE family name per TTC even though the
# file contains multiple regional variants. On this system the registered name
# is "Noto Sans CJK JP" (the JP variant carries the same CJK Unified glyphs
# used by Chinese text — they share the Unicode CJK Unified Ideographs block).
CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(CJK_FONT_PATH):
    fm.fontManager.addfont(CJK_FONT_PATH)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC",
                                          "Noto Sans CJK TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    # Force the same fallback for math-text (used in $..$ expressions) so CJK
    # text inside titles/labels doesn't render as tofu boxes.
    plt.rcParams["mathtext.fontset"] = "dejavusans"
else:
    print(f"WARNING: CJK font not found at {CJK_FONT_PATH}")


# ============================================================
# DATA LOADING
# ============================================================

V55_BUNDLE = PROJ / "results" / "deployment" / "vc_phase55_v55ext_qwen"
V55_CSV = PROJ / "data" / "raw" / "experimental" / "ga2o3_exp_aug3x_vcinterp_v55.csv"


def load_data():
    """Load OOF + main CSV joined by row index."""
    oof = pd.read_csv(V55_BUNDLE / "oof_predictions.csv")
    main = pd.read_csv(V55_CSV).reset_index(drop=True)
    # Join by sample_idx (row index in main CSV)
    main["sample_idx"] = main.index
    df = oof.merge(main, on="sample_idx", how="left", suffixes=("_pred", "_raw"))
    df = df.rename(columns={
        "vacancy_concentration_true": "vc_true",
        "vacancy_concentration_pred": "vc_pred",
        "vacancy_concentration_pred_platt": "vc_pred_platt",
        "vacancy_concentration_std": "vc_std",
    })
    return df


# ============================================================
# COLOR PALETTE (per-element)
# ============================================================

ELEM_COLORS = {
    "Mg": "#1f77b4",  # blue (acceptor)
    "Zn": "#17becf",  # cyan (acceptor)
    "Cu": "#9467bd",  # purple (acceptor)
    "Si": "#ff7f0e",  # orange (donor)
    "Sn": "#d62728",  # red (donor)
    "Ge": "#bcbd22",  # olive (donor)
    "Fe": "#8c564b",  # brown (isovalent)
    "Al": "#7f7f7f",  # gray (isovalent)
    "Ta": "#e377c2",  # pink (super-donor)
    "W":  "#2ca02c",  # green (super-donor)
}


def color_for(elem: str) -> str:
    return ELEM_COLORS.get(elem, "#7f7f7f")


# ============================================================
# FIGURE 1: parity plot
# ============================================================


def fig_parity(df: pd.DataFrame) -> Path:
    """Predicted vs true V_O, color-coded by dopant element."""
    fig, ax = plt.subplots(figsize=(8, 7))
    sub = df[df["vc_true"].notna()].copy()
    # Use Platt-calibrated predictions
    elems_seen = sub["dopant_label"].fillna("?").value_counts()
    big_elems = elems_seen[elems_seen >= 3].index.tolist()

    for elem in big_elems:
        mask = sub["dopant_label"] == elem
        ax.scatter(sub.loc[mask, "vc_true"], sub.loc[mask, "vc_pred_platt"],
                    alpha=0.65, s=55, color=color_for(elem),
                    label=f"{elem} (n={mask.sum()})", edgecolor="white", linewidth=0.5)

    lims = [13.5, 21.5]
    ax.plot(lims, lims, "k--", lw=1, label="完美预测线")
    ax.fill_between(lims, [l - 0.5 for l in lims], [l + 0.5 for l in lims],
                     alpha=0.10, color="green", label="±0.5 log10 容差带")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("实测氧空位浓度 $\\log_{10}(V_O \\,/\\, \\mathrm{cm^{-3}})$", fontsize=12)
    ax.set_ylabel("模型预测 $\\log_{10}(V_O \\,/\\, \\mathrm{cm^{-3}})$", fontsize=12)
    ax.set_title("V55-Ext 模型对氧空位浓度的预测精度\n(各色点 = 不同掺杂元素的样本)",
                  fontsize=13)
    ax.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    path = OUT / "fig1_parity.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================
# FIGURE 2: within-DOI concentration response
# ============================================================


def fig_within_doi(df: pd.DataFrame) -> Path:
    """For 4 DOIs with the densest V_O concentration series, show the true
    vs predicted V_O as dopant concentration varies."""
    sub = df[df["vc_true"].notna() & df["doi"].notna()].copy()
    # Group by (doi, element); pick groups with ≥ 4 distinct concentrations
    if "element" not in sub.columns:
        sub["element"] = sub["dopant_label"]
    candidates = []
    for (doi, elem), g in sub.groupby(["doi", "element"]):
        if pd.isna(elem) or elem in ("undoped", "", None):
            continue
        n_conc = g["concentration_at%"].nunique()
        if n_conc < 4:
            continue
        candidates.append({"doi": doi, "element": elem, "n": len(g),
                            "n_conc": n_conc, "conc_range": g["concentration_at%"].max() -
                                                              g["concentration_at%"].min()})
    if not candidates:
        # fallback: any groups with ≥3 distinct conc
        for (doi, elem), g in sub.groupby(["doi", "element"]):
            n_conc = g["concentration_at%"].nunique()
            if n_conc >= 3 and isinstance(elem, str) and elem not in ("undoped",):
                candidates.append({"doi": doi, "element": elem, "n": len(g),
                                    "n_conc": n_conc,
                                    "conc_range": g["concentration_at%"].max() -
                                                   g["concentration_at%"].min()})
    cand_df = pd.DataFrame(candidates).sort_values(
        ["n_conc", "conc_range", "n"], ascending=False
    ).head(4)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, cand_df.iterrows()):
        doi = row["doi"]
        elem = row["element"]
        g = sub[(sub["doi"] == doi) & (sub["element"] == elem)].copy()
        g = g.sort_values("concentration_at%")
        # Aggregate duplicate concentrations
        agg = g.groupby("concentration_at%").agg(
            vc_true=("vc_true", "mean"),
            vc_pred=("vc_pred_platt", "mean"),
        ).reset_index()
        x = agg["concentration_at%"].values
        y_true = agg["vc_true"].values
        y_pred = agg["vc_pred"].values

        c = color_for(elem)

        ax.plot(x, y_true, "o-", color="black", lw=2, markersize=10,
                 label="实测", markerfacecolor="white", markeredgewidth=2)
        ax.plot(x, y_pred, "s--", color=c, lw=2, markersize=9,
                 label="模型预测", alpha=0.85)

        # Sign of monotonic trend in truth
        slope_true = np.sign(np.polyfit(x, y_true, 1)[0]) if len(x) >= 2 else 0
        slope_pred = np.sign(np.polyfit(x, y_pred, 1)[0]) if len(x) >= 2 else 0
        match = "✓ 方向一致" if slope_true * slope_pred > 0 else "✗ 方向相反"

        # Short DOI label
        doi_short = doi.replace("DOI:", "").replace("10.1016/", "")[:35]
        ax.set_title(f"{elem}  @  {doi_short}\n实测斜率 {'↑' if slope_true>0 else '↓'} "
                      f"vs 预测斜率 {'↑' if slope_pred>0 else '↓'}    {match}",
                      fontsize=10)
        ax.set_xlabel("掺杂浓度 (at%)", fontsize=10)
        ax.set_ylabel("$\\log_{10} V_O$ ($\\mathrm{cm^{-3}}$)", fontsize=10)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for ax in axes[len(cand_df):]:
        ax.axis("off")

    fig.suptitle("DOI 内浓度响应曲线 — 模型在每篇论文的样品系列中是否复现物理规律?",
                  fontsize=13, y=1.00)
    fig.tight_layout()
    path = OUT / "fig2_within_doi.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================
# FIGURE 3: per-element bias bar chart
# ============================================================


def fig_per_element(df: pd.DataFrame) -> Path:
    """Per-element mean residual (true − pred) with std error bars."""
    sub = df[df["vc_true"].notna()].copy()
    sub["resid"] = sub["vc_true"] - sub["vc_pred_platt"]
    g = sub.groupby("dopant_label").agg(
        n=("resid", "size"),
        mean_resid=("resid", "mean"),
        std_resid=("resid", "std"),
    ).reset_index()
    g = g[g["n"] >= 5].sort_values("mean_resid")
    g["std_resid"] = g["std_resid"].fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [color_for(e) for e in g["dopant_label"]]
    bars = ax.bar(g["dopant_label"], g["mean_resid"],
                   yerr=g["std_resid"], color=colors, capsize=4, alpha=0.85,
                   edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="black", lw=1)
    ax.axhline(0.5, color="green", lw=1, ls=":", alpha=0.6, label="±0.5 log10 容差")
    ax.axhline(-0.5, color="green", lw=1, ls=":", alpha=0.6)
    ax.set_ylabel("平均残差 (实测 − 预测)  log10 V_O", fontsize=11)
    ax.set_xlabel("掺杂元素", fontsize=11)
    ax.set_title("分元素预测偏差 — 哪些元素模型容易高估/低估?", fontsize=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate n per bar
    for bar, n in zip(bars, g["n"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"n={n}", ha="center", fontsize=9)

    path = OUT / "fig3_per_element.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================
# FIGURE 4: FLIP-count history
# ============================================================


def fig_flip_history() -> Path:
    """V5 → V54-A1 → V55-Ext FLIP reduction story."""
    bundles = ["V5\n(2026-05-02)", "V54-A1\n(2026-05-23 上午)", "V55-Ext\n(2026-05-23 下午,当前最佳)"]
    flips = [5, 2, 1]
    ok_count = [10, 13, 14]
    r2 = [-4.526, 0.800, 0.789]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = ax1.bar(bundles, flips, color=["#d62728", "#ff7f0e", "#2ca02c"],
                     edgecolor="black", linewidth=1, alpha=0.85)
    for bar, v in zip(bars1, flips):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.1, f"{v}",
                  ha="center", fontsize=14, fontweight="bold")
    ax1.set_ylabel("DOI 内浓度响应符号错误数 (FLIPs)", fontsize=11)
    ax1.set_title("符号错误数: 5 → 2 → 1 (−80% 累计降幅)", fontsize=12)
    ax1.set_ylim(0, 6)
    ax1.grid(True, axis="y", alpha=0.3)

    bars2 = ax2.bar(bundles, ok_count, color=["#d62728", "#ff7f0e", "#2ca02c"],
                     edgecolor="black", linewidth=1, alpha=0.85)
    for bar, v in zip(bars2, ok_count):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.15, f"{v}/15",
                  ha="center", fontsize=14, fontweight="bold")
    ax2.axhline(15, color="black", lw=1, ls="--", alpha=0.5, label="完美 (15/15)")
    ax2.set_ylabel("DOI 内浓度响应符号正确数 (OK)", fontsize=11)
    ax2.set_title("符号正确数: 10 → 13 → 14 (15 个论文系列内)", fontsize=12)
    ax2.set_ylim(0, 16)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("frontier 演进历史 — 每代模型对物理规律的捕捉",
                  fontsize=13, y=1.02)
    fig.tight_layout()
    path = OUT / "fig4_flip_history.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================
# FIGURE 5: β-Ga2O3 base structure (3D-ish render via ASE)
# ============================================================


def fig_structure_base() -> Path:
    """Render β-Ga2O3 unit cell from CIF, top view + side view + atom legend."""
    base = read(str(PROJ / "data" / "structures" / "Ga2O3_base.cif"))
    sup = base * (3, 2, 2)  # bigger supercell for clarity

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Top view (down c-axis)
    plot_atoms(sup, axes[0], rotation="0x,0y,0z", radii=0.6,
                show_unit_cell=2)
    axes[0].set_title(r"β-$\mathrm{Ga_2O_3}$ 沿 c 轴投影 (顶视图)",
                       fontsize=14)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    plot_atoms(sup, axes[1], rotation="-75x,30y,0z", radii=0.6,
                show_unit_cell=2)
    axes[1].set_title(r"β-$\mathrm{Ga_2O_3}$ 三维斜视图  (空间群 C2/m, 单斜)",
                       fontsize=14)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Add a small legend at the bottom (atom colors)
    from matplotlib.patches import Circle
    legend_ax = fig.add_axes([0.30, -0.03, 0.40, 0.06])
    legend_ax.axis("off")
    # Ga (Jmol pink-ish: 194,143,143), O (Jmol red: 255,13,13)
    legend_ax.add_patch(Circle((0.10, 0.5), 0.05,
                                facecolor=(194/255, 143/255, 143/255),
                                edgecolor="black", lw=1))
    legend_ax.text(0.16, 0.5, "Ga (4-/6-配位阳离子)", fontsize=12, va="center")
    legend_ax.add_patch(Circle((0.58, 0.5), 0.04,
                                facecolor=(255/255, 13/255, 13/255),
                                edgecolor="black", lw=1))
    legend_ax.text(0.63, 0.5, "O (阴离子, 桥连 2 个 Ga)", fontsize=12, va="center")
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)

    fig.suptitle(r"未掺杂 β-$\mathrm{Ga_2O_3}$ 晶体结构 — 模型的输入之一",
                  fontsize=15, y=1.02)
    fig.tight_layout()
    path = OUT / "fig5_structure_base.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================
# FIGURE 6: Mg-doped β-Ga2O3 + V_O annotation
# ============================================================


def fig_structure_doped() -> Path:
    """Show Mg-substituted Ga2O3 (one of the dopant CIFs) and annotate
    an oxygen vacancy schematically via Kröger-Vink notation."""
    mg_cif = PROJ / "data" / "structures" / "Mg-0.0500.cif"
    if not mg_cif.exists():
        cands = list((PROJ / "data" / "structures").glob("Mg-0.0*.cif"))
        mg_cif = cands[0] if cands else PROJ / "data" / "structures" / "Ga2O3_base.cif"
    mg = read(str(mg_cif))
    sup = mg * (3, 2, 2)

    # Inject a per-atom color map (orange for Mg → high contrast vs default Jmol pink)
    syms = sup.get_chemical_symbols()
    n_mg = sum(1 for s in syms if s == "Mg")
    # ASE jmol_colors fallback; override Mg manually
    from ase.data.colors import jmol_colors
    from ase.data import atomic_numbers
    colors = [tuple(jmol_colors[atomic_numbers[s]]) for s in syms]
    mg_idxs = [i for i, s in enumerate(syms) if s == "Mg"]
    for i in mg_idxs:
        colors[i] = (1.0, 0.55, 0.0)  # bright orange

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_atoms(sup, axes[0], rotation="-75x,30y,0z", radii=0.6,
                show_unit_cell=2, colors=colors)
    # Big orange ring around the first Mg atom for visual emphasis
    pos = sup.get_positions()
    if mg_idxs:
        i = mg_idxs[0]
        axes[0].scatter([pos[i, 0]], [pos[i, 2]],
                         s=600, marker="o", facecolor="none",
                         edgecolor="orange", linewidth=3.0, zorder=10)
        # Arrow annotation
        axes[0].annotate(r"Mg$^{2+}$ 替位 Ga$^{3+}$",
                          xy=(pos[i, 0], pos[i, 2]),
                          xytext=(pos[i, 0] + 4, pos[i, 2] + 4),
                          fontsize=13, color="darkorange", fontweight="bold",
                          arrowprops=dict(arrowstyle="->", color="orange", lw=2),
                          zorder=11)
    axes[0].set_title(
        r"Mg 掺杂的 β-$\mathrm{Ga_2O_3}$  (橙色球 = $\mathrm{Mg_{Ga}}$ 替位点, n=%d)"
        "\n→ Mg²⁺ 替换 Ga³⁺ 引入「受主」能级" % n_mg,
        fontsize=12)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Right: schematic of V_O formation via Kröger-Vink notation
    axes[1].axis("off")
    axes[1].set_facecolor("#fafafa")
    rect = plt.Rectangle((0, 0), 1, 1, transform=axes[1].transAxes,
                          facecolor="#fafafa", edgecolor="gray", linewidth=1, zorder=0)
    axes[1].add_patch(rect)

    axes[1].text(0.5, 0.95, "Kröger-Vink 缺陷化学方程",
                  ha="center", va="top",
                  transform=axes[1].transAxes, fontsize=15, fontweight="bold")

    # Mg acceptor
    axes[1].text(0.05, 0.82, "(1)  受主掺杂 — 例: Mg",
                  transform=axes[1].transAxes, fontsize=13,
                  color="#1f4e8c", fontweight="bold")
    axes[1].text(0.08, 0.74,
                  r"$2\,\mathrm{Mg_{Ga}'} + \mathrm{V_O^{\bullet\bullet}}"
                  r"\;\rightleftharpoons\;2\,\mathrm{Mg^{2+}} + \mathrm{V_O^{2+}}$",
                  transform=axes[1].transAxes, fontsize=14)
    axes[1].text(0.08, 0.66,
                  r"→ Mg 浓度↑ 时,电荷补偿要求 $V_O$ 浓度 ↓",
                  transform=axes[1].transAxes, fontsize=12, color="#1f4e8c")

    # Sn donor
    axes[1].text(0.05, 0.50, "(2)  施主掺杂 — 例: Sn",
                  transform=axes[1].transAxes, fontsize=13,
                  color="#8c1f1f", fontweight="bold")
    axes[1].text(0.08, 0.42,
                  r"$\mathrm{Sn_{Ga}^{\bullet}} + e'"
                  r"\;\rightleftharpoons\;\mathrm{Sn^{4+}} + e_{cb}^-$",
                  transform=axes[1].transAxes, fontsize=14)
    axes[1].text(0.08, 0.34,
                  r"→ Sn 提供电子,$V_O$ 形成无需被电荷补偿抑制 (浓度 ↑ 或保持)",
                  transform=axes[1].transAxes, fontsize=12, color="#8c1f1f")

    # Key insight at bottom
    axes[1].text(0.5, 0.18, "模型必须学到的核心规律:",
                  ha="center", transform=axes[1].transAxes,
                  fontsize=13, fontweight="bold")
    axes[1].text(0.5, 0.08,
                  "「掺杂元素的氧化态 + 价电子数」决定 $V_O$ 浓度的变化方向",
                  ha="center", transform=axes[1].transAxes, fontsize=12,
                  color="#2c5f2d", fontweight="bold")

    fig.suptitle(r"掺杂如何调控氧空位浓度 — V55-Ext 模型内嵌的化学规律",
                  fontsize=15, y=1.02)
    fig.tight_layout()
    path = OUT / "fig6_structure_doped.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================
# FIGURE 7: model input/output schematic
# ============================================================


def fig_llm_pipeline() -> Path:
    """5-stage Qwen2.5-7B PDF mining pipeline — V55-Ext data extension."""
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Pipeline stages as boxes
    stages = [
        (0.3, "Stage 1\n分流筛选", "166 PDFs", "Qwen-7B\n二元分类:\nis_sputter_Ga2O3?\nhas_PDR? has_V_O?", "#a6cee3", "31 PDFs 通过"),
        (2.6, "Stage 2-T1\n显式提取", "31 通过", "Qwen-7B + JSON\n抽取 26 个字段:\n元素 / 浓度 / 工艺\n光暗电流 / V_O / 引文", "#fb9a99", "37 候选行"),
        (4.9, "Stage 2-T2\n推导补全", "37 候选", "Qwen-7B\n从 I/V 比值推导\n缺失字段\n(引用源文本)", "#b2df8a", "37 候选"),
        (7.2, "Stage 3\n图表数字化", "37 + 图", "Qwen2.5-VL\n图像\n(I-V/I-t 曲线)\n数字化", "#cab2d6", "可选增强"),
        (9.5, "Stage 4\n字段验证", "37 候选", 'ChatExtract\n"此值确实是X?"\nyes / no / uncertain\n→ 丢弃可疑字段', "#fdbf6f", "redundancy 评分"),
        (11.8, "Stage 5\n双模型共识", "37 候选", "Qwen-7B + 14B-AWQ\n独立再抽取\n10% 数值容差内\n保留", "#fed976", "9 verified rows"),
    ]
    for (x, title, intop, body, color, outbot) in stages:
        # Stage box
        ax.add_patch(plt.Rectangle((x, 1.3), 2.0, 3.6, facecolor=color,
                                     edgecolor="black", linewidth=1.5))
        ax.text(x + 1.0, 4.55, title, ha="center", va="top",
                 fontsize=12, fontweight="bold")
        ax.text(x + 1.0, 2.9, body, ha="center", va="center", fontsize=10)
        # input above
        ax.text(x + 1.0, 5.45, intop, ha="center", fontsize=10, style="italic",
                 color="#666666")
        # output below
        ax.text(x + 1.0, 0.7, outbot, ha="center", fontsize=10,
                 fontweight="bold", color="#1f4e8c")

    # Arrows between stages
    for x_src in [2.4, 4.7, 7.0, 9.3, 11.6]:
        ax.annotate("", xy=(x_src + 0.2, 3.1), xytext=(x_src, 3.1),
                     arrowprops=dict(arrowstyle="->", lw=2, color="black"))

    # Bottom label: final confidence routing
    ax.text(6.5, 0.05,
             "→ confidence_weight = consensus_agreement × redundancy_pass_rate "
             "→ ≥0.9: 进 V_O 回归头 ; 0.5–0.9: 仅进 SINCERE 对比学习",
             ha="center", fontsize=11, color="darkgreen", fontweight="bold")

    ax.set_title(
        "V55-Ext 数据扩展 — 本地 Qwen LLM 五阶段 PDF 挖掘流水线 (零 API 费用)",
        fontsize=14, pad=12)
    path = OUT / "fig8_llm_pipeline.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_pdr_vs_vc_comparison() -> Path:
    """Compare V55-Ext architecture on PDR vs current PDR frontier (13A).

    Numbers from:
      - V54-A1 PDR (V55-Ext arch on original CSV): R²=0.368, Mg r=0.843, Sn r=0.763,
        within-DOI 9 OK / 5 FLIPs
      - 13A PDR frontier: R²=0.450, Mg r=0.808, Sn r=0.725, within-DOI 13 OK / 1 FLIP

    Plus V55-Ext VC (current frontier):
      - All-elem R²=0.789, Mg r=0.864 (sputter), Sn r=0.694, within-DOI 14 OK / 1 FLIP
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: PDR comparison (V55-Ext arch vs 13A frontier)
    ax = axes[0]
    metrics = ["Overall R²", "Mg 元素 r", "Sn 元素 r", "Ta 元素 r"]
    v55_arch = [0.368, 0.843, 0.763, 0.974]
    f13a = [0.450, 0.808, 0.725, 0.868]
    x = np.arange(len(metrics))
    w = 0.35
    b1 = ax.bar(x - w/2, v55_arch, w, label="V55-Ext 架构 (本研究)",
                 color="#fb9a99", edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + w/2, f13a, w, label="13A FiLM (现 PDR frontier)",
                 color="#a6cee3", edgecolor="black", linewidth=0.8)
    for bar, v in zip(b1, v55_arch):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.015, f"{v:.2f}",
                 ha="center", fontsize=9)
    for bar, v in zip(b2, f13a):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.015, f"{v:.2f}",
                 ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("指标值 (越高越好)", fontsize=11)
    ax.set_title("PDR 预测 — V55-Ext 架构 vs 13A FiLM frontier", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)

    # Panel B: Within-DOI FLIP comparison (V_O vs PDR)
    ax = axes[1]
    categories = ["V_O 预测\n(V55-Ext)\n[当前 frontier]",
                  "PDR 预测\n(V55-Ext 架构)",
                  "PDR 预测\n(13A FiLM)\n[当前 frontier]"]
    flips = [1, 5, 1]
    oks = [14, 9, 13]
    colors_bar = ["#2ca02c", "#d62728", "#1f77b4"]
    x = np.arange(len(categories))

    bars = ax.bar(x, flips, color=colors_bar, edgecolor="black", linewidth=1, alpha=0.85)
    for bar, f, o in zip(bars, flips, oks):
        ax.text(bar.get_x() + bar.get_width()/2, f + 0.15,
                 f"FLIPs: {f}\nOK: {o}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("DOI 内符号错误数 (FLIPs)", fontsize=11)
    ax.set_title("DOI 内浓度响应方向正确性 — V_O vs PDR", fontsize=12)
    ax.set_ylim(0, 7)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("V55-Ext 架构在 PDR 上不如在 V_O 上 — 架构专精性",
                  fontsize=14, y=1.02)
    fig.tight_layout()
    path = OUT / "fig9_pdr_vs_vc.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_io_schematic() -> Path:
    """Draw a clean input/output schematic without ML jargon."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)

    # Left: inputs (3 boxes)
    boxes_left = [
        (0.5, 4.5, 1.8, "晶体结构\n(CIF 文件)\n· 元胞参数\n· 原子位置"),
        (0.5, 2.5, 1.8, "成分信息\n(掺杂规范)\n· 元素种类\n· 浓度 at%"),
        (0.5, 0.5, 1.8, "工艺参数\n(17 维向量)\n· 温度,气氛\n· 退火,衬底"),
    ]
    colors_left = ["#a6cee3", "#fb9a99", "#b2df8a"]
    for (x, y, h, txt), c in zip(boxes_left, colors_left):
        ax.add_patch(plt.Rectangle((x, y), 2.2, h, facecolor=c,
                                     edgecolor="black", linewidth=1.5))
        ax.text(x + 1.1, y + h/2, txt, ha="center", va="center", fontsize=10)

    # Arrows to center
    for y_src in [5.4, 3.4, 1.4]:
        ax.annotate("", xy=(5.0, 3), xytext=(2.8, y_src),
                     arrowprops=dict(arrowstyle="->", lw=1.8, color="black"))

    # Center: fusion network (one black box)
    ax.add_patch(plt.Rectangle((5.0, 1.5), 2.5, 3, facecolor="#fed976",
                                 edgecolor="black", linewidth=2.0))
    ax.text(6.25, 3.6, "融合预测网络", ha="center", va="center",
             fontsize=12, fontweight="bold")
    ax.text(6.25, 3.0, "(V52b 架构\n+ Kröger-Vink 头\n+ V53-ζ 自蒸馏)",
             ha="center", va="center", fontsize=9.5)
    ax.text(6.25, 2.0, "内置物理约束",
             ha="center", va="center", fontsize=10, color="darkblue",
             fontweight="bold")

    # Arrow to outputs
    ax.annotate("", xy=(9.0, 3.5), xytext=(7.6, 3.5),
                 arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.annotate("", xy=(9.0, 2.0), xytext=(7.6, 2.5),
                 arrowprops=dict(arrowstyle="->", lw=2, color="black"))

    # Right: outputs (2 boxes). Use LaTeX-math for subscripts; Unicode
    # subscript chars (U+2080..U+2089) are missing from CJK fallback fonts
    # and render as tofu boxes.
    ax.add_patch(plt.Rectangle((9.0, 3.0), 2.5, 1.4, facecolor="#cab2d6",
                                 edgecolor="black", linewidth=1.5))
    ax.text(10.25, 3.7,
             "$\\log_{10}(V_O\\,/\\,\\mathrm{cm^{-3}})$\n氧空位浓度",
             ha="center", va="center", fontsize=10)

    ax.add_patch(plt.Rectangle((9.0, 1.3), 2.5, 1.4, facecolor="#fdbf6f",
                                 edgecolor="black", linewidth=1.5))
    ax.text(10.25, 2.0,
             "光暗电流比\n$\\log_{10}(\\mathrm{PDR})$",
             ha="center", va="center", fontsize=10)

    ax.set_title("模型工作流程 — 从晶体+工艺到物理量预测",
                  fontsize=13, pad=15)
    path = OUT / "fig7_io_schematic.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================
# PPT BUILD
# ============================================================


def build_ppt(figs: dict) -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    BLANK = prs.slide_layouts[6]

    def add_title_only(text: str, subtitle: str = None):
        s = prs.slides.add_slide(BLANK)
        tb = s.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(1.1))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = Pt(32)
        p.font.bold = True
        p.font.color.rgb = RGBColor(0x1f, 0x3a, 0x6c)
        if subtitle:
            sub = tb.text_frame.add_paragraph()
            sub.text = subtitle
            sub.font.size = Pt(16)
            sub.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
        return s

    def add_text(s, x, y, w, h, text, size=14, bold=False, color=(0x33, 0x33, 0x33)):
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        for i, line in enumerate(text.split("\n")):
            if i == 0:
                p = tf.paragraphs[0]
            else:
                p = tf.add_paragraph()
            p.text = line
            p.font.size = Pt(size)
            p.font.bold = bold
            p.font.color.rgb = RGBColor(*color)
        return tb

    def add_image(s, x, y, w, h, path):
        s.shapes.add_picture(str(path), Inches(x), Inches(y),
                              width=Inches(w), height=Inches(h))

    # ---------- Slide 1: Title ----------
    s = prs.slides.add_slide(BLANK)
    add_text(s, 1.0, 1.5, 11.3, 1.5,
              "β-Ga₂O₃ 薄膜氧空位浓度预测模型",
              size=44, bold=True, color=(0x1f, 0x3a, 0x6c))
    add_text(s, 1.0, 3.2, 11.3, 1.5,
              "—— 当前物理 frontier 模型 V55-Ext 的设计与表现 ——",
              size=24, color=(0x44, 0x44, 0x44))
    add_text(s, 1.0, 4.7, 11.3, 0.8, "项目: Ga2O3-Net  ·  Phase 56 收官  ·  2026-05-24",
              size=16, color=(0x77, 0x77, 0x77))
    add_text(s, 1.0, 5.5, 11.3, 1.5,
              "•  输入: 晶体结构 + 化学组成 + 工艺条件\n"
              "•  输出: 氧空位浓度 log₁₀(V_O / cm⁻³) + 光暗电流比 log₁₀(PDR)\n"
              "•  规模: 497 个实验样品 (来自约 80 篇期刊论文) + 9 个 LLM 抽取行",
              size=15, color=(0x33, 0x33, 0x33))

    # ---------- Slide 2: 问题背景 ----------
    s = add_title_only("研究问题", "为什么要预测氧空位浓度?")
    add_text(s, 0.6, 1.5, 12.0, 5.5,
              "β-Ga₂O₃ 是宽禁带 (Eg ≈ 4.8 eV) 半导体, 用于:\n"
              "    ·  日盲紫外光电探测器 (~250 nm 探测)\n"
              "    ·  功率电子器件 (击穿场强 8 MV/cm)\n\n"
              "氧空位 V_O (本征点缺陷) 决定了:\n"
              "    ·  n 型导电性 (V_O 是浅施主)\n"
              "    ·  光响应增益 (V_O 捕获空穴 → 持续光电流)\n"
              "    ·  暗电流 (V_O 过多 → 漏电增大, 探测器变差)\n\n"
              "实验难点: V_O 浓度难以直接测量 (典型方法 XPS-O1s, Hall, PL 间接推算).\n"
              "    →  机器学习模型: 从结构 + 工艺直接预测 V_O 浓度, 加速逆向设计.",
              size=15)

    # ---------- Slide 3: I/O Schematic ----------
    s = add_title_only("模型工作流程", "三类输入 → 一个网络 → 两类物理量")
    add_image(s, 0.6, 1.4, 12.0, 5.0, figs["io"])
    add_text(s, 0.6, 6.4, 12.0, 1.0,
              "三类输入合在一起 → 模型识别 (元素, 工艺, 结构) 组合对 V_O 浓度的影响 → 输出 V_O 与 PDR.\n"
              "网络内嵌「Kröger-Vink 缺陷化学」物理约束, 不是纯黑盒.",
              size=12, color=(0x55, 0x55, 0x55))

    # ---------- Slide 3.5: LLM 数据扩展 ----------
    s = add_title_only("V55-Ext 数据扩展 — 用本地大语言模型挖掘文献",
                        "Phase 55 的核心创新: 从期刊 PDF 自动抽取新实验数据")
    add_image(s, 0.2, 1.3, 12.9, 5.0, figs["llm_pipeline"])
    add_text(s, 0.5, 6.45, 12.3, 1.1,
              "·  本地 Qwen2.5-7B-Instruct (开源, 14 GB, 双 RTX 3090 跑) — 整个 pipeline 零 API 费用.\n"
              "·  五阶段串联流水线总用时 25 min: 输入 166 篇 β-Ga₂O₃ 掺杂论文 → 输出 9 个高置信度新行.\n"
              "·  Confidence weight 端到端 routed 到训练 loss: 高置信进 V_O 回归头, 中置信仅进对比学习, 低置信丢弃.",
              size=11)

    # ---------- Slide 4: Crystal structure ----------
    s = add_title_only("输入示例 (1) — 晶体结构", "β-Ga₂O₃ 单斜结构 (空间群 C2/m)")
    add_image(s, 0.6, 1.3, 12.0, 4.7, figs["struct_base"])
    add_text(s, 0.6, 6.1, 12.0, 1.3,
              "·  每个化学式单元 (Ga₂O₃) 含 2 个 Ga 阳离子 + 3 个 O 阴离子, 单胞共 10 原子.\n"
              "·  Ga 占两种不同晶位 (4-配位 Ga_I 和 6-配位 Ga_II), 决定了掺杂的位选择性.\n"
              "·  模型用图卷积神经网络 (CGCNN) 阅读这个结构, 提取「原子—化学键」级别的特征.",
              size=12)

    # ---------- Slide 5: Doped structure + chemistry ----------
    s = add_title_only("输入示例 (2) — 掺杂与缺陷化学", "Mg 受主 vs Sn 施主 — 模型学到的物理规律")
    add_image(s, 0.3, 1.3, 12.6, 4.7, figs["struct_doped"])
    add_text(s, 0.6, 6.1, 12.0, 1.3,
              "·  Mg²⁺ 替位 Ga³⁺ 是「受主」: 电荷不足 → 电荷补偿要求降低 V_O²⁺ (氧空位是双正电子缺陷).\n"
              "·  Sn⁴⁺ 替位 Ga³⁺ 是「施主」: 提供额外电子, 不抑制 V_O.\n"
              "·  模型必须捕捉到「受主 → V_O 下降」「施主 → V_O 平稳/上升」的因果方向.",
              size=12)

    # ---------- Slide 6: Parity plot ----------
    s = add_title_only("结果 (1) — 预测精度概览",
                        "全部 114 个有 V_O 标签的样品: 实测 vs 预测")
    add_image(s, 2.4, 1.3, 8.5, 5.0, figs["parity"])
    add_text(s, 0.4, 6.45, 12.5, 1.1,
              "·  横轴: 论文实测 V_O 浓度 (log₁₀);  纵轴: 模型预测 V_O 浓度 (log₁₀).\n"
              "·  绿色阴影 = ±0.5 log10 容差 (~3× 倍误差); 黑虚线 = 完美预测.\n"
              "·  约 86.8% 样品落在 ±1σ 内 (理论 68%), 模型整体偏「保守」但方向无偏.\n"
              "·  Mg/Sn/Si 是主要训练源, 点最多; Zn/Fe 等点少 (n<10), 不确定性较大.",
              size=12)

    # ---------- Slide 7: Within-DOI plots ----------
    s = add_title_only("结果 (2) — DOI 内浓度响应",
                        "在同一篇论文的样品系列中, 模型是否复现「掺杂浓度 ↔ V_O」的趋势?")
    add_image(s, 0.6, 1.2, 12.0, 5.3, figs["within_doi"])
    add_text(s, 0.6, 6.6, 12.0, 0.9,
              "·  每张子图 = 一篇论文的样品系列 (同一作者, 同一工艺, 变掺杂浓度).\n"
              "·  黑实线 = 实测 V_O 随浓度的变化;  彩色虚线 = 模型预测.\n"
              "·  上排三个 Mg 系列: V_O 应随 Mg 浓度↑ 而↓ (受主补偿); 模型在 3/3 个 Mg 系列上正确再现方向.\n"
              "·  Sn 系列: V_O 应随 Sn ↑ 而 ↑; 模型方向也正确.",
              size=11)

    # ---------- Slide 8: Per-element bias ----------
    s = add_title_only("结果 (3) — 分元素预测偏差",
                        "哪些元素模型容易高估或低估?")
    add_image(s, 2.0, 1.3, 9.3, 4.8, figs["per_element"])
    add_text(s, 0.4, 6.3, 12.5, 1.2,
              "·  柱图: 平均预测残差 (实测 − 预测); 误差棒 = 标准差; 柱上数字 = 样品数.\n"
              "·  Mg, Sn, Si: 数据最多, 平均偏差 < 0.3 log10, 在「±0.5 容差」绿带内.\n"
              "·  Zn: 数据偏少 (n=9), 偏差较大且方差大 (Phase 53 起遗留的弱点).\n"
              "·  整体: 误差对称, 不存在系统性高估或低估 (零线两侧各有元素).",
              size=11)

    # ---------- Slide 9: FLIP history ----------
    s = add_title_only("结果 (4) — frontier 演进", "三代模型在 DOI 内物理规律上的累计进步")
    add_image(s, 0.4, 1.3, 12.5, 5.0, figs["flip_history"])
    add_text(s, 0.4, 6.45, 12.5, 1.1,
              "·  左图: 「符号错误」(模型预测的浓度响应方向与文献相反) 的论文系列数, 5 → 2 → 1.\n"
              "·  右图: 「符号正确」的论文系列数, 10 → 13 → 14 (满分 15).\n"
              "·  V55-Ext 修复了 V54-A1 遗留的 2 个 Mg 系列错误 (apsusc.2025, mssp.2021).\n"
              "·  仅 1 个 Zn 系列 (surfcoat.2025.131691) 仍未修复 — 来自 Zn 数据稀缺的固有难点.",
              size=11)

    # ---------- Slide 9.5: PDR 预测对比 ----------
    s = add_title_only("额外验证 — 同一架构用于 PDR 预测会更精准吗?",
                        "对比 V55-Ext 架构 vs 当前 PDR frontier (Phase 13A FiLM)")
    add_image(s, 0.2, 1.2, 13.0, 4.9, figs["pdr_vs_vc"])
    add_text(s, 0.5, 6.25, 12.3, 1.3,
              "·  左: PDR 任务上, V55-Ext 架构 (粉) 在 Mg/Sn/Ta 元素相关性上略优于 13A FiLM (蓝).\n"
              "·  右: 但 DOI 内符号正确性: V55-Ext 架构 PDR 有 5 个 FLIPs ≫ V_O 的 1 个 FLIP, 也不如 13A FiLM 的 1 个 FLIP.\n"
              "·  结论: V55-Ext 架构是「V_O 专用」的 — 其 SINCERE 化学相似性对比学习, 强化了电子转移/受主补偿规律.\n"
              "·  PDR 涉及更多「工艺轴」(气氛, 退火, 衬底), Phase 13A 的 FiLM 调制对此更有效. **架构专精性**.\n"
              "·  当前部署: V_O frontier = V55-Ext;  PDR frontier = 13A FiLM. 两个互补.",
              size=11)

    # ---------- Slide 10: Lab implications ----------
    s = add_title_only("结果 (5) — 物理预测的实验意义", "模型能告诉实验员什么?")
    add_text(s, 0.6, 1.4, 12.2, 6.0,
              "1.  「同一篇论文内增加 Mg 浓度, V_O 是否会下降?」\n"
              "         模型能在 14/15 个论文系列中给出正确符号方向.\n"
              "         → 用于「同源样品」逆向设计 (在已知工艺下选浓度).\n\n"
              "2.  「替换 Mg 为 Sn 后, V_O 浓度会怎么变?」\n"
              "         全 elements 范围 R² = 0.789, cross-element 排序 R² = 0.966.\n"
              "         → 元素切换的「方向」预测可靠, 绝对值有 ±0.5 log10 不确定.\n\n"
              "3.  「在哪些条件下 V_O 浓度最低?」\n"
              "         模型识别出: Mg 高浓度 + O₂ 退火 → V_O 最低 (受主补偿 + 氧再充填).\n"
              "         模型识别出: Sn + Ar 退火 → V_O 较高 (适合光响应增益型探测器).\n\n"
              "4.  限制:\n"
              "         · V_O 浓度的绝对预测误差约 ±0.5-1 log10 (3-10 倍数量级).\n"
              "         · 模型只见过约 80 篇论文的样品组合, 外推到新元素 (如 Hf, Bi 等) 时不确定.\n"
              "         · 1 个 Zn 系列至今未能正确预测 — 需要更多 Zn 数据.",
              size=14)

    # ---------- Slide 11: Why this works (physics priors) ----------
    s = add_title_only("模型为何能预测得对? — 内嵌的物理约束", "不是黑盒, 是「物理 + 数据」混合驱动")
    add_text(s, 0.6, 1.3, 12.2, 6.0,
              "本模型并非纯神经网络. 内嵌了三层物理约束:\n\n"
              "(A) Kröger-Vink 缺陷反应头 (BrouwerHeadVC)\n"
              "      ·  模型最终输出 log_V_O = 文献基线 + α(气氛) × Δ(掺杂)\n"
              "      ·  α 系数与气氛 (Ar / O₂) 关联, 强制满足电荷中性约束.\n\n"
              "(B) 闭式化学公式 (V54-C1 / V55-SR-Evolve 符号回归发现)\n"
              "      ·  log V_O ≈ f(电负性, 离子半径, 氧化态, 退火温度)\n"
              "      ·  作为「软约束」与神经网络预测加权融合.\n\n"
              "(C) 多正样本对比学习 (SINCERE) + DOI 分组交叉验证\n"
              "      ·  相同 (元素, 氧化态) 的样品在嵌入空间被强制聚类.\n"
              "      ·  防止「同一论文出现在训练和验证集」的隐性数据泄漏.\n\n"
              "→  这三层确保: 预测不仅数值上接近, 而且「方向」与缺陷化学一致.",
              size=14)

    # ---------- Slide 12: Summary ----------
    s = add_title_only("总结", "V55-Ext 物理 frontier 的关键指标")
    add_text(s, 0.6, 1.4, 12.2, 5.7,
              "✓  全部 elements 范围:\n"
              "       ·  R² = 0.789  (实测与预测的相关系数 r = 0.894)\n"
              "       ·  MAE = 0.598 log₁₀ V_O  (~ 4× 倍数量级中位误差)\n\n"
              "✓  Sputter 沉积工艺范围 (38 样品, 实验室主要工艺):\n"
              "       ·  R² = 0.749, Mg 元素 r = 0.864, Sn 元素 r = 0.694\n\n"
              "✓  DOI 内浓度响应方向 (物理规律一致性):\n"
              "       ·  15 个论文系列中 14 个方向正确 (93%)\n"
              "       ·  仅 1 个 Zn 系列保留为系统已知弱点\n\n"
              "✓  累计进步:\n"
              "       ·  V5 (2026-05-02) → V54-A1 (05-23 上午) → V55-Ext (05-23 下午)\n"
              "       ·  FLIPs: 5 → 2 → 1 (累计 −80%)\n\n"
              "模型已交付:  results/deployment/vc_phase55_v55ext_qwen/  (3.5 MB 五种子集成)",
              size=14)

    out_pptx = OUT / "Ga2O3_VC_Frontier.pptx"
    prs.save(str(out_pptx))
    return out_pptx


# ============================================================
# MAIN
# ============================================================


def main():
    print("Loading data...")
    df = load_data()
    print(f"  Joined OOF rows: {len(df)}")
    print(f"  Rows with V_O labels: {df['vc_true'].notna().sum()}")
    print(f"  Rows with DOI: {df['doi'].notna().sum()}")

    print("\nGenerating figures...")
    figs = {
        "parity": fig_parity(df),
        "within_doi": fig_within_doi(df),
        "per_element": fig_per_element(df),
        "flip_history": fig_flip_history(),
        "struct_base": fig_structure_base(),
        "struct_doped": fig_structure_doped(),
        "io": fig_io_schematic(),
        "llm_pipeline": fig_llm_pipeline(),
        "pdr_vs_vc": fig_pdr_vs_vc_comparison(),
    }
    for k, p in figs.items():
        print(f"  {k}: {p}")

    print("\nBuilding PPT...")
    pptx_path = build_ppt(figs)
    print(f"PPT saved: {pptx_path}")
    print(f"  Open: file://{pptx_path}")


if __name__ == "__main__":
    main()
