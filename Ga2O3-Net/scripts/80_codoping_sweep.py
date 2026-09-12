"""V62 co-doping MECHANISM sweep (the primary scientific deliverable; design docs/phase62 sec 5+7).

Built ENTIRELY on the verified multi-dopant solver (no co-doping data needed, none exists). For a curated
set of dopant pairs spanning the physically-distinct regimes, computes the equilibrium Brouwer response
and emits FALSIFIABLE directional predictions for the lab:

  - donor+acceptor  (compensation): the Fermi-level tug-of-war, compensation ratio, V_O self-compensation
  - donor+donor     : E_F stays high, V_O^2+ suppressed (no compensation)
  - acceptor+acceptor: E_F driven low
  - donor+isovalent : isovalent dopant is charge-neutral -> donor controls E_F (null co-doping effect)

Outputs results/phase62_codoping/codoping_sweep.json + Brouwer figures. CPU, deterministic.
"""
from __future__ import annotations
import json, sys, math
from pathlib import Path
import numpy as np
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from src.models.defect_equilibrium import EG_GA2O3  # noqa: E402
from src.models.defect_equilibrium_codoping import solve_equilibrium_codoping, elem_transition_level  # noqa: E402
from src.models.v61_net import carrier_sign  # noqa: E402

torch.set_default_dtype(torch.float64)
EG = EG_GA2O3
MIDGAP = 0.5 * EG
DHF = [3.5, 1.8, 0.3]                      # native V_O HSE06 baseline (host)

# curated pairs: (label, A, B). signs come from the project carrier table.
PAIRS = [
    ("Sn+Mg", "Sn", "Mg"), ("Si+Mg", "Si", "Mg"), ("Sn+Zn", "Sn", "Zn"), ("Si+N", "Si", "N"),
    ("Sn+Si", "Sn", "Si"),                 # donor+donor
    ("Mg+Zn", "Mg", "Zn"),                 # acceptor+acceptor
    ("Sn+Fe", "Sn", "Fe"), ("Sn+Al", "Sn", "Al"),   # donor+isovalent
]


def _solve(zA, cA, eA, zB, cB, eB, T_K=973.15, log_pO2=0.0):
    """Vectorized solve over arrays of (cA,cB) at fixed T,pO2. Returns dict of np arrays."""
    cA = np.atleast_1d(cA).astype(float); cB = np.atleast_1d(cB).astype(float)
    n = len(cA)
    dH = torch.tensor([DHF] * n)
    z = torch.tensor([[zA, zB]] * n)
    c = torch.tensor(np.stack([cA, cB], axis=1))
    elev = torch.tensor([[eA, eB]] * n)
    mask = torch.tensor(np.stack([(cA > 0).astype(float), (cB > 0).astype(float)], axis=1))
    out = solve_equilibrium_codoping(dH, torch.full((n,), T_K), torch.full((n,), log_pO2),
                                     z, c, elev, dop_mask=mask)
    return dict(E_F=out["E_F"].numpy(), logVO=out["log10_VO"].numpy(),
                VO2=out["conc"][:, 2].clamp_min(1.0).log10().numpy(),
                ionA=out["dop_ion"][:, 0].numpy(), ionB=out["dop_ion"][:, 1].numpy())


def analyze_pair(label, A, B):
    zA, zB = carrier_sign(A), carrier_sign(B)
    eA, eB = elem_transition_level(A, zA), elem_transition_level(B, zB)
    # 1D compensation curve: fix the donor (or A) at 1%, sweep the partner 1e-4..5e-2
    c_fix = 0.01
    c_sweep = np.logspace(-4, math.log10(0.05), 40)
    # choose which is donor for the "fix donor, add partner" narrative
    if zA > 0 and zB < 0:      # A donor, B acceptor: fix A, sweep B(acceptor)
        res = _solve(zA, np.full_like(c_sweep, c_fix), eA, zB, c_sweep, eB)
        partner, partner_sign = B, "acceptor"
    elif zB > 0 and zA < 0:    # B donor, A acceptor: fix B, sweep A
        res = _solve(zB, np.full_like(c_sweep, c_fix), eB, zA, c_sweep, eA)
        partner, partner_sign = A, "acceptor"
    else:                       # donor+donor / acc+acc / donor+isovalent: fix A, sweep B
        res = _solve(zA, np.full_like(c_sweep, c_fix), eA, zB, c_sweep, eB)
        partner, partner_sign = B, ("donor" if zB > 0 else "acceptor" if zB < 0 else "isovalent")
    EF, logVO, VO2 = res["E_F"], res["logVO"], res["VO2"]
    # compensation point: smallest partner conc where E_F crosses below mid-gap
    comp_idx = np.where(EF < MIDGAP)[0]
    comp_c = float(c_sweep[comp_idx[0]]) if len(comp_idx) else None
    comp_ratio = (comp_c / c_fix) if comp_c is not None else None
    # V_O self-compensation: change in log[V_O] and log[V_O^2+] from min to max partner conc
    dlogVO = float(logVO[-1] - logVO[0]); dVO2 = float(VO2[-1] - VO2[0])
    dEF = float(EF[-1] - EF[0])
    return dict(
        label=label, A=A, B=B, zA=zA, zB=zB, eA_eV=eA, eB_eV=eB,
        regime=("donor+acceptor (compensation)" if zA * zB < 0 else
                "donor+donor" if zA > 0 and zB > 0 else
                "acceptor+acceptor" if zA < 0 and zB < 0 else
                "donor+isovalent (null)" if (zA == 0) ^ (zB == 0) else "isovalent"),
        narrative=f"fix {A if (zA>0 or zB<=0) else B} at {c_fix:.0%}, sweep {partner} ({partner_sign})",
        c_fixed=c_fix, c_sweep=c_sweep.tolist(),
        E_F=EF.tolist(), log10_VO=logVO.tolist(), log10_VO2plus=VO2.tolist(),
        compensation_conc=comp_c, compensation_ratio=comp_ratio,
        d_EF_eV=dEF, d_log10_VO=dlogVO, d_log10_VO2plus=dVO2,
        falsifiable=_falsifiable_claim(label, partner, partner_sign, zA, zB, comp_ratio, dEF, dlogVO, dVO2),
    )


def _falsifiable_claim(label, partner, psign, zA, zB, ratio, dEF, dlogVO, dVO2):
    if zA * zB < 0:   # compensation
        return (f"{label}: adding the {psign} {partner} to donor-doped Ga2O3 pulls E_F DOWN by "
                f"{abs(dEF):.2f} eV and RAISES [V_O] by {dlogVO:+.2f} dex (V_O^2+ {dVO2:+.2f} dex) "
                f"via self-compensation; the n->compensated crossover is near a {partner}:donor ratio of "
                f"{ratio:.2f}. FALSIFIED if measured [V_O] DROPS or carrier type stays n-type past that ratio.")
    if zA > 0 and zB > 0:
        return (f"{label}: two donors keep E_F near CBM; V_O^2+ stays SUPPRESSED (Δ={dVO2:+.2f} dex). "
                f"FALSIFIED if adding the second donor raises [V_O] strongly.")
    if zA < 0 and zB < 0:
        return (f"{label}: two deep acceptors drive E_F toward VBM (Δ={dEF:.2f} eV); [V_O] RISES "
                f"({dlogVO:+.2f} dex) as the compensating-donor V_O forms. FALSIFIED if E_F/[V_O] move oppositely.")
    return (f"{label}: the isovalent partner is charge-neutral; the donor alone sets E_F and [V_O] "
            f"(Δ[V_O]={dlogVO:+.2f} dex is small). FALSIFIED if the isovalent dopant shifts [V_O] strongly.")


def make_figs(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return f"matplotlib unavailable: {e}"
    figdir = PROJ / "results/phase62_codoping/figs"; figdir.mkdir(parents=True, exist_ok=True)
    made = []
    # 1D compensation panels for the donor+acceptor pairs
    comp = [r for r in results if r["zA"] * r["zB"] < 0]
    if comp:
        fig, axes = plt.subplots(1, len(comp), figsize=(4 * len(comp), 3.4), squeeze=False)
        for ax, r in zip(axes[0], comp):
            c = np.array(r["c_sweep"]) * 100
            ax.plot(c, r["E_F"], "b-", label="E_F (eV)")
            ax.axhline(MIDGAP, ls=":", c="gray", lw=0.8); ax.set_xscale("log")
            ax.set_xlabel(f"{r['B'] if r['zB']<0 else r['A']} at%"); ax.set_ylabel("E_F (eV)", color="b")
            ax.set_ylim(0, EG); ax.set_title(r["label"], fontsize=10)
            ax2 = ax.twinx(); ax2.plot(c, r["log10_VO"], "r--", label="log[V_O]")
            ax2.set_ylabel("log10[V_O]", color="r")
        fig.suptitle("Co-doping compensation: fix donor 1%, add acceptor (solver, 973 K, 1 atm)", fontsize=11)
        fig.tight_layout(); p = figdir / "compensation_panels.png"; fig.savefig(p, dpi=130); plt.close(fig)
        made.append(str(p))
    # 2D Brouwer heatmap of E_F for the flagship Sn+Mg pair
    sm = next((r for r in results if r["label"] == "Sn+Mg"), None)
    if sm:
        gr = np.logspace(-4, math.log10(0.05), 36)
        CA, CB = np.meshgrid(gr, gr)
        res = _solve(carrier_sign("Sn"), CA.ravel(), elem_transition_level("Sn", carrier_sign("Sn")),
                     carrier_sign("Mg"), CB.ravel(), elem_transition_level("Mg", carrier_sign("Mg")))
        EFmap = res["E_F"].reshape(CA.shape)
        fig, ax = plt.subplots(figsize=(5, 4.2))
        im = ax.pcolormesh(CA * 100, CB * 100, EFmap, shading="auto", cmap="RdBu_r", vmin=0, vmax=EG)
        cs = ax.contour(CA * 100, CB * 100, EFmap, levels=[MIDGAP], colors="k", linewidths=1.5)
        ax.clabel(cs, fmt="E_F=mid-gap")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Sn at% (donor)"); ax.set_ylabel("Mg at% (acceptor)")
        ax.set_title("Sn+Mg Brouwer map: Fermi level E_F (eV)")
        fig.colorbar(im, label="E_F (eV)"); fig.tight_layout()
        p = figdir / "brouwer_SnMg_EF.png"; fig.savefig(p, dpi=130); plt.close(fig)
        made.append(str(p))
    return made


def main():
    results = [analyze_pair(*p) for p in PAIRS]
    figs = make_figs(results)
    rep = dict(T_K=973.15, log_pO2=0.0, native_VO_dHf=DHF, E_g=EG, midgap=MIDGAP,
               pairs=results, figures=figs)
    out = PROJ / "results/phase62_codoping/codoping_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print("=== co-doping mechanism predictions (973 K, 1 atm) ===\n")
    for r in results:
        print(f"[{r['regime']}] {r['label']}")
        print(f"   {r['falsifiable']}")
        if r["compensation_ratio"] is not None:
            print(f"   compensation ratio (partner:donor) ~ {r['compensation_ratio']:.2f}")
        print()
    print(f"figures: {figs}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
