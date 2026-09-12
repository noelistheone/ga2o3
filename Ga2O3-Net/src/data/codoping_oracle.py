"""Phase 63 — solver-oracle data generator for the doping-type-general world model.

The V63 self-consistent complex solver (src/models/defect_equilibrium_complex.py) is a FREE oracle: it
maps ANY doping recipe (single / co-doping / multi-dopant / element+compound mixtures -- all reduce to a
cation multiset, see docs/phase63_design.md sec 1) to the physical state (E_F, log10[V_O], bound DAP
fraction). This module samples recipes over arbitrary doping TYPES and labels them with the oracle, so a
learned world model can be distilled from unlimited synthetic data (Method-of-Manufactured-Learning /
NeuralSCF distillation; research Angle-1). It also featurizes each recipe into a permutation-invariant
SET of physical-descriptor dopant tokens (DopNet-style host/dopant decoupling; research Angle-4 sec 4.2/4.6
-- physical encodings, NOT one-hot).

HONEST scope (research-validated): the oracle is uncorrelated with cross-paper ABSOLUTE experimental [V_O]
(Phase 61, Pearson ~0). So this trains/tests LAW learning (relative trends, signs, mechanism transfer
across doping types), NOT experimental magnitude accuracy. Zero external API; deterministic given a seed
tensor (we pass an explicit RNG; Math.random/Date.now-free per workspace rules is N/A here -- this is a
torch script, seeded via torch.Generator).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch

from src.data.element_descriptors import ELEMENT_DESCRIPTORS, DESCRIPTOR_DIM
from src.models.v61_net import carrier_sign
from src.models.defect_equilibrium_codoping import elem_transition_level
from src.models.defect_equilibrium_complex import solve_equilibrium_complex
from src.models.defect_equilibrium import EG_GA2O3

PROJ = Path(__file__).resolve().parents[2]

# substitutional-cation dopant vocab with carrier_sign() signs (anion-site N/F + interstitial H excluded
# so the substitutional charge-neutrality law is clean). All present in ELEMENT_DESCRIPTORS.
DONORS = ["Sn", "Si", "Ge", "Ti", "Zr", "Ta", "W", "V"]
ACCEPTORS = ["Mg", "Zn", "Cu"]
ISOVALENT = ["Al", "Fe", "B", "In", "Cr"]
VOCAB = DONORS + ACCEPTORS + ISOVALENT
D_MAX = 4
TOKEN_DIM = DESCRIPTOR_DIM + 3   # 11 descriptor + [z_sign, log10c, transition_level]

# short-range (E_assoc_near) binding for donor-acceptor pairs; measured pairs from MACE, else class default.
_MEASURED_BIND = None


def _load_measured_binding():
    global _MEASURED_BIND
    if _MEASURED_BIND is not None:
        return _MEASURED_BIND
    _MEASURED_BIND = {}
    p = PROJ / "dft/codoping/results/binding_energies.json"
    if p.exists():
        bj = json.load(open(p))
        for pr in bj.get("pairs", []):
            if str(pr.get("verdict", "")).startswith("BOUND"):
                key = frozenset([pr["A"], pr["B"]])
                _MEASURED_BIND[key] = float(pr["E_assoc_near_eV"])   # short-range part (Opt-B decomposition)
    return _MEASURED_BIND


def _short_range_binding(elem_a: str, elem_b: str) -> float:
    """E_assoc_near short-range binding [eV] for a donor-acceptor pair (negative=bound); default -0.55."""
    m = _load_measured_binding()
    return m.get(frozenset([elem_a, elem_b]), -0.55)


def _desc(elem: str) -> list[float]:
    return list(ELEMENT_DESCRIPTORS.get(elem, ELEMENT_DESCRIPTORS["other"]))


def sample_recipes(n: int, gen: torch.Generator, d_choices=(1, 2, 3, 4),
                   c_lo: float = 1e-4, c_hi: float = 5e-2,
                   T_lo: float = 600.0, T_hi: float = 1300.0,
                   lpo2_lo: float = -8.0, lpo2_hi: float = 0.0,
                   force_da: float = 0.0) -> dict:
    """Sample n recipes over the given cardinalities. Returns padded arrays [n,D_MAX,...] + masks + knobs.

    Each recipe: D distinct elements (from VOCAB), log-uniform concentrations, uniform T and log pO2.
    Token features (physical descriptors + z + log10c + level), the binding matrix (short-range, donor-
    acceptor only), and bookkeeping (elem indices, D) are returned. Deterministic given `gen`.

    force_da: fraction of samples forced (when D>=2) to contain >=1 donor AND >=1 acceptor, so the bound
    DAP-pairing channel has training signal (natural sampling is mostly bound_frac~0). The TEST
    distribution stays natural (force_da=0) for honesty.
    """
    don_idx = [VOCAB.index(e) for e in DONORS]
    acc_idx = [VOCAB.index(e) for e in ACCEPTORS]

    def _pick(D, want_da):
        if want_da and D >= 2:
            d0 = don_idx[int(torch.randint(0, len(don_idx), (1,), generator=gen))]
            a0 = acc_idx[int(torch.randint(0, len(acc_idx), (1,), generator=gen))]
            rest = [j for j in range(len(VOCAB)) if j not in (d0, a0)]
            extra = [rest[k] for k in torch.randperm(len(rest), generator=gen)[:D - 2].tolist()]
            picks = [d0, a0] + extra
            return [picks[k] for k in torch.randperm(len(picks), generator=gen).tolist()]
        return torch.randperm(len(VOCAB), generator=gen)[:D].tolist()
    rint = lambda hi, k=(): torch.randint(0, hi, k, generator=gen).tolist() if k else \
        int(torch.randint(0, hi, (1,), generator=gen).item())
    tokens = torch.zeros(n, D_MAX, TOKEN_DIM)
    mask = torch.zeros(n, D_MAX)
    z = torch.zeros(n, D_MAX)
    c = torch.zeros(n, D_MAX)
    level = torch.zeros(n, D_MAX)
    elem_ids = -torch.ones(n, D_MAX, dtype=torch.long)
    bind = torch.zeros(n, D_MAX, D_MAX)
    D_arr = torch.zeros(n, dtype=torch.long)
    nV = len(VOCAB)
    T = T_lo + (T_hi - T_lo) * torch.rand(n, generator=gen)
    lpo2 = lpo2_lo + (lpo2_hi - lpo2_lo) * torch.rand(n, generator=gen)
    for i in range(n):
        D = int(d_choices[rint(len(d_choices))])
        D_arr[i] = D
        want_da = bool(force_da > 0 and float(torch.rand(1, generator=gen)) < force_da)
        perm = _pick(D, want_da)
        elems = [VOCAB[j] for j in perm]
        logc = c_lo and (np.log10(c_lo) + (np.log10(c_hi) - np.log10(c_lo)) *
                         torch.rand(D, generator=gen).numpy())
        for k, e in enumerate(elems):
            zk = carrier_sign(e)
            ck = float(10.0 ** logc[k])
            lv = elem_transition_level(e, zk, EG_GA2O3)
            z[i, k] = zk; c[i, k] = ck; level[i, k] = lv; mask[i, k] = 1.0; elem_ids[i, k] = perm[k]
            tokens[i, k] = torch.tensor(_desc(e) + [float(zk), float(np.log10(ck)), float(lv)])
        for a in range(D):
            for b in range(a + 1, D):
                if z[i, a] * z[i, b] < 0:               # donor-acceptor -> short-range binding
                    eb = _short_range_binding(elems[a], elems[b])
                    bind[i, a, b] = bind[i, b, a] = eb
    return dict(tokens=tokens, mask=mask, z=z, c=c, level=level, elem_ids=elem_ids,
                bind=bind, T=T, lpo2=lpo2, D=D_arr)


def build_batch(recipes: list, T_list, lpo2_list) -> dict:
    """Build padded tensors for an explicit list of recipes (for law-test counterfactuals).

    recipes: list of recipes, each a list of (elem:str, conc:float). T_list/lpo2_list: per-recipe scalars.
    Returns the same dict format as sample_recipes (tokens/mask/z/c/level/bind/T/lpo2/D), so both the
    surrogate and the V63 solver consume IDENTICAL configs. Handles any cardinality up to D_MAX.
    """
    n = len(recipes)
    tokens = torch.zeros(n, D_MAX, TOKEN_DIM); mask = torch.zeros(n, D_MAX)
    z = torch.zeros(n, D_MAX); c = torch.zeros(n, D_MAX); level = torch.zeros(n, D_MAX)
    bind = torch.zeros(n, D_MAX, D_MAX); D_arr = torch.zeros(n, dtype=torch.long)
    for i, rc in enumerate(recipes):
        D_arr[i] = len(rc)
        elems = [e for e, _ in rc]
        for k, (e, ck) in enumerate(rc):
            zk = carrier_sign(e); lv = elem_transition_level(e, zk, EG_GA2O3)
            ck = float(max(ck, 1e-7))
            z[i, k] = zk; c[i, k] = ck; level[i, k] = lv; mask[i, k] = 1.0
            tokens[i, k] = torch.tensor(_desc(e) + [float(zk), float(np.log10(ck)), float(lv)])
        for a in range(len(rc)):
            for b in range(a + 1, len(rc)):
                if z[i, a] * z[i, b] < 0:
                    eb = _short_range_binding(elems[a], elems[b])
                    bind[i, a, b] = bind[i, b, a] = eb
    return dict(tokens=tokens, mask=mask, z=z, c=c, level=level, bind=bind,
                T=torch.tensor(T_list, dtype=torch.float32), lpo2=torch.tensor(lpo2_list, dtype=torch.float32),
                D=D_arr)


def label_with_oracle(rec: dict, device: str = "cpu", chunk: int = 4000,
                      use_coulomb: bool = True, n_outer: int = 30) -> dict:
    """Label sampled recipes with the V63 self-consistent complex solver. Returns E_F, log10_VO, bound_frac."""
    n = rec["tokens"].shape[0]
    dHf_base = torch.tensor([3.5, 1.8, 0.3], dtype=torch.float64, device=device)
    EF = torch.zeros(n); VO = torch.zeros(n); BF = torch.zeros(n)
    with torch.no_grad():
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            B = e - s
            dHf = dHf_base.view(1, 3).expand(B, 3).contiguous()
            out = solve_equilibrium_complex(
                dHf, rec["T"][s:e].to(device), rec["lpo2"][s:e].to(device),
                rec["z"][s:e].to(device), rec["c"][s:e].to(device), rec["level"][s:e].to(device),
                E_bind_neutral=rec["bind"][s:e].to(device), dop_mask=rec["mask"][s:e].to(device),
                use_coulomb=use_coulomb, n_outer=n_outer)
            EF[s:e] = out["E_F"].cpu().float()
            VO[s:e] = out["log10_VO"].cpu().float()
            BF[s:e] = out["bound_frac"].cpu().float()
    return dict(E_F=EF, log10_VO=VO, bound_frac=BF)
