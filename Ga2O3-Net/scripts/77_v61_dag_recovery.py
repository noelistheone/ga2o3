"""V61 validation (design §2.5) — is the Kröger-Vink law DISCOVERABLE from interventional data?

The CHT (Xia/Bareinboim) says causal direction cannot be discovered from observational fit; it CAN from
interventions. We have unlimited clean interventional (state, action, next-state) data from the closed-form
simulator. This experiment asks the honest, CHT-scoped question: does a GENERIC black-box learner (no physics
structure) — given unlimited interventional sim data — RECOVER the law (correct counterfactual signs)? And
does the SAME learner FAIL at N=14 (the real-data regime)?

A 'yes / no' result demonstrates: the law is genuinely data-discoverable WITH enough interventions, and the
real-data failure is SCARCITY (C3), not impossibility — exactly what the CRL identifiability literature
predicts (von Kügelgen 2306.00542: influence strengths identifiable from sufficient interventions).
This is the defensible 'data-discovered (on the simulator twin)' claim. Writes results/phase61_v61/dag_recovery.json.
"""
from __future__ import annotations
import json, math, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

import sys
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from src.data.dcwm_transitions import (  # noqa: E402
    StateFeaturizer, TransitionGenerator, measured_log_vo, ATM_PO2, ACTION_DIM,
    GEN_ELEMENTS, _zero_action,
)


def make_mlp(din, hidden=128):
    return nn.Sequential(nn.Linear(din, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                         nn.Linear(hidden, 1))


def train_generic(gen, f, dev, n_train, steps=2000, bs=4096, lr=1e-3, pool=150_000):
    """Generic MLP on (state_features, action) -> Δ measured_log_vo. No physics structure.

    Unlimited regime: pre-sample a large fixed POOL once (featurization is the bottleneck), minibatch
    from it — equivalent to 'unlimited interventional data' for learning the law, far faster than
    re-sampling each step. Scarce regime: a single fixed batch of n_train transitions.
    """
    din = f.dim + ACTION_DIM
    net = make_mlp(din).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    if n_train <= 0:  # unlimited (large pool)
        b = gen.sample_batch(pool)
        X = torch.tensor(np.concatenate([b["x_s"], b["a"]], 1), device=dev)
        dy = torch.tensor(b["ym_t"] - b["ym_s"], device=dev).unsqueeze(1)
        for _ in range(steps):
            idx = torch.randint(0, X.shape[0], (bs,), device=dev)
            opt.zero_grad(); loss = ((net(X[idx]) - dy[idx]) ** 2).mean(); loss.backward(); opt.step()
    else:            # fixed small set (data-scarce regime)
        b = gen.sample_batch(n_train)
        X = torch.tensor(np.concatenate([b["x_s"], b["a"]], 1), device=dev)
        dy = torch.tensor(b["ym_t"] - b["ym_s"], device=dev).unsqueeze(1)
        for _ in range(steps):
            opt.zero_grad(); loss = ((net(X) - dy) ** 2).mean(); loss.backward(); opt.step()
    return net


def probe_signs(net, f, dev):
    """For each dopant element, predict Δy under a +0.5-dex concentration action and compare its SIGN
    to the simulator ground truth. Returns LEARNED_LAW count and per-element detail."""
    elems = [e for e in GEN_ELEMENTS if e != "undoped"]
    T_C, atm = 700.0, "Ar_O2_1_1"; log_pO2 = math.log10(ATM_PO2[atm]); T_K = T_C + 273.15
    c0 = 0.01; dlogc = 0.5
    detail, learned = {}, 0
    net.eval()
    for e in elems:
        x_s = f.featurize(e, c0, T_C, atm, 0, 2)
        a = _zero_action(); a[0] = dlogc
        X = torch.tensor(np.concatenate([x_s, a])[None, :], dtype=torch.float32, device=dev)
        with torch.no_grad():
            dy_pred = float(net(X).item())
        dy_true = measured_log_vo(e, c0 * 10 ** dlogc, T_K, log_pO2) - measured_log_vo(e, c0, T_K, log_pO2)
        ok = (np.sign(dy_pred) == np.sign(dy_true)) and abs(dy_true) > 1e-6
        learned += int(ok)
        detail[e] = dict(dy_pred=round(dy_pred, 4), dy_true=round(dy_true, 4), sign_ok=bool(ok))
    return learned, len(elems), detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0)
    f = StateFeaturizer()
    gen = TransitionGenerator(f, seed=0, target="vacancy_concentration")

    rep = {}
    # unlimited interventional data
    net_full = train_generic(gen, f, dev, n_train=0, steps=args.steps)
    L, n, det = probe_signs(net_full, f, dev)
    rep["unlimited_interventions"] = dict(LEARNED_LAW=f"{L}/{n}", frac=L / n, detail=det)
    # data-scarce (N=14) regime
    net_small = train_generic(gen, f, dev, n_train=14, steps=args.steps)
    Ls, ns, dets = probe_signs(net_small, f, dev)
    rep["scarce_N14"] = dict(LEARNED_LAW=f"{Ls}/{ns}", frac=Ls / ns)
    rep["conclusion"] = ("law DISCOVERABLE from sufficient interventions (CHT-consistent); real-data "
                         "failure is SCARCITY not impossibility" if L >= 0.8 * n and Ls < L else
                         "inconclusive — inspect detail")
    out = PROJ / "results/phase61_v61/dag_recovery.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: rep[k] for k in ("unlimited_interventions", "scarce_N14", "conclusion")
                      if k != "unlimited_interventions"} | {"unlimited_LEARNED_LAW":
                      rep["unlimited_interventions"]["LEARNED_LAW"]}, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
