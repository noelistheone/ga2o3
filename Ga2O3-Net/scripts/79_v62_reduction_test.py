"""V62 model-level regression test: V62CoDopingNet at D=1 (V61-default transition levels) == V61Net.

Proves the co-doping model is a STRICT generalization of V61 (not a rewrite). Copies a V61Net state_dict
into V62CoDopingNet (identical parameter names) and compares every output head on random single-dopant
inputs. Any divergence > 1e-5 fails. Writes results/phase62_codoping/reduction_test.json. CPU, deterministic.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from src.models.v61_net import V61Net  # noqa: E402
from src.models.v62_codoping_net import V62CoDopingNet  # noqa: E402
from src.models.defect_equilibrium import EG_GA2O3  # noqa: E402

EG = EG_GA2O3


def main():
    torch.manual_seed(0)
    B, F = 32, 71
    v61 = V61Net(feat_dim=F).eval()
    v62 = V62CoDopingNet(feat_dim=F).eval()
    # randomize v61 so the test is non-trivial, then copy identical-named params into v62
    for p in v61.parameters():
        p.data = torch.randn_like(p) * 0.3
    missing, unexpected = v62.load_state_dict(v61.state_dict(), strict=False)

    feats = torch.randn(B, F)
    T_K = torch.empty(B).uniform_(400, 1300)
    log_pO2 = torch.empty(B).uniform_(-6, 0)
    z = torch.randint(-1, 2, (B,)).float()                 # {-1,0,+1}
    c = torch.empty(B).uniform_(1e-4, 3e-2)
    elem_idx = torch.randint(0, 20, (B,))
    # V61-default transition levels by sign (donor E_g-0.05; acceptor 1.3; isovalent irrelevant)
    e_level = torch.where(z > 0, torch.full_like(z, EG - 0.05),
                          torch.where(z < 0, torch.full_like(z, 1.3), torch.full_like(z, 0.5 * EG)))

    with torch.no_grad():
        o61 = v61(feats, T_K, log_pO2, z, c, elem_idx)
        o62 = v62(feats, T_K, log_pO2, z.view(-1, 1), c.view(-1, 1), elem_idx.view(-1, 1),
                  e_level.view(-1, 1))
    diffs = {k: float((o61[k] - o62[k]).abs().max().item())
             for k in ("bulk_logVO", "E_F", "r", "log_sigma")}
    diffs["dHf"] = float((o61["dHf"] - o62["dHf"]).abs().max().item())
    rep = dict(max_abs_diff=diffs, all_match=bool(max(diffs.values()) < 1e-5),
               state_dict_missing=list(missing), state_dict_unexpected=list(unexpected))
    out = PROJ / "results/phase62_codoping/reduction_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
