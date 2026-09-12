"""Phase 46 V11 pre-train sanity check.

Verifies that the 5-dim element_descriptors actually separate super-donors
(Sb/Ta/Bi) from regular donors (Si/Sn/Ge) and acceptors (Mg/Zn) in
descriptor space. If the geometry isn't there, the ElementHyperNet has no
way to fix Tier-1C FLIPS for Bi/Sb regardless of how it's trained.

The hard assertion: Bi↔Ta descriptor distance must be smaller than Bi↔Mg.
If it isn't, the Δv channel is not actually separating super-donors from
acceptors and V11 should not be launched as-is.

Usage:  python scripts/sanity_check_hypernet_init.py
Exit codes:  0 = sanity pass; 1 = geometry insufficient.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.data.element_descriptors import ELEMENT_DESCRIPTORS, DESCRIPTOR_DIM
from src.models.element_hypernet import ElementHyperNet


def main() -> int:
    elements = list(ELEMENT_DESCRIPTORS.keys())
    desc = torch.tensor([ELEMENT_DESCRIPTORS[e] for e in elements], dtype=torch.float32)
    print(f"DESCRIPTOR_DIM = {DESCRIPTOR_DIM}; n_elements = {len(elements)}")
    print(f"Descriptor matrix: {tuple(desc.shape)}")
    print()

    name_to_idx = {e: i for i, e in enumerate(elements)}

    pairs_to_check = [
        ("Bi", "Ta", "small (super-donor pair)"),
        ("Bi", "Sb", "small (super-donor pair)"),
        ("Sb", "Ta", "small (super-donor pair)"),
        ("Bi", "Mg", "LARGE (super-donor ↔ acceptor)"),
        ("Sb", "Mg", "LARGE (super-donor ↔ acceptor)"),
        ("Bi", "Sn", "moderate (super ↔ regular donor)"),
        ("Sb", "Sn", "moderate (super ↔ regular donor)"),
        ("Ge", "Sn", "small (regular-donor pair)"),
        ("Ge", "Mg", "LARGE (donor ↔ acceptor)"),
        ("Ge", "Si", "small (regular-donor pair)"),
    ]
    print(f"{'pair':<14s} {'L2 dist':>10s}   expectation")
    for a, b, exp in pairs_to_check:
        d = (desc[name_to_idx[a]] - desc[name_to_idx[b]]).norm().item()
        print(f"{a}-{b:<10s} {d:>10.3f}   {exp}")
    print()

    H = ElementHyperNet(desc_dim=DESCRIPTOR_DIM, hidden=16, mid=8)
    # Override zero-init final layer with kaiming so output has actual variance
    # for inspection. (Training will use the original zero-init.)
    for m in H.net:
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    descriptors_per_elem = desc.unsqueeze(1)              # [n_elem, 1, DIM]
    weights_per_elem = torch.ones(len(elements), 1)       # [n_elem, 1]
    with torch.no_grad():
        offsets = H(descriptors_per_elem, weights_per_elem).squeeze(-1)

    print(f"{'elem':<8s} {'HN offset (kaiming-random)':>30s}")
    for e, o in zip(elements, offsets.tolist()):
        if e in ("Sb", "Ta", "Bi"):
            tag = "  <-- super-donor"
        elif e in ("Si", "Sn", "Ge", "Ti"):
            tag = "  <-- regular donor"
        elif e in ("Mg", "Zn", "Cu"):
            tag = "  <-- acceptor"
        else:
            tag = ""
        print(f"{e:<8s} {o:>+30.4f}{tag}")
    print()

    # Hard assertions on geometry
    super_super = (desc[name_to_idx["Bi"]] - desc[name_to_idx["Ta"]]).norm().item()
    super_acc = (desc[name_to_idx["Bi"]] - desc[name_to_idx["Mg"]]).norm().item()
    sb_sn = (desc[name_to_idx["Sb"]] - desc[name_to_idx["Sn"]]).norm().item()
    sb_mg = (desc[name_to_idx["Sb"]] - desc[name_to_idx["Mg"]]).norm().item()

    failures = []
    if super_super >= super_acc:
        failures.append(
            f"Bi↔Ta ({super_super:.3f}) ≥ Bi↔Mg ({super_acc:.3f}) — "
            f"super-donor pair is not closer than super↔acceptor."
        )
    if sb_sn >= sb_mg:
        failures.append(
            f"Sb↔Sn ({sb_sn:.3f}) ≥ Sb↔Mg ({sb_mg:.3f}) — "
            f"super↔regular-donor not closer than super↔acceptor."
        )

    if failures:
        print("SANITY FAILED:")
        for f in failures:
            print(f"  - {f}")
        print()
        print("V11 should not be launched as-is. Revisit Δv encoding in "
              "src/data/element_descriptors.py before retraining.")
        return 1

    print("SANITY OK: descriptor geometry separates super-donors from acceptors.")
    print(f"  Bi↔Ta = {super_super:.3f} < Bi↔Mg = {super_acc:.3f}")
    print(f"  Sb↔Sn = {sb_sn:.3f} < Sb↔Mg = {sb_mg:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
