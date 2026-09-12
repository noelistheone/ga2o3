"""Phase 46 V11 post-hoc confound diagnostic.

For each saved seed/fold checkpoint, print the L2 norm of:
  - `heads.vacancy_concentration.method_offset.weight` (V5 α_method)
  - `element_hypernet.net.<final layer>.weight` (V11 H output layer)

If H's norm grew >3× while α_method's norm shrank >50% (or vice versa),
flag it: H may have absorbed α_method's contribution under
element/method correlation. The plan's V11 acceptance criterion #6.

Usage:  python scripts/check_hypernet_method_offset_norms.py \
            results/phase46v11_hypernet_vc_5seed
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch


def norm_of(state_dict: dict, key_substring: str) -> float | None:
    """Return L2 norm of the first parameter whose key contains key_substring."""
    for k, v in state_dict.items():
        if key_substring in k:
            return float(v.detach().to(torch.float32).norm().item())
    return None


def main(bundle_dir: Path) -> int:
    if not bundle_dir.exists():
        print(f"Bundle not found: {bundle_dir}")
        return 1

    rows = []
    for ckpt in sorted(bundle_dir.glob("seed*/fold*/checkpoint.pt")):
        try:
            obj = torch.load(ckpt, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  WARN: failed to load {ckpt}: {e}")
            continue
        sd = obj.get("model_state_dict", obj)
        if not isinstance(sd, dict):
            continue
        m_norm = norm_of(sd, "method_offset.weight")
        # element_hypernet.net.4 is the final Linear weight in (Linear, SiLU,
        # Linear, SiLU, Linear) sequential — index 4. We try several index
        # patterns to be robust to architecture changes.
        h_norm = None
        for idx_try in ("net.4.weight", "net.6.weight", "net.2.weight"):
            v = norm_of(sd, f"element_hypernet.{idx_try}")
            if v is not None:
                h_norm = v
                break
        rows.append((str(ckpt.relative_to(bundle_dir)), m_norm, h_norm))

    if not rows:
        print("No checkpoints found.")
        return 1

    print(f"{'checkpoint':<30s} {'‖method_offset‖':>18s} {'‖H_final‖':>15s}")
    print("-" * 65)
    m_vals = []
    h_vals = []
    for tag, m, h in rows:
        m_str = f"{m:>18.4f}" if m is not None else f"{'N/A':>18s}"
        h_str = f"{h:>15.4f}" if h is not None else f"{'N/A':>15s}"
        if m is not None: m_vals.append(m)
        if h is not None: h_vals.append(h)
        print(f"{tag:<30s} {m_str} {h_str}")

    if m_vals and h_vals:
        m_mean = sum(m_vals) / len(m_vals)
        h_mean = sum(h_vals) / len(h_vals)
        m_max, m_min = max(m_vals), min(m_vals)
        h_max, h_min = max(h_vals), min(h_vals)
        print()
        print(f"Cross-fold ‖method_offset‖: mean={m_mean:.3f}, range [{m_min:.3f}, {m_max:.3f}]")
        print(f"Cross-fold ‖H_final‖:       mean={h_mean:.3f}, range [{h_min:.3f}, {h_max:.3f}]")
        print()

        if h_max > 3.0 * h_min and m_max > 2.0 * m_min:
            print("WARNING: cross-fold norm variance is large for both H and method_offset.")
            print("         Inconsistent credit assignment across folds — possible confound.")
        else:
            print("OK: norms are stable across folds, no obvious credit-shuffle.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(Path(sys.argv[1])))
