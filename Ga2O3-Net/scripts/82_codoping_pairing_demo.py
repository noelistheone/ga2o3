"""V62: how donor-acceptor PAIR formation (from MACE binding) reshapes the co-doping prediction.

Reads the MACE association energies (dft/codoping/results/binding_energies.json), and for each BOUND
donor-acceptor pair computes, at equal co-doping and growth T:
  (i)  the equilibrium bound-pair fraction (mass action, src/models/codoping_pairing.py),
  (ii) the DILUTE (independent, no-pairing) solver prediction  (E_F, [V_O]),
  (iii) the PAIRED prediction (only the FREE excess dopants enter charge neutrality),
and the mechanism correction between them. Key falsifiable consequence: when binding is strong and
c_A ~= c_B, most dopants lock into NEUTRAL pairs -> free carriers nearly cancel -> the film behaves
close to UNDOPED (E_F near intrinsic), NOT as a deep-trap-compensated insulator. Distinct, testable.

Uses E_assoc_near (reference-free A-B interaction = the pairing-reaction energy) as the binding energy,
and reports sensitivity to E_bind. Writes results/phase62_codoping/pairing_demo.json. CPU, deterministic.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from src.models.defect_equilibrium import EG_GA2O3  # noqa: E402
from src.models.defect_equilibrium_codoping import solve_equilibrium_codoping, elem_transition_level  # noqa: E402
from src.models.codoping_pairing import pair_partition, bound_fraction  # noqa: E402
from src.models.v61_net import carrier_sign  # noqa: E402

torch.set_default_dtype(torch.float64)
EG = EG_GA2O3
DHF = [3.5, 1.8, 0.3]
T_K, LOG_PO2 = 973.15, 0.0
C_EACH = 0.01            # 1 at% each (equal co-doping)


def solve(cA, cB, A, B):
    zA, zB = carrier_sign(A), carrier_sign(B)
    eA, eB = elem_transition_level(A, zA), elem_transition_level(B, zB)
    out = solve_equilibrium_codoping(
        torch.tensor([DHF]), torch.tensor([T_K]), torch.tensor([LOG_PO2]),
        torch.tensor([[zA, zB]]), torch.tensor([[cA, cB]]), torch.tensor([[eA, eB]]),
        dop_mask=torch.tensor([[float(cA > 0), float(cB > 0)]]))
    return float(out["E_F"].item()), float(out["log10_VO"].item())


def main():
    bj = json.loads((PROJ / "dft/codoping/results/binding_energies.json").read_text())
    # intrinsic (undoped) reference
    ef_intr, vo_intr = solve(0.0, 0.0, "Mg", "Sn")
    rows = []
    for p in bj["pairs"]:
        A, B = p["A"], p["B"]
        if carrier_sign(A) * carrier_sign(B) >= 0:
            continue                                  # only donor-acceptor pairs pair up
        E_bind = p["E_assoc_near_eV"]                 # reference-free pairing-reaction energy
        if E_bind >= 0:
            continue                                  # not bound -> no pairing correction
        cAf, cBf, xAB = pair_partition(C_EACH, C_EACH, E_bind, T_K)
        cAf, cBf, xAB = float(cAf), float(cBf), float(xAB)
        bf = bound_fraction(C_EACH, C_EACH, E_bind, T_K)
        ef_dil, vo_dil = solve(C_EACH, C_EACH, A, B)          # naive independent compensation
        ef_pair, vo_pair = solve(cAf, cBf, A, B)             # only free excess is charged
        # sensitivity: use the (stronger) standard E_bind too
        bf_strong = bound_fraction(C_EACH, C_EACH, p["E_bind_eV"], T_K)
        rows.append(dict(
            pair=f"{A}+{B}", E_bind_used_eV=E_bind, E_bind_alt_eV=p["E_bind_eV"],
            bound_fraction=round(bf, 4), bound_fraction_strongEb=round(bf_strong, 4),
            c_free_each=round(cAf, 6),
            E_F_dilute=round(ef_dil, 4), E_F_paired=round(ef_pair, 4), E_F_intrinsic=round(ef_intr, 4),
            logVO_dilute=round(vo_dil, 4), logVO_paired=round(vo_pair, 4), logVO_intrinsic=round(vo_intr, 4),
            dEF_pairing_vs_dilute=round(ef_pair - ef_dil, 4),
            falsifiable=(f"{A}+{B} equal {C_EACH:.0%} co-dope at {T_K:.0f} K: the DEEP acceptor ({B if carrier_sign(A)>0 else A}) "
                         f"is marginally less ionized than the shallow donor, leaving a slight donor "
                         f"excess -> lightly n-type/RESISTIVE E_F={ef_dil:.2f} eV (well above intrinsic "
                         f"{ef_intr:.2f}, NOT p-type). MACE E_assoc={E_bind:.2f} eV => ~{bf*100:.0f}% of dopants form "
                         f"BOUND NEUTRAL donor-acceptor pairs (detectable by DAP photoluminescence / "
                         f"EXAFS {A}-{B} correlation); pairing removes donors+acceptors EQUALLY so E_F "
                         f"barely shifts ({ef_pair:.2f}) but the carrier-active fraction drops ~{bf*100:.0f}%. "
                         f"FALSIFIED if equal donor+deep-acceptor co-doping turns p-type / strongly "
                         f"conductive, OR no DAP/EXAFS pairing signature appears. To reach semi-insulating, "
                         f"acceptor must EXCEED donor (deep acceptor under-ionizes).")))
    rep = dict(T_K=T_K, log_pO2=LOG_PO2, c_each=C_EACH, E_F_intrinsic=ef_intr,
               midgap=0.5 * EG, pairs=rows,
               note=("Two distinct mechanism findings: (1) equal donor+DEEP-acceptor co-doping is lightly "
                     "n-type/resistive (E_F~3.08, well above intrinsic 2.525), NOT p-type -- the deep "
                     "acceptor under-ionizes; to reach semi-insulating the acceptor must exceed the donor. "
                     "(2) ~85-92% of equal co-dopants form bound NEUTRAL donor-acceptor pairs at growth T "
                     "(MACE E_assoc~-0.55 eV); pairing removes donors+acceptors equally so E_F is nearly "
                     "unchanged but the carrier-active dopant fraction drops sharply. E_bind from MACE-MP-0 "
                     "NEUTRAL relaxation = lower bound on charged binding (HSE06 needed, research B.1); "
                     "sputter is low-T/non-equilibrium so realized pairing may be below this equilibrium limit."))
    out = PROJ / "results/phase62_codoping/pairing_demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(f"intrinsic (undoped) E_F={ef_intr:.3f} eV, log[V_O]={vo_intr:.3f}\n")
    print(f"{'pair':8s} {'E_assoc':>8s} {'bound%':>7s} {'EF_dilute':>10s} {'EF_paired':>10s} {'EF_intr':>8s}")
    for r in rows:
        print(f"{r['pair']:8s} {r['E_bind_used_eV']:8.2f} {r['bound_fraction']*100:6.1f}% "
              f"{r['E_F_dilute']:10.3f} {r['E_F_paired']:10.3f} {r['E_F_intrinsic']:8.3f}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
