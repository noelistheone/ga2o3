"""Gate-2: O2 molecule PBE reference (for μ_O) + raw V_O formation-energy assembly from the
completed perfect-cell + V_O calcs. Writes the O2 input; after O2 runs, formation energies follow.
"""
import re
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
CC = PROJ / "dft/qe_cc"
PSEUDO = PROJ.parent / "Ga2O3-Net/dft/qe_hse06/pseudos"

# O2 molecule in a 12 Å cubic box, triplet (nspin=2), gamma point, PBE
O2_IN = f"""\
&CONTROL
    calculation = 'scf'
    restart_mode = 'from_scratch'
    prefix = 'gate2_O2'
    pseudo_dir = '{PSEUDO}'
    outdir = '{CC / 'outdir' / 'gate2_O2'}'
    tprnfor = .true.
    verbosity = 'low'
/
&SYSTEM
    ibrav = 1
    celldm(1) = 22.68
    nat = 2
    ntyp = 1
    ecutwfc = 80.0
    ecutrho = 480.0
    occupations = 'smearing'
    smearing = 'gaussian'
    degauss = 0.005
    nspin = 2
    starting_magnetization(1) = 1.0
    tot_magnetization = 2.0
/
&ELECTRONS
    electron_maxstep = 200
    conv_thr = 1.0d-7
    mixing_beta = 0.3
    diagonalization = 'david'
/
ATOMIC_SPECIES
  O     15.9994  O.upf
ATOMIC_POSITIONS angstrom
  O     6.00000000  6.00000000  6.00000000
  O     6.00000000  6.00000000  7.21000000
K_POINTS gamma
"""
(CC / "gate2_O2.in").write_text(O2_IN)
print("wrote gate2_O2.in (O2 triplet, 12 A box, PBE)")

# raw V_O energetics from completed calcs
RY = 13.605693


def energy(out):
    p = CC / out
    if not p.exists() or "JOB DONE" not in p.read_text():
        return None
    e = re.findall(r"^!\s+total energy\s+=\s+([-\d.]+)", p.read_text(), re.M)
    return float(e[-1]) * RY if e else None


E_perfect = energy("gate2_perfect.out")
E_VO_q0 = energy("cc_VO_q0_relax.out")
E_VO_q2 = energy("cc_VO_q2_relax.out")
print(f"\nE(perfect 80-atom) = {E_perfect} eV")
print(f"E(V_O q0)          = {E_VO_q0} eV")
print(f"E(V_O q2)          = {E_VO_q2} eV")
if E_perfect and E_VO_q0:
    # raw E_f at mu_O=0 (O-rich elemental ref, before adding 1/2 E(O2)); no charge corr for q0
    raw_q0 = E_VO_q0 - E_perfect
    print(f"\nRaw E(V_O,q0) − E(perfect) = {raw_q0:.3f} eV  (add μ_O = ½E(O2) for formation energy;")
    print("  KROGER HSE V_O q0 E_f at μ=0 ≈ 4.5-5.2 eV — PBE value will differ; validates the pipeline)")
