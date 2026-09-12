"""Phase 58 — Generate PBE+U Quantum ESPRESSO inputs for V58 gap-dopant V_O calcs.

Closes the gap in `dft/qe_hse06/results/v_o_ef.csv` for Tier-1C unsupervised and
V55-Ext-supervised dopants. Replaces the broken `inputs_v58_extra/PBEU_*.in`
stubs (`U=5.0`, `ecutwfc=60`, empty ATOMIC_POSITIONS).

Locked convention (phase58_v58_dft_llm_hybrid_design.md §7 "实现决策" callout,
2026-05-27):
  - Pseudos: ONCV nc-sr-04 PBE (norm-conserving) — same family as the 87-calc
             HSE06 corpus already in `dft/qe_hse06/pseudos/`. NOT SSSP PAW.
  - Cutoffs: ecutwfc=80 Ry, ecutrho=480 Ry (4× — NC pseudos).
  - Hubbard U: Hubbard_U(Ga-3d) = 4.0 eV (Ga is species index 1).
  - Supercell: 80-atom (nat=79 with the substituted Ga site as the dopant);
               coordinates loaded from the MACE-MP-0 relaxed `_q0.json` cell.
  - nspin=2 for open-shell 3d/4d/5d transition metals: Ti, V, Cr, Fe, Ni, Cu, W.
  - tot_charge = q ∈ {0, +1, +2}. nelec is taken from the species count of
    valence electrons in the ONCV pseudos minus tot_charge — QE handles this
    automatically when tot_charge is set, so we do NOT write nelec explicitly.
  - K-points: 1×1×1 Γ-only (matches the 87 HSE06 calcs, which used the same
              80-atom supercell with nqx=1×1×1; Γ is adequate at this size).

Dopant scope (12 elements, all with pseudo + relaxed q0 JSON present):

  Zn  — V55-Ext supervised (V_O cache currently missing Zn)
  Cu  — Tier-1C unsupervised
  Cr  — Tier-1C (isovalent v=3)
  Fe  — Tier-1C-adjacent
  Ni  — Tier-1C-adjacent
  B   — Tier-1C unsupervised
  V   — Tier-1C unsupervised
  W   — Tier-1C unsupervised (super-donor)
  Ti  — V55-Ext-adjacent donor
  Al  — isovalent reference
  Er  — Tier-1C (lanthanide)
  Eu  — Tier-1C (lanthanide)

For each (dopant) × (O_I, O_II, O_III) × q ∈ {0, +1, +2} = up to 9 calcs.
Total: 12 × 9 = 108 calcs. Each ~1.5 h PBE+U on 16-core CPU (vs HSE06 3-5 days).

Blocked dopants (pseudo missing) — documented in V58_GAP_DFT_BLOCKED.md:
  F, H, La, In, Pd, Hf. F is the highest priority (Tier-1C +2.2 dex anomaly).

Outputs (writes-only — does NOT submit any QE job):
  dft/qe_hse06/inputs_v58_gap/PBEU_v58_{dopant}_{site}_q{0,1,2}.in
  dft/qe_hse06/inputs_v58_gap/manifest.json
  dft/qe_hse06/V58_GAP_DFT_BLOCKED.md
  dft/qe_hse06/run_v58_gap.sh

Usage:
  conda activate ga2o3
  python scripts/44_v58_gen_gap_qe_inputs.py
"""
from __future__ import annotations

import json
import logging
import stat
from pathlib import Path
from typing import Optional

from pymatgen.core import Element, Structure

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("v58_gap")

PROJ = Path(__file__).resolve().parents[1]
PSEUDO_DIR = PROJ / "dft" / "qe_hse06" / "pseudos"
RELAX_DIR = PROJ / "dft" / "screening" / "results" / "relaxed"
OUT_DIR = PROJ / "dft" / "qe_hse06" / "inputs_v58_gap"
LOG_DIR = PROJ / "dft" / "qe_hse06" / "logs"
RUN_SCRIPT = PROJ / "dft" / "qe_hse06" / "run_v58_gap.sh"
BLOCKED_DOC = PROJ / "dft" / "qe_hse06" / "V58_GAP_DFT_BLOCKED.md"

# --- Locked convention (§7 callout) ---------------------------------------
ECUTWFC = 80.0
ECUTRHO = 480.0
HUBBARD_U_GA = 4.0   # eV on Ga 3d (species index 1)
KMESH = (1, 1, 1)    # Γ-only, matches the 87 HSE06 calcs

# Open-shell transition metals → need spin polarization
TM_NSPIN2 = {"Ti", "V", "Cr", "Fe", "Ni", "Cu", "W"}

DOPANTS = ["Zn", "Cu", "Cr", "Fe", "Ni", "B", "V", "W", "Ti", "Al", "Er", "Eu"]
SITES = ["O_I", "O_II", "O_III"]
CHARGES = [0, 1, 2]

# Blocked dopants (pseudo missing) documented in V58_GAP_DFT_BLOCKED.md.
# F, H, La, In, Pd, Hf are blocked by spec; Er, Eu are added dynamically
# in main() when their .upf is not found in dft/qe_hse06/pseudos/.
BLOCKED_DOPANTS = {
    "F":  ("Tier-1C OFFSET case (+2.2 dex anomaly, doc §1.1) — HIGHEST PRIORITY",
           "https://www.pseudo-dojo.org/  (NC, scalar-relativistic PBE, standard, F.upf)"),
    "H":  ("Required for V_O–H complexes (doc §7.9 P1 list)",
           "https://www.pseudo-dojo.org/  (NC, scalar-relativistic PBE, standard, H.upf)"),
    "La": ("Tier-1C lanthanide reference (paired with Er, Eu)",
           "https://www.pseudo-dojo.org/  (NC, scalar-relativistic PBE, standard, La.upf — note: lanthanides may be in the 'stringent' set, check both)"),
    "In": ("Tier-1C-adjacent isovalent donor (In³⁺ vs Ga³⁺)",
           "https://www.pseudo-dojo.org/  (NC, scalar-relativistic PBE, standard, In.upf)"),
    "Pd": ("Tier-1C contact-metal reference (Schottky-barrier proxy)",
           "https://www.pseudo-dojo.org/  (NC, scalar-relativistic PBE, standard, Pd.upf)"),
    "Hf": ("V58 deferred from V57 Roadmap (doc §7.9 P1) — donor v=4",
           "https://www.pseudo-dojo.org/  (NC, scalar-relativistic PBE, standard, Hf.upf)"),
}

# Reason text for each dopant in DOPANTS — used to populate the blocked-doc
# entry if its pseudo turns out to be missing at runtime.
DOPANT_REASON = {
    "Zn": "V55-Ext supervised dopant (V_O cache currently missing Zn)",
    "Cu": "Tier-1C unsupervised",
    "Cr": "Tier-1C (isovalent v=3)",
    "Fe": "Tier-1C-adjacent",
    "Ni": "Tier-1C-adjacent",
    "B":  "Tier-1C unsupervised",
    "V":  "Tier-1C unsupervised",
    "W":  "Tier-1C unsupervised (super-donor)",
    "Ti": "V55-Ext-adjacent donor",
    "Al": "Isovalent reference",
    "Er": "Tier-1C (lanthanide, 4f¹¹ open-shell)",
    "Eu": "Tier-1C (lanthanide, 4f⁶ open-shell)",
}

# Per-calc wall-clock estimate (hours) — PBE+U on 80-atom β-Ga2O3 supercell, 16-core
PBEU_HOURS_PER_CALC = 1.5

# --- QE input template ----------------------------------------------------
QE_TEMPLATE = """\
&CONTROL
    calculation = 'scf'
    restart_mode = 'from_scratch'
    prefix = '{prefix}'
    pseudo_dir = '../pseudos'
    outdir = '../outdir/{prefix}'
    tprnfor = .true.
    tstress = .false.
    verbosity = 'low'
/
&SYSTEM
    ibrav = 0
    nat = {nat}
    ntyp = {ntyp}
    ecutwfc = {ecutwfc}
    ecutrho = {ecutrho}
    occupations = 'smearing'
    smearing = 'gaussian'
    degauss = 0.005
    nspin = {nspin}
{starting_mag_block}    tot_charge = {tot_charge}
    ! PBE+U on Ga 3d (locked convention: U=4.0 eV, design doc §7 callout)
    lda_plus_u = .true.
    Hubbard_U(1) = {hubbard_u}
/
&ELECTRONS
    electron_maxstep = 200
    conv_thr = 1.0d-7
    mixing_beta = 0.4
    mixing_mode = 'plain'
    diagonalization = 'david'
/
ATOMIC_SPECIES
{species_block}
ATOMIC_POSITIONS angstrom
{positions_block}
K_POINTS automatic
{kx} {ky} {kz} 0 0 0
CELL_PARAMETERS angstrom
{cell_block}
"""


def render_qe_input(struct: Structure, prefix: str, charge: int, dopant: str) -> str:
    """Build the QE pw.x input for one (struct, charge) pair using the locked convention."""

    # Species block: Ga first (species index 1, gets Hubbard U), O second, dopant last.
    # Sort with Ga first so Hubbard_U(1) targets Ga; pymatgen's struct.species may include
    # the dopant species — we order Ga → O → dopant alphabetical for stability.
    species_present = sorted({sp.symbol for sp in struct.species})
    ordered = ["Ga"]
    if "O" in species_present:
        ordered.append("O")
    for s in species_present:
        if s not in ("Ga", "O"):
            ordered.append(s)
    # sanity: every species must appear
    assert set(ordered) == set(species_present), (ordered, species_present)

    species_lines = []
    for s in ordered:
        mass = Element(s).atomic_mass.real  # strip the amu unit suffix
        species_lines.append(f"  {s:<3s}  {mass:8.4f}   {s}.upf")
    species_block = "\n".join(species_lines)

    positions_lines = []
    for site in struct:
        x, y, z = site.coords
        positions_lines.append(f"  {site.specie.symbol:<3s}  {x:14.8f}  {y:14.8f}  {z:14.8f}")
    positions_block = "\n".join(positions_lines)

    cell = struct.lattice.matrix
    cell_lines = []
    for row in cell:
        cell_lines.append(f"  {row[0]:14.8f}  {row[1]:14.8f}  {row[2]:14.8f}")
    cell_block = "\n".join(cell_lines)

    # Spin polarization
    nspin = 2 if dopant in TM_NSPIN2 else 1
    if nspin == 2:
        # Initialize the dopant species with a small magnetic guess so SCF can
        # converge to a non-trivial magnetic state. Ga and O get 0.
        # Species index follows `ordered` order: Ga=1, O=2, dopant=last.
        mag_lines = []
        for idx, s in enumerate(ordered, start=1):
            if s == dopant:
                mag_lines.append(f"    starting_magnetization({idx}) = 0.3")
            else:
                mag_lines.append(f"    starting_magnetization({idx}) = 0.0")
        starting_mag_block = "\n".join(mag_lines) + "\n"
    else:
        starting_mag_block = ""

    return QE_TEMPLATE.format(
        prefix=prefix,
        nat=len(struct),
        ntyp=len(ordered),
        ecutwfc=ECUTWFC,
        ecutrho=ECUTRHO,
        nspin=nspin,
        starting_mag_block=starting_mag_block,
        tot_charge=charge,
        hubbard_u=HUBBARD_U_GA,
        species_block=species_block,
        positions_block=positions_block,
        kx=KMESH[0], ky=KMESH[1], kz=KMESH[2],
        cell_block=cell_block,
    )


def load_relaxed_structure(dopant: str, site: str) -> Optional[Structure]:
    """Load MACE-MP-0 relaxed 80-atom supercell from the q0 json (used as starting
    geometry for q=0/+1/+2 per gen_qe_inputs.py convention)."""
    f = RELAX_DIR / f"{dopant}_{site}_q0.json"
    if not f.exists():
        return None
    with open(f) as fh:
        payload = json.load(fh)
    return Structure.from_dict(payload["structure"])


def pseudo_exists(elem: str) -> bool:
    return (PSEUDO_DIR / f"{elem}.upf").exists()


def write_run_script(generated: list[dict]) -> None:
    """Per-card serial driver (one calc at a time). Safe to resume — skips calcs
    whose .out already contains 'JOB DONE'. Per memory rule feedback_ga2o3_gpu
    (2× RTX 3090; per-GPU jobs must be serial). pw.x in qe75 conda env runs on
    CPU (pure-CPU build), so no GPU is touched."""
    inputs_sorted = sorted(g["prefix"] for g in generated)
    lines = [
        "#!/bin/bash",
        "# V58 gap-dopant PBE+U driver (Phase 58).",
        "# Pure CPU. Serial per-card execution (no GPU touched).",
        "# Safe to resume: a calc is skipped if its .out already contains 'JOB DONE'.",
        "#",
        "# Usage:",
        "#   tmux new -d -s v58_gap 'bash dft/qe_hse06/run_v58_gap.sh > dft/qe_hse06/logs/run_v58_gap.log 2>&1'",
        "#   tmux attach -t v58_gap",
        "#",
        f"# Total queued: {len(generated)} PBE+U calcs",
        f"# Wall-clock estimate: {len(generated) * PBEU_HOURS_PER_CALC:.0f} h serial on 16-core CPU",
        "",
        "set -u  # -e omitted: a single failed calc must not kill the queue",
        "",
        "PROJ=/home/lawrence/Physics/Ga2O3-Net",
        "QE_DIR=$PROJ/dft/qe_hse06",
        "IN_DIR=$QE_DIR/inputs_v58_gap",
        "OUT_DIR=$QE_DIR/outputs",
        "LOG_DIR=$QE_DIR/logs",
        "mkdir -p \"$OUT_DIR\" \"$LOG_DIR\" \"$QE_DIR/outdir\"",
        "",
        "NTHREAD=${NTHREAD:-16}",
        "MPI_LAUNCHER=\"mpirun -np $NTHREAD\"",
        "PW_X=\"pw.x\"",
        "",
        "source /home/lawrence/anaconda3/etc/profile.d/conda.sh",
        "conda activate qe75",
        "",
        "run_calc() {",
        "    local prefix=$1",
        "    local in_file=\"$IN_DIR/${prefix}.in\"",
        "    local out_file=\"$OUT_DIR/${prefix}.out\"",
        "    local log_file=\"$LOG_DIR/${prefix}.log\"",
        "    if [ -f \"$out_file\" ] && grep -q \"JOB DONE\" \"$out_file\" 2>/dev/null; then",
        "        echo \"[$(date '+%Y-%m-%d %H:%M:%S')] SKIP ${prefix} (already JOB DONE)\" | tee -a \"$log_file\"",
        "        return 0",
        "    fi",
        "    echo \"[$(date '+%Y-%m-%d %H:%M:%S')] START ${prefix}\" | tee -a \"$log_file\"",
        "    cd \"$IN_DIR\"",
        "    $MPI_LAUNCHER $PW_X -in \"${prefix}.in\" > \"$out_file\" 2>&1",
        "    local rc=$?",
        "    if [ $rc -ne 0 ]; then",
        "        echo \"[$(date '+%Y-%m-%d %H:%M:%S')] FAIL ${prefix} (rc=$rc)\" | tee -a \"$log_file\"",
        "        return 1",
        "    fi",
        "    if grep -q \"JOB DONE\" \"$out_file\" 2>/dev/null; then",
        "        echo \"[$(date '+%Y-%m-%d %H:%M:%S')] DONE ${prefix}\" | tee -a \"$log_file\"",
        "        return 0",
        "    fi",
        "    echo \"[$(date '+%Y-%m-%d %H:%M:%S')] FAIL ${prefix} (no JOB DONE token)\" | tee -a \"$log_file\"",
        "    return 1",
        "}",
        "",
        "echo \"[$(date '+%Y-%m-%d %H:%M:%S')] V58 gap-dopant PBE+U queue starting\"",
        f"echo \"Total calcs: {len(generated)}\"",
        "",
    ]
    for prefix in inputs_sorted:
        lines.append(f"run_calc \"{prefix}\"")
    lines += [
        "",
        "echo \"[$(date '+%Y-%m-%d %H:%M:%S')] V58 gap-dopant PBE+U queue finished\"",
        "conda deactivate",
        "",
    ]
    RUN_SCRIPT.write_text("\n".join(lines))
    RUN_SCRIPT.chmod(RUN_SCRIPT.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    log.info(f"Wrote {RUN_SCRIPT} (chmod +x)")


def write_blocked_doc(blocked_dopants: dict[str, tuple[str, str]]) -> None:
    lines = [
        "# V58 Gap-Dopant DFT Blockers (Phase 58)",
        "",
        "Some Phase-58 priority dopants cannot be computed because their ONCV",
        "norm-conserving PBE pseudopotentials are not present in",
        "`dft/qe_hse06/pseudos/`. Each is listed below with the reason and the",
        "pseudo-dojo source needed to unblock.",
        "",
        "Locked convention (must match the 87-calc HSE06 corpus, design doc §7):",
        "",
        "- Family: **ONCV nc-sr-04 PBE**, scalar-relativistic, standard accuracy",
        "- Source: https://www.pseudo-dojo.org/  → NC SR (ONCVPSP v0.4) → PBE → standard",
        "- Format: `.upf` (rename to `{Element}.upf`, place into `dft/qe_hse06/pseudos/`)",
        "",
        "Mixing pseudo families inside a single ΔE_f table (e.g. SSSP PAW for F + ONCV",
        "for the rest) is physically wrong — pick the same family for every species.",
        "",
        "## Blocked dopants",
        "",
        "| Dopant | Why needed | Pseudo source |",
        "|--------|------------|---------------|",
    ]
    for elem, (reason, src) in blocked_dopants.items():
        lines.append(f"| **{elem}** | {reason} | {src} |")
    lines += [
        "",
        "## F is the top priority",
        "",
        "Design doc §1.1 names F as one of the two Tier-1C OFFSET cases (+2.2 dex above",
        "baseline). V55-Ext currently has *no* training data in F's chemical class",
        "(anion substitution V_O, n=5 corpus rows). DFT-anchored BrouwerHead cannot fix",
        "F unless F's ΔE_f^DFT is in the cache. Download F.upf first and re-run",
        "`scripts/44_v58_gen_gap_qe_inputs.py` to add the missing 9 calcs.",
        "",
        "## Re-run procedure once a pseudo is added",
        "",
        "```bash",
        "# Example: F unblock",
        "cp ~/Downloads/F.upf  /home/lawrence/Physics/Ga2O3-Net/dft/qe_hse06/pseudos/",
        "# Build the relaxed q0 supercells first (MACE-MP-0 screening):",
        "conda activate ga2o3",
        "python dft/screening/gen_supercells.py --elements F",
        "python dft/screening/mace_relax.py --elements F",
        "# Now generate the QE inputs:",
        "python scripts/44_v58_gen_gap_qe_inputs.py",
        "# Then queue:",
        "tmux new -d -s v58_gap_F 'bash dft/qe_hse06/run_v58_gap.sh > dft/qe_hse06/logs/run_v58_gap_F.log 2>&1'",
        "```",
        "",
        "## Lanthanides (Er, Eu) — note on spin",
        "",
        "Er³⁺ (4f¹¹) and Eu³⁺ (4f⁶) carry unpaired f-electrons and are paramagnetic in",
        "real β-Ga₂O₃:Ln samples. The Phase-58 spec restricts `nspin=2` to the listed",
        "open-shell 3d/4d/5d transition metals (Fe, Cr, V, W, Ti, Cu, Ni) only — Er, Eu",
        "are generated with `nspin=1`. If Tier-1C residuals for Er/Eu turn out to be",
        "sensitive to f-electron polarization, re-generate those 6 calcs (Er, Eu × 3",
        "sites × 1 q) with `nspin=2 + starting_magnetization` and treat as a tier-2",
        "refinement.",
        "",
    ]
    BLOCKED_DOC.write_text("\n".join(lines))
    log.info(f"Wrote {BLOCKED_DOC}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Start from the hard-coded blocker list (F/H/La/In/Pd/Hf) and append any
    # priority-list dopant whose pseudo is missing at runtime.
    blocked_dopants = dict(BLOCKED_DOPANTS)

    generated: list[dict] = []
    skipped_pseudo: list[str] = []
    skipped_relax: list[tuple[str, str]] = []

    for dopant in DOPANTS:
        if not pseudo_exists(dopant):
            log.warning(f"  SKIP dopant {dopant}: pseudo {dopant}.upf missing")
            skipped_pseudo.append(dopant)
            reason = DOPANT_REASON.get(dopant, "Phase-58 priority dopant")
            blocked_dopants[dopant] = (
                f"{reason} — pseudo {dopant}.upf NOT FOUND in dft/qe_hse06/pseudos/",
                f"https://www.pseudo-dojo.org/  (NC, scalar-relativistic PBE, standard, {dopant}.upf)",
            )
            continue
        for site in SITES:
            struct = load_relaxed_structure(dopant, site)
            if struct is None:
                log.warning(f"  SKIP {dopant}/{site}: no relaxed q0 json")
                skipped_relax.append((dopant, site))
                continue
            # Confirm pseudo for Ga + O too
            if not pseudo_exists("Ga") or not pseudo_exists("O"):
                raise FileNotFoundError(
                    f"Ga.upf or O.upf missing under {PSEUDO_DIR}")
            # 3 charge states share the same q0 starting geometry (gen_qe_inputs.py
            # convention — no q1/q2 supercells exist in dft/screening/results/relaxed/)
            for q in CHARGES:
                prefix = f"PBEU_v58_{dopant}_{site}_q{q}"
                inp = render_qe_input(struct, prefix=prefix, charge=q, dopant=dopant)
                out_file = OUT_DIR / f"{prefix}.in"
                out_file.write_text(inp)
                species_present = sorted({sp.symbol for sp in struct.species})
                generated.append({
                    "prefix": prefix,
                    "dopant": dopant,
                    "site": site,
                    "charge": q,
                    "nat": len(struct),
                    "ntyp": len(species_present),
                    "nspin": 2 if dopant in TM_NSPIN2 else 1,
                    "hubbard_U_Ga_eV": HUBBARD_U_GA,
                    "ecutwfc": ECUTWFC,
                    "ecutrho": ECUTRHO,
                    "starting_geometry_source": str(
                        (RELAX_DIR / f"{dopant}_{site}_q0.json").relative_to(PROJ)),
                    "out_file": str(out_file.relative_to(PROJ)),
                })

    log.info(f"Generated {len(generated)} input files in {OUT_DIR.relative_to(PROJ)}/")

    # Manifest
    per_dopant_counts = {}
    for g in generated:
        per_dopant_counts[g["dopant"]] = per_dopant_counts.get(g["dopant"], 0) + 1
    manifest = {
        "phase": "Phase 58 — V58 gap-dopant PBE+U queue",
        "generator": "scripts/44_v58_gen_gap_qe_inputs.py",
        "convention": {
            "pseudo_family": "ONCV nc-sr-04 PBE (norm-conserving)",
            "ecutwfc_Ry": ECUTWFC,
            "ecutrho_Ry": ECUTRHO,
            "hubbard_U_Ga_3d_eV": HUBBARD_U_GA,
            "supercell_nat_with_dopant_no_vacancy": 79,
            "k_points": list(KMESH),
            "nspin_2_for": sorted(TM_NSPIN2),
            "starting_geometry": "MACE-MP-0 relaxed q0 cell (shared across q=0/+1/+2)",
        },
        "total_calcs": len(generated),
        "per_dopant_count": per_dopant_counts,
        "estimated_wall_clock_h_per_calc_PBE_U_16core_CPU": PBEU_HOURS_PER_CALC,
        "estimated_wall_clock_h_total_serial": len(generated) * PBEU_HOURS_PER_CALC,
        "blocked_dopants": {e: {"reason": r, "pseudo_source": s}
                            for e, (r, s) in blocked_dopants.items()},
        "skipped_dopants_missing_pseudo": skipped_pseudo,
        "skipped_combinations_missing_relax_json": [
            {"dopant": d, "site": s} for (d, s) in skipped_relax],
        "rows": generated,
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info(f"Wrote manifest {manifest_path.relative_to(PROJ)} "
             f"(total={len(generated)} calcs)")

    # Driver + blocked doc
    write_run_script(generated)
    write_blocked_doc(blocked_dopants)

    # Summary
    print("\n=== V58 gap-dopant PBE+U generation summary ===")
    print(f"Total calcs queued: {len(generated)}")
    for d, c in sorted(per_dopant_counts.items()):
        print(f"  {d:3s}: {c} calcs")
    print(f"\nBlocked dopants (no ONCV pseudo): {', '.join(blocked_dopants.keys())}")
    if skipped_pseudo:
        print(f"  ⚠ Of these, the following were on the Phase-58 priority list but had no pseudo on disk: "
              f"{', '.join(skipped_pseudo)}")
    print(f"Total wall-clock estimate (serial, PBE+U 16-core CPU): "
          f"{len(generated) * PBEU_HOURS_PER_CALC:.0f} h "
          f"= {len(generated) * PBEU_HOURS_PER_CALC / 24:.1f} d")
    print(f"\nNext step (do NOT run yet):")
    print(f"  tmux new -d -s v58_gap 'bash dft/qe_hse06/run_v58_gap.sh "
          f"> dft/qe_hse06/logs/run_v58_gap.log 2>&1'")


if __name__ == "__main__":
    main()
