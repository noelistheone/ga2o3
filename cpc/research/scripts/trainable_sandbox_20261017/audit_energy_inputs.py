"""Read-only inventory of prospective multi-fidelity energy labels.

Certifies raw SCF/geometry pairing only. It does not certify thermodynamic CTLs,
finite-size corrections, spin ground states or rare-earth transferability.
"""
from pathlib import Path
import re,json,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/"dft/qe_cc"
OUT=ROOT/"results/trainable_sandbox_20261017"

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):
    if p is None:return None
    try:return str(p.resolve().relative_to(ROOT.resolve()))
    except ValueError:return str(p.resolve())
def value(text,key,default=None):
    m=re.search(r"(?im)^\s*"+re.escape(key)+r"\s*=\s*([^,!\n]+)",text)
    return m.group(1).strip().strip("'\"") if m else default
def card(text,name,n,last=False):
    matches=list(re.finditer(r"(?m)^\s*"+name+r"[^\n]*\n",text))
    if not matches:return None,None
    m=matches[-1] if last else matches[0]
    rows=text[m.end():].splitlines()[:n]
    if len(rows)!=n:return None,None
    return m.group().strip(),rows
def positions(text,n,last=False):
    header,rows=card(text,"ATOMIC_POSITIONS",n,last)
    if header is None or "angstrom" not in header.lower():return None,None
    try:return [x.split()[0] for x in rows],np.array([[float(v) for v in x.split()[1:4]] for x in rows])
    except (ValueError,IndexError):return None,None
def cell(text):
    header,rows=card(text,"CELL_PARAMETERS",3)
    if header is None or "angstrom" not in header.lower():return None
    try:return np.array([[float(v) for v in x.split()[:3]] for x in rows])
    except (ValueError,IndexError):return None
def pseudo_metadata(path):
    if path is None or not path.exists():return {}
    s=path.read_text(errors="replace")
    z=re.search(r'z_valence\s*=\s*"\s*([\d.]+)',s)
    info=re.search(r"<PP_INPUTFILE>(.*?)</PP_INPUTFILE>",s,re.S)
    frozen=[]
    if info:
        c=re.search(r"(?m)^\s*[A-Za-z]+\s+[\d.]+\s+(\d+)\s+(\d+)\s+",info.group(1))
        conf=re.findall(r"(?m)^\s*(\d+)\s+(\d+)\s+([\d.]+)\s*$",info.group(1))
        if c:
            frozen=[f"{n}f^{v}" for n,l,v in conf[:int(c.group(1))] if int(l)==3]
    return dict(pseudo_sha256=sha(path),pseudo_z_valence=float(z.group(1)) if z else None,
                frozen_core_f_reference=";".join(frozen) or None)

def main():
    rows=[]
    for p in sorted(FOLDER.glob("hse_dop_*.in")):
        s=p.read_text(errors="replace");o=p.with_suffix(".out")
        t=o.read_text(errors="replace") if o.exists() else ""
        m=re.match(r"hse_dop_([A-Za-z]+)_q(-?\d+)(_a033)?$",p.stem)
        if not m:continue
        label,q=m.group(1),int(m.group(2))
        element="Sb" if label=="Sbtet" else label
        pp=FOLDER/(f"dop_{label}_relax.out" if q==0 else f"dop_{label}_q{q}_relax.out")
        if not pp.exists():pp=FOLDER/f"dop_{label}_q{q}.out"
        low=pp if pp.exists() else None
        pt=low.read_text(errors="replace") if low else ""
        pi=low.with_suffix(".in") if low else None
        ps=pi.read_text(errors="replace") if pi is not None and pi.exists() else ""
        nat=int(value(s,"nat","0"))
        hi_species,hi_pos=positions(s,nat)
        lo_species,lo_pos=positions(pt,nat,last=True)
        geom_error=(float(np.max(np.abs(hi_pos-lo_pos))) if hi_pos is not None
                    and lo_pos is not None and hi_species==lo_species else None)
        hc,lc=cell(s),cell(ps)
        cell_error=float(np.max(np.abs(hc-lc))) if hc is not None and lc is not None else None
        pseudo=Path(value(s,"pseudo_dir",""))/(element+".upf")
        if not pseudo.exists():pseudo=ROOT/"dft/pseudos_r8"/(element+".upf")
        meta=pseudo_metadata(pseudo)
        highE=re.findall(r"!\s+total energy\s*=\s*([-\d.]+)\s+Ry",t)
        lowE=re.findall(r"!\s+total energy\s*=\s*([-\d.]+)\s+Ry",pt)
        high_complete="JOB DONE." in t and "EXX self-consistency reached" in t
        low_complete="JOB DONE." in pt and ("End of BFGS Geometry Optimization" in pt or "bfgs converged" in pt)
        charge_matches=(float(value(s,"tot_charge","0"))==float(value(ps,"tot_charge","0")))
        # Resolve actual pseudopotential entries, not just directory names.
        lowpseudo=Path(value(ps,"pseudo_dir",""))/(element+".upf")
        samepseudo=lowpseudo.exists() and pseudo.exists() and sha(lowpseudo)==sha(pseudo)
        pair=(high_complete and low_complete and geom_error is not None and geom_error<1e-6
              and cell_error is not None and cell_error<1e-6 and charge_matches and samepseudo)
        rows.append(dict(element=element,configuration_label=label,charge=q,high_input=rel(p),high_output=rel(o),
            high_input_sha256=sha(p),high_output_sha256=sha(o) if o.exists() else None,
            high_job_done="JOB DONE." in t,high_exx_self_consistent="EXX self-consistency reached" in t,
            high_total_energy_Ry=float(highE[-1]) if highE else None,
            exchange_fraction_input=value(s,"exx_fraction","QE-HSE-default-not-assumed"),
            nat=nat,nspin=value(s,"nspin"),ecutwfc_Ry=value(s,"ecutwfc"),degauss_Ry=value(s,"degauss"),
            calculation=value(s,"calculation"),low_output=rel(low),low_input=rel(pi),
            low_job_done="JOB DONE." in pt,low_bfgs_converged=low_complete,
            low_total_energy_Ry=float(lowE[-1]) if lowE else None,
            high_vs_low_geometry_max_abs_A=geom_error,high_vs_low_cell_max_abs_A=cell_error,
            charge_matches=charge_matches,dopant_pseudo_matches=samepseudo,
            raw_energy_pair_checks_passed=pair,pseudo_path=rel(pseudo) if pseudo.exists() else None,
            **meta,certified_corrected_thermodynamic_CTL=False,
            label_grade="HSE single point at PBE geometry; raw pairing is separate from corrected thermodynamic CTL"))
    d=pd.DataFrame(rows);d.to_csv(OUT/"energy_input_inventory_v3.csv",index=False)
    summary=dict(raw_input_files=len(d),chemical_elements=int(d.element.nunique()),
        high_complete=int((d.high_job_done&d.high_exx_self_consistent).sum()),
        low_bfgs_complete=int(d.low_bfgs_converged.sum()),
        raw_pair_checks_passed=int(d.raw_energy_pair_checks_passed.sum()),
        exchange_fraction_groups=d.exchange_fraction_input.value_counts().to_dict(),
        frozen_f_core_elements=d[d.frozen_core_f_reference.notna()].groupby("element").frozen_core_f_reference.first().to_dict(),
        certified_corrected_multifidelity_CTLs_in_this_audit=0,
        incomplete_high_outputs=d[~(d.high_job_done&d.high_exx_self_consistent)].high_output.tolist(),
        limitations=["Raw pairing does not establish finite-size/potential alignment or HSE-relaxed CTL.",
                     "Frozen-core f occupations cannot change in the SCF charge calculation.",
                     "Default-HSE and explicit 0.33 exchange are separate fidelity groups.",
                     "Existing 80-atom potential alignment audit reports failure; reference gaps must match functional."],
        official_parameter_reference="https://www.quantum-espresso.org/Doc/INPUT_PW.html",
        next="Build labels only under explicit geometry, band-reference and correction contracts; assess required additional calculations.")
    (OUT/"energy_input_inventory_summary_v3.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
