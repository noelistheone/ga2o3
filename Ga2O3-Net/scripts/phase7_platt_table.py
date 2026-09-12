"""
Phase 7A — Table 4: per-dopant Platt calibrator (a, b, N_train) table.

Reads the stdout text produced by scripts/apply_per_dopant_platt.py for both
targets and parses the "Per-dopant calibrators fitted for ..." block into a
LaTeX booktabs table + a matching CSV.

Expected input: two plain-text files with the captured stdout, one per target.
If the text file is missing, the script gracefully skips that target.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


_PATTERN = re.compile(
    r"^\s*(?P<dopant>[A-Za-z0-9+\-/]+)\s+N_train=\s*(?P<n>\d+)\s+"
    r"a=(?P<a>[+\-\d\.eE]+)\s+b=(?P<b>[+\-\d\.eE]+)\s*$"
)


def parse_stdout(text: str) -> pd.DataFrame:
    """Extract per-dopant (dopant, N, a, b) rows from an apply_per_dopant_platt stdout."""
    rows = []
    in_block = False
    for line in text.splitlines():
        if "Per-dopant calibrators fitted" in line:
            in_block = True
            continue
        if in_block:
            m = _PATTERN.match(line)
            if m:
                rows.append(dict(
                    dopant=m.group("dopant"),
                    N_train=int(m.group("n")),
                    a=float(m.group("a")),
                    b=float(m.group("b")),
                ))
                continue
            if line.strip() == "" or "OOF metrics" in line or "metrics" in line.lower() and "[" in line:
                in_block = False
    return pd.DataFrame(rows)


def to_latex(df_pdr: pd.DataFrame | None,
             df_vc: pd.DataFrame | None,
             caption: str, label: str) -> str:
    parts = set()
    if df_pdr is not None and not df_pdr.empty:
        parts.update(df_pdr["dopant"].tolist())
    if df_vc is not None and not df_vc.empty:
        parts.update(df_vc["dopant"].tolist())
    dopants = sorted(parts, key=lambda d: -(df_pdr.set_index("dopant").at[d, "N_train"]
                                            if df_pdr is not None and d in df_pdr["dopant"].values
                                            else 0))
    has_pdr = df_pdr is not None and not df_pdr.empty
    has_vc = df_vc is not None and not df_vc.empty

    col_spec = "l"
    head_cells = ["Dopant"]
    if has_pdr:
        col_spec += "r r r"
        head_cells += [r"$N_{\text{train}}$ (PDR)", r"$a$ (PDR)", r"$b$ (PDR)"]
    if has_vc:
        col_spec += "r r r"
        head_cells += [r"$N_{\text{train}}$ (VC)", r"$a$ (VC)", r"$b$ (VC)"]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        " & ".join(head_cells) + r" \\",
        r"\midrule",
    ]
    for dop in dopants:
        row = [dop]
        if has_pdr:
            m = df_pdr[df_pdr["dopant"] == dop]
            if len(m):
                row += [str(int(m["N_train"].iloc[0])),
                        f"{float(m['a'].iloc[0]):+.3f}",
                        f"{float(m['b'].iloc[0]):+.3f}"]
            else:
                row += ["---", "---", "---"]
        if has_vc:
            m = df_vc[df_vc["dopant"] == dop]
            if len(m):
                row += [str(int(m["N_train"].iloc[0])),
                        f"{float(m['a'].iloc[0]):+.3f}",
                        f"{float(m['b'].iloc[0]):+.3f}"]
            else:
                row += ["---", "---", "---"]
        lines.append(" & ".join(row) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdr-stdout", type=Path, default=None,
                    help="Captured stdout from apply_per_dopant_platt.py for PDR")
    ap.add_argument("--vc-stdout", type=Path, default=None,
                    help="Same for VC")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    df_pdr = parse_stdout(args.pdr_stdout.read_text()) if args.pdr_stdout and args.pdr_stdout.exists() else None
    df_vc = parse_stdout(args.vc_stdout.read_text()) if args.vc_stdout and args.vc_stdout.exists() else None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tex = to_latex(df_pdr, df_vc,
                   "Per-dopant Platt calibrators (slope $a$, intercept $b$) fitted on "
                   "OOF predictions. Global slope/intercept applies to dopants with "
                   "$N_{\\text{train}} < 5$.",
                   "tab:platt_calibrators")
    (args.out_dir / "tab4_platt_calibrators.tex").write_text(tex)

    combined = []
    if df_pdr is not None:
        d = df_pdr.copy()
        d["target"] = "photo_dark_ratio"
        combined.append(d)
    if df_vc is not None:
        d = df_vc.copy()
        d["target"] = "vacancy_concentration"
        combined.append(d)
    if combined:
        pd.concat(combined, ignore_index=True).to_csv(args.out_dir / "tab4_platt_calibrators.csv", index=False)

    print(f"Wrote tab4: {args.out_dir / 'tab4_platt_calibrators.tex'}")


if __name__ == "__main__":
    main()
