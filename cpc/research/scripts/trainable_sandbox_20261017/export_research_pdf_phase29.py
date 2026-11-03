"""Render local research Markdown with intact LaTeX mathematics via XeLaTeX."""
from pathlib import Path
import argparse,re,subprocess,shutil,json,hashlib,datetime
from zoneinfo import ZoneInfo

def esc(s):
    replacements={"\\":r"\textbackslash{}","&":r"\&","%":r"\%","$":r"\$",
        "#":r"\#","_":r"\_","{":r"\{","}":r"\}","^":r"\textasciicircum{}","~":r"\textasciitilde{}"}
    return ("".join(replacements.get(c,c) for c in str(s))
            .replace("/",r"/\allowbreak{}")
            .replace(r"\_",r"\_\allowbreak{}")
            .replace("→",r"→\allowbreak{}"))

def inline(s):
    token=re.compile(r"(\\\(.*?\\\)|\[[^\]]+\]\([^)]+\)|\*\*.*?\*\*)")
    parts=token.split(s);out=[]
    for p in parts:
        if p.startswith(r"\("):out.append(p)
        elif p.startswith("**"):out.append(r"\textbf{"+esc(p[2:-2])+"}")
        elif re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)",p):
            m=re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)",p)
            out.append(r"\href{"+m[2]+r"}{"+esc(m[1])+"}")
        else:out.append(esc(p))
    return "".join(out)

def convert(source):
    lines=source.read_text().splitlines();body=[];i=1
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line==r"\[":
            end=i+1
            while end<len(lines) and lines[end].strip()!=r"\]":end+=1
            if end==len(lines):raise ValueError("Unclosed display math")
            body.append("\n".join(lines[i:end+1]));i=end+1;continue
        if line.startswith("## "):
            body.append(r"\section*{"+esc(line[3:])+"}");i+=1;continue
        if line.startswith("### "):
            body.append(r"\subsection*{"+esc(line[4:])+"}");i+=1;continue
        m=re.fullmatch(r"!\[(.*?)\]\((.*?)\)",line)
        if m:
            path=(source.parent/m[2]).resolve().as_posix()
            body.append(r"\begin{center}\includegraphics[width=.98\textwidth]{"+
                path+r"}\\{\footnotesize "+inline(m[1])+r"}\end{center}")
            i+=1;continue
        if line.startswith("|"):
            table=[]
            while i<len(lines) and lines[i].strip().startswith("|"):
                cells=[v.strip() for v in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[- :]+",c) for c in cells):table.append(cells)
                i+=1
            n=len(table[0])
            widths=([.44,.14,.14,.14] if n==4 else
                    [.31,.155,.155,.155,.155] if n==5 else
                    [.56]+[.36/(n-1)]*(n-1))
            columns="".join(r">{\raggedright\arraybackslash}p{"+str(w)+r"\textwidth}" for w in widths)
            body.append(r"{\footnotesize\setlength{\tabcolsep}{3pt}\begin{longtable}{@{}"+columns+r"@{}}\toprule")
            body.append(" & ".join(inline(c) for c in table[0])+r"\\\midrule\endhead")
            body.extend(" & ".join(inline(c) for c in row)+r"\\" for row in table[1:])
            body.append(r"\bottomrule\end{longtable}}")
            continue
        body.append(inline(line)+"\n\n");i+=1
    return lines[0].lstrip("# "),"\n".join(body)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("source",type=Path)
    a=ap.parse_args();source=a.source.resolve()
    title,body=convert(source)
    date_label=datetime.datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y年%m月%d日')
    build=source.parent/("_pdf_build_"+hashlib.sha256(source.name.encode()).hexdigest()[:8])
    build.mkdir(exist_ok=True)
    preamble=r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=20mm]{geometry}
\usepackage{fontspec,amsmath,amssymb,booktabs,longtable,array,graphicx,hyperref,fancyhdr}
\setmainfont{Noto Serif CJK SC}[ItalicFont={Noto Serif CJK SC}]
\setsansfont{Noto Sans CJK SC}
\setmonofont{DejaVu Sans Mono}
\XeTeXlinebreaklocale "zh"
\XeTeXlinebreakskip=0pt plus 1pt
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\setlength{\emergencystretch}{4em}\setlength{\headheight}{15pt}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=blue}
\pagestyle{fancy}\fancyhf{}\lhead{ML＋Sandbox 掺杂性质预测}\rhead{理论与计算证据}\cfoot{\thepage}
\title{"""+esc(title)+r"""}\author{可复现研究记录；不预设有利精度结论}
\date{"""+date_label+r"""}
\begin{document}\maketitle
"""
    tex=build/"research_report.tex";tex.write_text(preamble+body+r"\end{document}")
    compiler=shutil.which("xelatex") or "/home/lawrence/.local/bin/xelatex"
    for k in range(2):
        run=subprocess.run([compiler,"-interaction=nonstopmode","-halt-on-error",tex.name],
                           cwd=build,capture_output=True,text=True)
        (build/f"build_{k+1}.log").write_text(run.stdout+"\n"+run.stderr)
        run.check_returncode()
    target=source.with_suffix(".pdf");shutil.copyfile(tex.with_suffix(".pdf"),target)
    info=subprocess.run(["pdfinfo",str(target)],capture_output=True,text=True,check=True).stdout
    (build/"pdfinfo.txt").write_text(info)
    warnings=[line for line in run.stdout.splitlines() if "Overfull" in line or "Missing character" in line]
    (build/"export.json").write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        pdf_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        equations_preserved=True,warnings=warnings,renderer="XeLaTeX",report_local_date=date_label),ensure_ascii=False,indent=2)+"\n")
    print(str(target));print(info.split("Pages:")[1].splitlines()[0].strip()+" pages")
    if warnings:print("\n".join(warnings))

if __name__=="__main__":main()
