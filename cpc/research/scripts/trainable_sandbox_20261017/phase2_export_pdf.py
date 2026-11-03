"""Export the phase2 result Markdown, tables and figures to a local PDF."""
from pathlib import Path
import re,subprocess,shutil,json,hashlib
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/trainable_sandbox_20261017/phase2_energy"
SOURCE=OUT/"第二阶段_能量学习与掺杂预测结果.md"

def esc(s):
    mapping={"\\":r"\textbackslash{}","&":r"\&","%":r"\%","$":r"\$",
             "#":r"\#","_":r"\_","{":r"\{","}":r"\}","^":r"\textasciicircum{}","~":r"\textasciitilde{}"}
    return "".join(mapping.get(c,c) for c in str(s))
def inline(s):
    parts=re.split(r"(\*\*.*?\*\*)",s)
    return "".join(r"\textbf{"+esc(p[2:-2])+"}" if p.startswith("**") else esc(p) for p in parts)
def main():
    lines=SOURCE.read_text().splitlines();body=[];i=0
    title=lines[0].lstrip("# ")
    while i<len(lines):
        line=lines[i].strip()
        if i==0 or not line:i+=1;continue
        if line.startswith("## "):
            body.append(r"\section*{"+esc(line[3:])+"}");i+=1;continue
        image=re.fullmatch(r"!\[(.*?)\]\((.*?)\)",line)
        if image:
            body.append(r"\begin{center}\includegraphics[width=.98\textwidth]{"+
                        image[2]+r"}\\{\footnotesize "+esc(image[1])+r"}\end{center}")
            i+=1;continue
        if line.startswith("|"):
            table=[]
            while i<len(lines) and lines[i].strip().startswith("|"):
                cells=[v.strip() for v in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[- :]+",v) for v in cells):table.append(cells)
                i+=1
            n=len(table[0]);cols="p{.34\\textwidth}"+"r"*(n-1) if n==4 else ("lll"+"r"*(n-3) if n==7 else "l"+"r"*(n-1))
            body.append(r"{\footnotesize\setlength{\tabcolsep}{4pt}\begin{longtable}{@{}"+cols+r"@{}}\toprule")
            body.append(" & ".join(esc(v) for v in table[0])+r"\\\midrule\endhead")
            body.extend(" & ".join(esc(v) for v in row)+r"\\" for row in table[1:])
            body.append(r"\bottomrule\end{longtable}}")
            continue
        body.append(inline(line)+"\n\n");i+=1
    preamble=r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=21mm]{geometry}
\usepackage{fontspec,booktabs,longtable,array,graphicx,hyperref,fancyhdr}
\setmainfont{Noto Serif CJK SC}[ItalicFont={Noto Serif CJK SC}]
\setsansfont{Noto Sans CJK SC}
\XeTeXlinebreaklocale "zh"
\XeTeXlinebreakskip=0pt plus 1pt
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\setlength{\emergencystretch}{4em}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=blue}
\pagestyle{fancy}\fancyhf{}\lhead{ML＋Sandbox 掺杂性质预测}\rhead{第二阶段研发记录}\cfoot{\thepage}
\setlength{\headheight}{15pt}
\title{"""+esc(title)+r"""}\author{回顾性计算验证；goal仍为active}
\date{2026年10月17日}
\begin{document}\maketitle
"""
    build=OUT/"_latex_build";build.mkdir(exist_ok=True)
    (build/"figures").mkdir(exist_ok=True)
    for p in (OUT/"figures").glob("*.png"):shutil.copy2(p,build/"figures"/p.name)
    tex=build/"phase2_results.tex";tex.write_text(preamble+"\n".join(body)+r"\end{document}")
    compiler=shutil.which("xelatex") or "/home/lawrence/.local/bin/xelatex"
    for run in range(2):
        result=subprocess.run([compiler,"-interaction=nonstopmode","-halt-on-error",tex.name],
                              cwd=build,capture_output=True,text=True)
        (OUT/f"pdf_build_{run+1}.log").write_text(result.stdout+"\n"+result.stderr)
        result.check_returncode()
    dest=SOURCE.with_suffix(".pdf");shutil.copy2(tex.with_suffix(".pdf"),dest)
    info=subprocess.run(["pdfinfo",str(dest)],capture_output=True,text=True,check=True).stdout
    (OUT/"pdfinfo.txt").write_text(info)
    (OUT/"pdf_export.json").write_text(json.dumps(dict(
        markdown_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        pdf_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
        renderer="local XeLaTeX; complete result Markdown plus all three figures"),indent=2)+"\n")
    print(dest);print(info.split("Pages:")[1].splitlines()[0].strip()+" pages")
if __name__=="__main__":main()
