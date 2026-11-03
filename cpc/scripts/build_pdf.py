"""Build the official-template manuscript and separate supplement."""
from pathlib import Path
import os, shutil, subprocess
ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'build'
BUILD.mkdir(exist_ok=True)
env=os.environ.copy()
env['PATH']=str(Path.home()/'.local/bin')+os.pathsep+env.get('PATH','')
for name in ('main','supplementary'):
 commands=[['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build',name+'.tex'],
           ['bibtex','build/'+name],
           ['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build',name+'.tex'],
           ['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build',name+'.tex']]
 for j,cmd in enumerate(commands):
  run=subprocess.run(cmd,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  (BUILD/f'{name}_pass{j}.txt').write_text(run.stdout)
  if run.returncode:
   print(run.stdout[-6000:]);raise SystemExit(run.returncode)
 shutil.copy2(BUILD/(name+'.pdf'),ROOT/(name+'.pdf'))
 print('Built',name,flush=True)
