"""Preserve nominal-target identifiers when reading the new source CSV.

The V1 reader converted the numeric-looking sample_label column to floats.
Use the textual specimen ID as the authority; equations/protocol are unchanged.
"""
from pathlib import Path
import json,sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parent))
import phase19_leedy as parent

OLD_READ=parent.read;OLD_FREEZE=parent.freeze
parent.SOURCE=parent.OUT/'leedy2018_v2'


def read(path):
    d=OLD_READ(path)
    if 'sample_label' in d and 'sample_id' in d:
        d['sample_label']=d.sample_id.str.rsplit('::target-',n=1).str[-1]
    return d


def freeze():
    OLD_FREEZE()
    payload=dict(created_utc=parent.energy.now(),wrapper_sha256=parent.energy.sha(Path(__file__)),
        original_source_sha256=parent.energy.sha(Path(parent.__file__)),
        retained_v1='phase19_delivery/leedy2018 andleedy.log:CSVnumericidentifierconversion error before predictions were saved',
        repair='reconstruct nominaltarget identifier from explicit sample_id;no numeric/model/selection change')
    p=parent.SOURCE/'reader_amendment.json'
    if p.exists():assert json.loads(p.read_text())['wrapper_sha256']==payload['wrapper_sha256']
    else:parent.energy.write(p,payload)


parent.read=read;parent.freeze=freeze
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--extract',action='store_true');ap.add_argument('--evaluate',action='store_true');a=ap.parse_args();freeze()
    if a.extract:parent.extract()
    if a.evaluate:parent.evaluate()
