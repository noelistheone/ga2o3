"""Copy frozen shared weights and learned source-distribution choices into v9."""
from pathlib import Path
import json,sys,copy,shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import phase22_prediction as p
BUNDLE=p.OUT/'delivery/predictor_bundle_v9'


def build(bundle,choices,*,grade):
    parent=json.loads((p.s.OUT/'delivery/predictor_bundle_v8/manifest.json').read_text());m=copy.deepcopy(parent);m.update(version=9,variants={},files=[],energy_methods=[],
        grid_size=p.SIZE,scope='finite normalized source inference;no new shared training',grade=grade)
    files={}
    for variant in sorted({c['variant'] for c in choices}):
        cs=[c for c in choices if c['variant']==variant];assert len(cs)==3 and len({c['fit_key'] for c in cs})==1
        c=cs[0];src=ROOT/c['fit_path'];dest=Path(bundle)/'models'/f"{c['fit_key']}.json";dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert p.s.energy.sha(dest)==p.s.energy.sha(src)
        else:shutil.copyfile(src,dest)
        files[str(dest.relative_to(bundle))]=p.s.energy.sha(dest)
        m['variants'][variant]=dict(model_key=c['fit_key'],source_kind=c['kind'],reg=c['reg'],
            source_selection={str(x['budget']):x['selected_source']['config'] for x in cs})
    m['files']=[dict(path=k,sha256=v) for k,v in files.items()];p.s.energy.write(Path(bundle)/'manifest.json',m)
    return m


def main():
    assert json.loads((p.OUT/'execution_complete.json').read_text())['variants']==10
    choices=[c for c in json.loads((p.OUT/'all_selections.json').read_text()) if c['held'] is None]
    m=build(BUNDLE,choices,grade='full5source development predictor;not independent validation')
    p.s.energy.write(p.OUT/'delivery/export_audit.json',dict(passed=True,presets=len(m['variants']),copied_model_files=len(m['files']),
        shared_weights_unchanged=True,new_shared_optimizer_starts=0,goal_complete=False))
    print('exported v9',len(m['variants']),'presets',len(m['files']),'copied models',flush=True)


if __name__=='__main__':main()
