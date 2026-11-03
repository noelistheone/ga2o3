"""Export every selected context model and source policy; no target labels in bundles."""
from pathlib import Path
import sys,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from sandbox.coherent_fermi import published_coherent_profile
from sandbox.kroger_db import NPZ
import phase25_training as t
import phase2_energy as e


def main():
    out=t.OUT/'delivery/predictor_bundle';pred=t.OUT/'prediction'
    assert json.loads((pred/'audit.json').read_text())['passed']
    completed=json.loads((pred/'execution_complete.json').read_text())
    assert completed['prediction_rows']==3762
    choices=json.loads((pred/'selections.json').read_text());presets={};models={}
    minimal=('key','method','parameters','coefficients','intercept','center','scale','offset_dex','training_valid','selected_success')
    for choice in choices:
        scheme=choice['scheme'];held=choice['held_unit'];context=scheme+('::full' if held is None else '::without:'+held)
        name=context+'::'+choice['variant']
        fpath=ROOT/choice['fit_path'];assert e.sha(fpath)==choice['fit_sha256']
        f=json.loads(fpath.read_text());models[f['key']]={k:f[k] for k in minimal if k in f}
        if name not in presets:
            tr=t.subset(scheme,choice['train_units'])
            support={element:dict(dose_cm3=[float(g.incorporated_cm3.min()),float(g.incorporated_cm3.max())],
                growth_T_C=[float(g.growth_T_C.min()),float(g.growth_T_C.max())]) for element,g in tr.groupby('element')}
            presets[name]=dict(context=context,variant=choice['variant'],method=choice['method'],source_kind=choice['kind'],
                model_key=choice['fit_key'],shared_reg=choice['reg'],source_lambdas={},training_support=support,
                validation_scheme=scheme,train_families=f['train_families'],held_unit=held,
                scope='retrospective study;full uses all33 conditions;source-excluded context excludes complete heldunit')
        assert presets[name]['model_key']==choice['fit_key']
        presets[name]['source_lambdas'][str(choice['budget'])]=choice['selected_source']['lam']
    assert len(presets)==247
    for spec in presets.values():assert set(spec['source_lambdas'])=={'0','1','2'}
    files=[]
    for key,value in models.items():
        path=out/'models'/(key+'.json');e.write(path,value);files.append(dict(path=str(path.relative_to(out)),sha256=e.sha(path)))
    profile=published_coherent_profile()
    manifest=dict(version=1,created_utc=e.now(),presets=presets,files=files,profile=profile.to_dict(),profile_sha256=profile.fingerprint,
        physical_reference_sha256=e.sha(NPZ),third_party_numeric_reference_copied=False,
        fitting_phase='phase25',conditions=33,publication_families=6,related_clusters=5,
        validation_grade='retrospective development;not blind or guaranteedindependentlaboratories',
        source_design='deduplicated supplied inputs;0/1/2Hall anchors;known-query-input conditional adaptation',
        new_mu_supervision=False,independent_sigma_labels=0,new_material_measurements=0,new_DFT_MD=0,goal_complete=False)
    e.write(out/'manifest.json',manifest)
    (t.OUT/'delivery/README.md').write_text('''# Saved conditional ML+Sandbox predictor

Use `predict_context_properties_v1.py --bundle <predictor_bundle> --request <request.json> --output <output.json>`.
The request has preset, inputs and optional anchors. Inputs explicitly supply the10fields in context_saved_model.INPUT_NAMES;anchors contain input_index and measured_nH_cm3 only. No query targets or dataset paths are accepted.

Each preset saves one shared model and three training-selected source policies. DOI::full and cluster::full are full33condition deployments with different source weighting. Without presets are source/group-excluded retrospective models. All247presets are retained; this does not mean247independentphysicalexperiments.

The public interface is the300Kcomparison Si/Sn research task. Output flags distinguish physical validity, support extrapolation, physical versus CT mobility and derived conductivity. Unknown oxygen conditions are explicit effective scenarios,not measured equilibrium pressures. No calibrated uncertainty or rareearth validation is supplied.

Use --zero-anchor-gradient with preset and input for physical methods only. It exports actual learned-parameter and input IFT chains including dg/dh times dh/dlogC;it excludes CV/source-refit differentiation. Thirdparty KROGER data are referenced by SHA and not copied into the bundle.
''')
    e.write(t.OUT/'delivery/export_complete.json',dict(created_utc=e.now(),presets=len(presets),selected_model_files=len(models),
        model_parameters_retrained=False,query_targets_in_bundle=False,goal_complete=False))
    print('Exported context predictor',len(presets),'presets',len(models),'models',flush=True)


if __name__=='__main__':main()
