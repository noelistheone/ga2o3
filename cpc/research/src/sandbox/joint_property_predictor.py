"""Saved phase29 research prediction with explicit condition/anchor budgets."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib
import numpy as np
from .joint_saved_model import SavedJointModel
from .context_saved_model import INPUT_NAMES
from .context_source import solve_context_source,solve_affine_context
from .kroger_db import NPZ

ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def linear_value(value):
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        answer=float(np.power(10.,float(value)))
    return answer if np.isfinite(answer) and answer>0 else None


class JointPropertyPredictor:
    """Conditional Si/Sn,300K predictor; fixed global weights, Hall anchors only.

    Multiple rows of one condition are input scenarios, not extra samples.
    A positive anchor budget requires the same scenario count per condition,
    matching the frozen per-DOI experiment. Results carry scenario diagnostics;
    scenario ranges are not confidence or credible intervals.
    """
    def __init__(self,bundle):
        self.bundle=Path(bundle).resolve()
        self.manifest=json.loads((self.bundle/'manifest.json').read_text())
        if self.manifest['version']!='phase29_joint_scattering_v1':raise ValueError('Unsupported research bundle')
        if sha(NPZ)!=self.manifest['physical_reference_sha256']:raise ValueError('Physical catalogue hash differs')
        for name,digest in self.manifest['code_hashes'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Bound implementation changed: '+name)
        self.models={}

    def model(self,preset):
        if preset not in self.manifest['presets']:raise ValueError('Unknown preset')
        spec=self.manifest['presets'][preset];relative=Path(spec['model_file'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Bundle-relative model file required')
        path=self.bundle/relative
        if sha(path)!=spec['model_sha256']:raise ValueError('Saved model hash differs')
        if not spec['serving_qualified']:raise ValueError('Saved training fit is unqualified;original refusal retained')
        if preset not in self.models:self.models[preset]=SavedJointModel(json.loads(path.read_text()))
        return self.models[preset],spec

    @staticmethod
    def design(conditions,anchors):
        if not isinstance(conditions,list) or not conditions:raise ValueError('Nonempty input-scenario list required')
        if not isinstance(anchors,list) or len(anchors)>2:raise ValueError('Exactly0/1/2 Hall-condition anchors permitted')
        nodes=[];identities=[];caseids=[];groups={}
        forbidden={'target','observed','observed_log_nH','observed_nH_cm3','mu_cm2Vs',
            'observed_log_mu','observed_log_sigma','carrier_cm3'}
        for i,item in enumerate(conditions):
            if not isinstance(item,dict) or forbidden.intersection(item):raise ValueError('Response labels belong only in the explicit anchor list')
            required=set(INPUT_NAMES)|{'condition_id','case_id'}
            if set(item)!=required:raise ValueError('Each scenario needs exactly condition_id,case_id and the declared model inputs')
            cid=item['condition_id'];case=item['case_id']
            if not isinstance(cid,str) or not cid or not isinstance(case,str) or not case:raise ValueError('Nonempty string condition/case IDs required')
            if case in caseids:raise ValueError('Duplicate input case ID')
            if not isinstance(item['H_known'],(bool,np.bool_)) or not isinstance(item['carbon_known'],(bool,np.bool_)):
                raise ValueError('Composition availability flags must be Boolean')
            numeric=[item[k] for k in ('incorporated_cm3','Tobs_K','H_cm3','carbon_cm3','growth_T_C','scenario_pO2_atm')]
            if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.integer,np.floating)) for v in numeric):
                raise ValueError('Numeric model inputs must be numbers')
            if not np.isfinite(np.asarray(numeric,float)).all():raise ValueError('Finite explicit input scenarios required')
            if not isinstance(item['oxygen_input_kind'],str):raise ValueError('Explicit oxygen-input status string required')
            row=SimpleNamespace(**{k:item[k] for k in INPUT_NAMES})
            if row.element not in ('Si','Sn') or row.Tobs_K!=300. or row.incorporated_cm3<=0 or row.scenario_pO2_atm<=0:
                raise ValueError('Declared positive-dose Si/Sn300K domain required')
            if row.H_cm3<0 or row.carbon_cm3<0 or bool(row.H_known)!=(row.H_cm3>0) or bool(row.carbon_known)!=(row.carbon_cm3>0):
                raise ValueError('Known positive composition or explicit unknown-zero scenario required')
            invariant=(row.element,row.incorporated_cm3,row.Tobs_K,row.H_known,row.H_cm3,row.carbon_known,row.carbon_cm3)
            if cid in groups and groups[cid]['invariant']!=invariant:raise ValueError('One condition cannot change its reported chemistry between scenarios')
            groups.setdefault(cid,dict(invariant=invariant,indices=[]))['indices'].append(i)
            nodes.append(row);identities.append(cid);caseids.append(case)
        names=[];observed={}
        for anchor in anchors:
            if not isinstance(anchor,dict) or set(anchor)!={'condition_id','observed_nH_cm3'}:
                raise ValueError('Each anchor is a condition_id and one Hall density only')
            cid=anchor['condition_id'];raw=anchor['observed_nH_cm3']
            if isinstance(raw,(bool,np.bool_)) or not isinstance(raw,(int,float,np.integer,np.floating)):
                raise ValueError('Hall anchor density must be a number')
            n=float(raw)
            if cid not in groups or cid in observed or not np.isfinite(n) or n<=0:
                raise ValueError('Distinct supplied anchor conditions with finite positive Hall labels required')
            names.append(cid);observed[cid]=float(np.log10(n))
        if anchors and len({len(g['indices']) for g in groups.values()})!=1:
            raise ValueError('Source adaptation requires equal scenario counts per condition')
        indices=[i for i,cid in enumerate(identities) if cid in observed]
        y=[observed[identities[i]] for i in indices]
        return nodes,identities,caseids,groups,names,indices,y

    def predict(self,*,preset,conditions,anchors=None):
        anchors=[] if anchors is None else anchors
        nodes,cids,caseids,groups,names,ai,y=self.design(conditions,anchors)
        mapper,spec=self.model(preset);kind=spec['source_kind'];budget=len(anchors)
        if kind not in ('offset','internal','affine'):raise ValueError('Unknown source policy')
        policy=spec['source_lambdas'][str(budget)];cache={};model_calls=0;errors=[]
        def points(z):
            nonlocal model_calls
            z=float(z)
            if z not in cache:
                base=points(0.) if kind=='offset' and z!=0. else None
                result=[]
                for i,row in enumerate(nodes):
                    try:
                        if base is not None:p=None if base[i] is None else mapper.offset_point(row,base[i],z,require_observation=False)
                        else:
                            model_calls+=1
                            p=mapper.point(row,source_kind='share' if kind=='internal' else 'offset' if kind=='offset' else 'none',
                                shift=z,require_observation=False)
                        result.append(p)
                    except (ValueError,RuntimeError,ArithmeticError,np.linalg.LinAlgError) as exc:
                        errors.append(dict(case_id=caseids[i],source_shift=z,error=repr(exc)));result.append(None)
                cache[z]=result
            return cache[z]
        def forward(z):
            p=points(z)
            return (np.array([v.log_nH if v is not None else np.nan for v in p]),
                np.array([v.source_derivative if v is not None else np.nan for v in p]),
                np.array([v is not None and v.valid for v in p]))
        meta=None;source_parameters=None;shift=0.;source_valid=True
        if budget==0 or policy is None:
            values=points(0.);source_parameters=[0.,0.] if kind=='affine' else [0.]
        elif kind=='affine':
            base=points(0.)
            if any(p is None or not p.valid for p in base):raise ValueError('Affine baseline fails the declared input domain')
            meta=solve_affine_context([p.log_nH for p in base],np.log10([r.incorporated_cm3 for r in nodes]),ai,y,[policy])
            selected=meta['solutions'][0];source_parameters=selected['source_parameters'];shift=None
            values=[mapper.offset_point(row,p,delta,require_observation=False)
                for row,p,delta in zip(nodes,base,selected['response_delta'])]
            source_valid=bool(selected['finite_response_bound_valid'])
        else:
            meta=solve_context_source(forward,ai,y,[policy],linear=kind=='offset',
                nodes=33,local_candidates=3,maxiter=80)
            selected=meta['solutions'][0]
            if not selected.get('valid'):raise ValueError('Frozen source policy has no qualified solution on the supplied design')
            shift=float(selected['raw_shift']);source_parameters=[shift];values=points(shift)
        all_valid=bool(source_valid and all(v is not None and v.valid for v in values))
        rows=[]
        for i,p in enumerate(values):
            record=dict(case_id=caseids[i],condition_id=cids[i],is_Hall_anchor=cids[i] in names,
                scenario_weight=1/len(groups[cids[i]]['indices']),
                numeric=bool(p is not None and np.isfinite([p.log_nH,p.log_muH,p.log_sigma]).all()),
                valid=all_valid and p is not None and p.valid)
            if p is not None:
                record.update(log10_nH_cm3=p.log_nH,log10_muH_cm2Vs=p.log_muH,log10_sigma_Scm=p.log_sigma,
                    nH_cm3=linear_value(p.log_nH),muH_cm2Vs=linear_value(p.log_muH),sigma_Scm=linear_value(p.log_sigma),
                    log10_mu_CT_cm2Vs=p.log_mu_CT,diagnostics=p.diagnostics)
            rows.append(record)
        summary=[]
        for cid,group in groups.items():
            present=[rows[i] for i in group['indices']]
            item=dict(condition_id=cid,scenario_count=len(present),is_Hall_anchor=cid in names,
                numeric=all(r['numeric'] for r in present),valid=all(r['valid'] for r in present))
            if item['numeric']:
                for field in ('log10_nH_cm3','log10_muH_cm2Vs','log10_sigma_Scm'):
                    v=np.array([r[field] for r in present])
                    item[field+'_scenario_mean']=float(v.mean())
                    item[field+'_scenario_range']=[float(v.min()),float(v.max())]
            summary.append(item)
        return clean(dict(preset=preset,variant=spec['variant'],context=spec['context'],budget=budget,
            source_kind=kind,source_lambda=policy,source_parameters=source_parameters,source_raw_shift=shift,
            physical_anchor_conditions=names,anchor_scenario_rows=len(ai),model_point_calls=model_calls,
            source_solver=meta,point_failures=errors,all_inputs_observation_valid=all_valid,
            rows=rows,condition_summaries=summary,global_model_retrained=False,
            query_labels_read=False,query_mu_used=False,
            scenario_range_is_calibrated_interval=False,material_Born_validity_established=False,
            property_scope='same trained relation and declared state/readout controls;conditional Si/Sn300K',
            physical_carbon_defects_included=False if mapper.physical else None,
            full_context_is_not_a_held_source_validation=spec['context']=='full',
            no_new_element_or_rare_earth_validation=True))
