"""Conditional, versioned Si complexes via primary-paper attachment energies.

No existing data/state is replaced. These are n-type single-charge templates,
not a computed full charge-transition diagram or a same-functional DFT set.
Binding-reference cancellation removes the energy/chemical-potential gauge;
it does not remove cross-method or multiplicity uncertainty.
"""
import numpy as np
from sandbox.kroger_db import load,DefectDB,N_SITE

PRIMARY_DOI='10.1063/1.5019938'
BINDING=np.array([-.94,-1.60,-1.83])


def pure_mask(db,ga,element,count,charge=None):
    dm=np.zeros(len(db.elements),int);dm[0]=ga
    if element is not None:dm[db.col(element)]=count
    mask=np.all(db.dm==dm,axis=1)
    if charge is not None:mask&=db.charge==charge
    return mask


def make_overlay(*,anchor='existing_pair',attachment_shift_eV=(0.,0.),prefactor_scale=1.,original=None):
    if anchor not in ('existing_pair','isolated_species'):raise ValueError('Declared binding-reference construction required')
    db=load() if original is None else original
    offsets=np.asarray(attachment_shift_eV,float);g=float(prefactor_scale)
    if offsets.shape!=(2,) or not np.isfinite(offsets).all() or not np.isfinite(g) or g<=0:
        raise ValueError('Two finite attachment shifts and positive statistical-weight scenario required')
    for k in [2,3]:
        if pure_mask(db,-k-1,'Si',k).any():raise ValueError('Do not duplicate already present pure-Si multi-donor complexes')
    def lowest(mask):
        indices=np.flatnonzero(mask)
        if len(indices)==0:raise ValueError('Required original anchor charge state missing')
        return int(indices[np.argmin(db.dHo[indices])])
    donor=lowest(pure_mask(db,-1,'Si',1,1));vacancy=lowest(pure_mask(db,-1,None,0,-3));pair=lowest(pure_mask(db,-2,'Si',1,-2))
    if anchor=='existing_pair':A1=float(db.dHo[pair])
    else:A1=float(db.dHo[vacancy]+db.dHo[donor]+BINDING[0])
    A2=A1+db.dHo[donor]+BINDING[1]-BINDING[0]+offsets[0]
    A3=A2+db.dHo[donor]+BINDING[2]-BINDING[1]+offsets[1]
    newdm=np.zeros((2,len(db.elements)),int);newdm[:,0]=[-3,-4];newdm[:,db.col('Si')]=[2,3]
    # Statistical capacities are explicitly conditional structural proxies:
    # use the smallest original pure-Sn same-stoichiometry prefactor, with
    # unit electronic/configurational degeneracy; they are not measured Si g.
    capacities=[]
    for k in [2,3]:
        m=pure_mask(db,-k-1,'Sn',k,k-3)
        capacities.append(float(min(db.prefactor[m])) if m.any() else N_SITE/(k+1))
    capacities=np.asarray(capacities)*g
    result=DefectDB(dHo=np.r_[db.dHo,A2,A3],charge=np.r_[db.charge,-1,0].astype(int),
        dm=np.r_[db.dm,newdm].astype(int),cs_ID=np.r_[db.cs_ID,db.nD+1,db.nD+2].astype(int),
        prefactor=np.r_[db.prefactor,capacities],sum_dm=np.r_[db.sum_dm,newdm.sum(axis=1)],
        elements=list(db.elements),nD=db.nD+2,nCS=db.nCS+2)
    meta=dict(primary_DOI=PRIMARY_DOI,primary_table='TableII; PDFpage4,printed046103-4; signed binding visually verified',
        primary_binding_eV=BINDING.tolist(),attachment_increment_eV=np.diff(BINDING).tolist(),
        anchor=anchor,attachment_shift_eV=offsets.tolist(),prefactor_scale=g,
        anchor_original_charge_rows=dict(donor=donor,vacancy=vacancy,pair=pair),
        anchor_original_parent_ids=dict(donor=int(db.cs_ID[donor]),vacancy=int(db.cs_ID[vacancy]),pair=int(db.cs_ID[pair])),
        anchor_original_intercepts_eV=dict(donor=float(db.dHo[donor]),vacancy=float(db.dHo[vacancy]),pair=float(db.dHo[pair])),
        existing_pair_binding_relative_isolated_eV=float(db.dHo[pair]-db.dHo[vacancy]-db.dHo[donor]),
        new_intercepts_eV=[float(A2),float(A3)],new_charges=[-1,0],new_dm=newdm.tolist(),
        new_prefactors_cm3=capacities.tolist(),new_parent_ids=[db.nD+1,db.nD+2],
        original_rows_unchanged=True,original_NPZ_unchanged=True,
        energetic_status='published attachment-difference transfer anchored to original energies;conditional cross-method hypothesis,not new DFT truth',
        charge_scope='only n-type charges explicitly used in primary table;other CTLs and validity away from n-type unavailable',
        statistical_scope='same-stoichiometry Sn site-capacity proxy,not measured Si orientation degeneracy or full shared-site exclusion',
        primary_counterevidence='paper reports charged VGa2Si more abundant than neutral VGa3Si due weak third attachment;neutral dominance is not assumed',
        material_prediction_validity_established=False)
    return result,meta
