"""Differentiable closed-atom frozen association extent, not a measured rate.

Effective reaction: (VGa-D)+2DGa -> (VGa-3D). Atom counts are exact.
Charges are re-equilibrated by the subsequent cold solver, never assumed
unchanged for parents with multiple charge states. Energies are not fitted.
"""
import numpy as np
from scipy.special import expit,logsumexp
from sandbox.fixed_inventory_charge import InventoryState


def associate_neutral(solver,inventory,psi,psi_jacobian,*,softmin_power=20.):
    c=np.asarray(inventory.totals,float);dc=np.asarray(inventory.jacobian,float)
    dpsi=np.asarray(psi_jacobian,float);psi=float(psi);p=float(softmin_power)
    if not np.isfinite(psi) or abs(psi)>80 or p<=0 or not np.isfinite(p):raise ValueError('Declared finite association coordinate/power required')
    if dc.ndim!=2 or dc.shape[0]!=len(c) or dpsi.shape!=(dc.shape[1],) or not np.isfinite(dpsi).all():
        raise ValueError('Matching complete inventory and association Jacobians required')
    layer=solver.layer;dcol=layer.db.col(layer.donor_element);dm=layer.db.dm[layer.first]
    pair_dm=np.zeros(len(layer.db.elements),int);pair_dm[0]=-2;pair_dm[dcol]=1
    dest_dm=np.zeros_like(pair_dm);dest_dm[0]=-4;dest_dm[dcol]=3
    pair=np.all(dm==pair_dm,axis=1);donor=solver.receivers.copy()
    candidates=[]
    for i in np.flatnonzero(np.all(dm==dest_dm,axis=1)):
        ix=np.flatnonzero((layer.parent==i)&(layer.q==0))
        if len(ix):candidates.append((float(layer.db.dHo[ix[0]]),int(i)))
    if not candidates or not pair.any() or not donor.any():raise ValueError('Required original-pair/donor/neutral-triple parents unavailable')
    dest=min(candidates)[1]
    foreign=np.any(solver.parent_atoms>0,axis=1)
    caps=layer.db.prefactor[layer.first]*np.where(foreign,.1*(1-1e-6),1.)
    P=float(c[pair].sum());D=float(c[donor].sum());headroom=float(caps[dest]-c[dest])
    bounds=np.array([P,D/2,headroom]);dbounds=np.stack([dc[pair].sum(axis=0),dc[donor].sum(axis=0)/2,-dc[dest]])
    if not np.isfinite(bounds).all() or np.any(bounds<=0):raise ValueError('Strictly positive reactant stocks and product headroom required')
    logs=np.log(bounds);normal=float(logsumexp(-p*logs));weights=np.exp(-p*logs-normal)
    M=float(np.exp(-normal/p));dM=M*np.sum(weights[:,None]*dbounds/bounds[:,None],axis=0)
    s=float(expit(psi));complement=float(expit(-psi));extent=M*s
    dextent=s*dM+M*s*complement*dpsi
    out=c.copy();J=dc.copy()
    for mask,multiple,total,dtotal in [(pair,1.,P,dc[pair].sum(axis=0)),(donor,2.,D,dc[donor].sum(axis=0))]:
        ratio=multiple*extent/total;dr=(multiple*dextent-ratio*dtotal)/total
        out[mask]=c[mask]*(1-ratio);J[mask]=dc[mask]*(1-ratio)-c[mask,None]*dr[None,:]
    out[dest]+=extent;J[dest]+=dextent
    if np.any(out<=0) or np.any(out>=caps) or not np.isfinite(J).all():raise ValueError('Association positivity/cap domain failed')
    delta=out-c;dDelta=J-dc
    balance=dm.T@delta;dbalance=dm.T@dDelta
    massscale=np.maximum(abs(dm).T@(abs(out)+abs(c)),1.)
    derivative_scale=np.maximum(abs(dm).T@(abs(J)+abs(dc)),1.)
    error=float(max(abs(balance)/massscale));derror=float(np.max(abs(dbalance)/derivative_scale))
    if error>2e-12 or derror>2e-10:raise ValueError('Association full atom/derivative conservation failed')
    result=InventoryState(out,J,dict(inventory.diagnostics,
        neutral_association_psi=psi,neutral_association_fraction=s,neutral_association_extent_cm3=extent,
        neutral_association_smooth_capacity_cm3=M,neutral_association_softmin_power=p,
        neutral_association_product_parent=int(dest),neutral_association_reactant_bounds_cm3=bounds.tolist(),
        association_atom_conservation_relative_error=error,association_atom_derivative_relative_error=derror,
        association_scope='learned frozen atomic partition;not measured microscopic rate/equilibrium temperature',
        association_closed_Ga_O_and_all_foreign=True,
        association_charge_re_equilibrated_after_atomic_update=True,
        association_does_not_assume_parent_charge_or_free_n_unchanged=True,
        association_formation_energies_changed=False))
    solver.last_inventory=result
    return result
