"""Relative-entropy projection of a frozen dopant-atom inventory.

Only the pure-dopant vacancy-complex/isolated pools are reallocated. Foreign
atoms are conserved, Ga is open, and native inventories remain fixed. This is
a learned endpoint closure, not an elementary reaction or kinetic law.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit
from .fixed_inventory_charge import InventoryState
from .implicit_defects import ImplicitDomainError,LN10


def allocate_relative_odds(solver,pars,chain,log10_shift,shift_jacobian):
    chain=np.asarray(chain,float);h=float(log10_shift);dh=np.asarray(shift_jacobian,float)
    if not np.isfinite(h) or abs(h)>12.+1e-12:raise ValueError('Declared log10 inventory odds interval is [-12,12]')
    if chain.ndim!=2 or chain.shape[0]!=solver.layer.np or dh.shape!=(chain.shape[1],):
        raise ValueError('Explicit parameter and allocation derivative matrices required')
    ref=solver.reference(pars);c0=ref.totals;dc0=ref.jacobian@chain
    cx=solver.complex;iso=solver.receivers;k=solver.dopant_count
    U=float(k[cx]@c0[cx]);V=float(c0[iso].sum());W=U+V
    if not np.isfinite([U,V,W]).all() or min(U,V)<=0:raise ImplicitDomainError('Positive complex/isolated reference pools required')
    dU=k[cx]@dc0[cx];dV=dc0[iso].sum(axis=0);dW=dU+dV
    ell=np.log(V)-np.log(U)+LN10*h;fi=float(expit(ell));fc=float(expit(-ell))
    dell=dV/V-dU/U+LN10*dh;response=fi*fc*dell
    sc=W*fc/U;si=W*fi/V
    dsc=sc*(dW/W-dU/U)-W/U*response
    dsi=si*(dW/W-dV/V)+W/V*response
    c=c0.copy();dc=dc0.copy()
    c[cx]=sc*c0[cx];dc[cx]=sc*dc0[cx]+c0[cx,None]*dsc
    c[iso]=si*c0[iso];dc[iso]=si*dc0[iso]+c0[iso,None]*dsi
    l=solver.layer;p,_,_,doses=l._unpack(pars)
    achieved=solver.parent_atoms.T@c
    expected=doses[:,None]*LN10*chain[2+l.b:2+l.b+l.k]
    atom_error=float(np.max(np.abs(achieved/doses-1)))
    derivative_error=float(np.max(np.abs(solver.parent_atoms.T@dc-expected)/(1+doses[:,None]*LN10)))
    occ=c/l.db.prefactor[l.first];foreign=np.any(solver.parent_atoms>0,axis=1)
    if np.any(c<0) or not np.isfinite(c).all() or not np.isfinite(dc).all():raise ImplicitDomainError('Invalid projected inventory')
    if atom_error>3e-8 or derivative_error>3e-7:raise ImplicitDomainError('Foreign atom/derivative budget failed')
    if occ.max()>1 or occ[foreign].max()>=.1*(1-1e-6):raise ImplicitDomainError('Projected parent dilution domain failed')
    host=l.db.dm[l.first,:2].T@(c-c0)
    if abs(host[1])>max(W,1.)*1e-12:raise ImplicitDomainError('Unexpected O exchange in pure dopant reallocation')
    before=V/W
    # Binary KL equals the full atom-share KL because within-class ratios stay
    # fixed. It is an information metric, not measured free-energy production.
    KL=fi*(np.log(fi)-np.log(V/W)) if fi>0 else 0.
    KL+=fc*(np.log(fc)-np.log(U/W)) if fc>0 else 0.
    result=InventoryState(c,dc,dict(**ref.diagnostics,
        relative_inventory_log10_shift=h,isolated_atom_fraction_before=before,
        isolated_atom_fraction_after=fi,complex_atom_fraction_after=fc,
        atom_share_KL=float(KL),complex_reference_atom_cm3=U,isolated_reference_atom_cm3=V,
        foreign_atom_relative_error=atom_error,foreign_atom_derivative_error=derivative_error,
        max_parent_fraction=float(occ.max()),native_inventory_changed=False,
        host_Ga_O_inventory_change_cm3=host.tolist(),host_atom_exchange_scope='openGa endpoint;O andforeign atoms conserved',
        allocation_interpretation='atom-share relative-entropy tilt;not thermodynamic entropy or measured kinetics'))
    solver.last_inventory=result;return result
