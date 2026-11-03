"""Absolute dopant-atom share coordinates for the existing frozen parent pools.

This is an effective open-Ga inventory map, not an elementary reaction. Its
state set agrees with unbounded relative odds; finite learned function classes
and priors need not agree. No experimental labels are consumed here.
"""
import numpy as np
from scipy.special import expit
from .fixed_inventory_charge import InventoryState
from .implicit_defects import ImplicitDomainError, LN10


def allocate_absolute_share(solver, pars, chain, psi, psi_jacobian):
    chain=np.asarray(chain,float);psi=float(psi);dpsi=np.asarray(psi_jacobian,float)
    if not np.isfinite(psi) or abs(psi)>80:
        raise ValueError('Finite absolute logit within the declared numerical interval [-80,80] required')
    if chain.ndim!=2 or chain.shape[0]!=solver.layer.np or dpsi.shape!=(chain.shape[1],):
        raise ValueError('Explicit compatible reference and logit derivative chains required')
    ref=solver.reference(pars);c0=ref.totals;dc0=ref.jacobian@chain
    cx=solver.complex;iso=solver.receivers;k=solver.dopant_count
    U=float(k[cx]@c0[cx]);V=float(c0[iso].sum());W=U+V
    if not np.isfinite([U,V,W]).all() or min(U,V)<=0:
        raise ImplicitDomainError('Positive reference dopant-atom pools required')
    dU=k[cx]@dc0[cx];dV=dc0[iso].sum(axis=0);dW=dU+dV
    fi=float(expit(psi));fc=float(expit(-psi));df=fi*fc*dpsi
    # Normalize within each pool first; W/V itself can be enormous when the
    # reference isolated population is exponentially small.
    rc=c0[cx]/U;ri=c0[iso]/V
    drc=(dc0[cx]-rc[:,None]*dU)/U
    dri=(dc0[iso]-ri[:,None]*dV)/V
    c=c0.copy();dc=dc0.copy()
    c[cx]=W*fc*rc;dc[cx]=rc[:,None]*(fc*dW-W*df)+W*fc*drc
    c[iso]=W*fi*ri;dc[iso]=ri[:,None]*(fi*dW+W*df)+W*fi*dri
    layer=solver.layer;_,_,_,doses=layer._unpack(pars)
    achieved=solver.parent_atoms.T@c
    expected=doses[:,None]*LN10*chain[2+layer.b:2+layer.b+layer.k]
    atom_error=float(np.max(abs(achieved/doses-1)))
    derivative_error=float(np.max(abs(solver.parent_atoms.T@dc-expected)/(1+doses[:,None]*LN10)))
    occupancy=c/layer.db.prefactor[layer.first];foreign=np.any(solver.parent_atoms>0,axis=1)
    if np.any(c<0) or not np.isfinite(c).all() or not np.isfinite(dc).all():
        raise ImplicitDomainError('Nonpositive/nonfinite absolute-share inventory or derivative')
    if atom_error>3e-8 or derivative_error>3e-7:
        raise ImplicitDomainError('Absolute-share foreign-atom or derivative budget failed')
    if occupancy.max()>1 or occupancy[foreign].max()>=.1*(1-1e-6):
        raise ImplicitDomainError('Absolute-share parent dilution domain failed')
    host=layer.db.dm[layer.first,:2].T@(c-c0)
    if abs(host[1])>max(W,1)*1e-12:
        raise ImplicitDomainError('Unexpected oxygen exchange in pure-dopant pool allocation')
    result=InventoryState(c,dc,dict(**ref.diagnostics,absolute_share_logit=psi,
        isolated_atom_fraction_before=V/W,isolated_atom_fraction_after=fi,complex_atom_fraction_after=fc,
        complex_reference_atom_cm3=U,isolated_reference_atom_cm3=V,
        equivalent_relative_log10_shift=float((psi-np.log(V/U))/LN10),
        foreign_atom_relative_error=atom_error,foreign_atom_derivative_error=derivative_error,
        max_parent_fraction=float(occupancy.max()),native_inventory_changed=False,
        host_Ga_O_inventory_change_cm3=host.tolist(),host_atom_exchange_scope='openGa endpoint;O/foreign conserved',
        allocation_interpretation='absolute dopant-atom share;not measured kinetics or an independent microscopic mechanism'))
    solver.last_inventory=result
    return result
