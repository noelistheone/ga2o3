"""Shared log-odds allocation composed with the frozen fixed-stock solver.

This module learns conditional endpoint speciation, not microscopic reaction
rates. Foreign-atom conservation and the charge/Hall equations are inherited
from the unchanged, audited InventoryCoherentModel.
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit
from .inventory_coherent_model import InventoryCoherentModel,InventoryPrediction

LN10=float(np.log(10.))


class LogOddsInventoryModel:
    def __init__(self,definition,*,profile,pO2_atm,db=None):
        self.definition=dict(definition);self.closure=definition['closure']
        if self.closure not in ('inventory','affine_output'):raise ValueError('Explicit supported closure required')
        inner_definition={k:v for k,v in definition.items() if k!='closure'}
        inner_definition.update(inventory_kind='complex' if self.closure=='inventory' else 'none',relation='constant')
        self.inner=InventoryCoherentModel(inner_definition,profile=profile,pO2_atm=pO2_atm,db=db)
        self.profile=self.inner.profile;self.pO2_atm=self.inner.pO2_atm
        self.layers=self.inner.layers;self.solvers=self.inner.solvers
        self.iE=self.inner.iE;self.iscreen=self.inner.iscreen
        self.nbase=len(self.inner.parameter_names)-(1 if self.closure=='inventory' else 0)
        self.ia=self.nbase;self.ib=self.ia+1
        suffix=('allocation_logit_a','allocation_logit_b') if self.closure=='inventory' else ('output_affine_c0','output_affine_c1')
        self.parameter_names=tuple(self.inner.parameter_names[:self.nbase])+suffix

    def layer_for(self,row):return self.inner.layer_for(row)

    def source_interval(self,parameters):
        if self.closure!='inventory':raise ValueError('Internal log-odds adaptation requires inventory closure')
        return -6.,6.

    def point(self,row,parameters,*,source_kind='none',source_shift=0.):
        w=np.asarray(parameters,float);z=float(source_shift);nt=len(self.parameter_names)
        if w.shape!=(nt,) or not np.isfinite(w).all():raise ValueError('Finite shared parameter vector required')
        if source_kind not in ('none','offset','logodds') or not np.isfinite(z):raise ValueError('Explicit supported source coordinate required')
        if source_kind=='none' and z!=0:raise ValueError('Nonzero source coordinate requires a kind')
        if source_kind=='logodds' and (self.closure!='inventory' or abs(z)>6.+1e-12):raise ValueError('Internal log-odds shift outside supported domain')
        C=float(row.incorporated_cm3)
        if not np.isfinite(C) or C<=0:raise ValueError('Positive actual chemical dose required')
        x=float(np.log10(C)-18.)
        if self.closure=='inventory':
            g=float(w[self.ia]+w[self.ib]*x+(LN10*z if source_kind=='logodds' else 0.))
            S=float(np.logaddexp(0.,g));alpha=float(expit(g))
            if not np.isfinite(S) or S>120.:raise ValueError('Outside finite allocation-coordinate domain')
            v=np.r_[w[:self.nbase],S]
            W=np.zeros((self.nbase+1,nt));W[:self.nbase,:self.nbase]=np.eye(self.nbase)
            W[-1,self.ia]=alpha;W[-1,self.ib]=x*alpha
        else:
            g=None;S=None;alpha=None;v=w[:self.nbase]
            W=np.zeros((self.nbase,nt));W[:self.nbase,:self.nbase]=np.eye(self.nbase)
        p=self.inner.point(row,v,source_kind='offset' if source_kind=='offset' else 'none',
            source_shift=z if source_kind=='offset' else 0.)
        values=p.values.copy();transport=p.transport_values.copy()
        J=p.jacobian@W;TJ=p.transport_jacobian@W
        IJ=p.input_jacobian.copy();ITJ=p.transport_input_jacobian.copy()
        SJ=p.source_jacobian.copy();STJ=p.transport_source_jacobian.copy()
        if self.closure=='inventory':
            IJ[:,0]+=p.jacobian[:,-1]*(w[self.ib]*alpha)
            ITJ[:,0]+=p.transport_jacobian[:,-1]*(w[self.ib]*alpha)
            if source_kind=='logodds':
                SJ=p.jacobian[:,-1]*(LN10*alpha)
                STJ=p.transport_jacobian[:,-1]*(LN10*alpha)
            correction=0.
        else:
            correction=float(w[self.ia]+w[self.ib]*x)
            values[0]+=correction;transport[1]+=correction
            J[0,self.ia]=1.;J[0,self.ib]=x
            TJ[1,self.ia]=1.;TJ[1,self.ib]=x
            IJ[0,0]+=w[self.ib];ITJ[1,0]+=w[self.ib]
        same=bool(p.diagnostics['same_physical_state_for_output'] and (self.closure=='inventory' or correction==0.))
        dg={**p.diagnostics,'learning_definition':self.definition,'source_kind':source_kind,'source_shift':z,
            'same_physical_state_for_output':same,'allocation_logit':g,'allocation_positive_coordinate':S,
            'logodds_alpha':alpha,'affine_output_correction_dex':correction,
            'predicted_log_mu_H':float(transport[0]),'predicted_log_sigma':float(transport[1]),
            'closure_scope':'conditional shared endpoint allocation;not a new microscopic rate or measured complex concentration'}
        return InventoryPrediction(values=values,jacobian=J,source_jacobian=SJ,physical_values=p.physical_values.copy(),
            diagnostics=dg,transport_values=transport,transport_jacobian=TJ,transport_source_jacobian=STJ,
            input_names=p.input_names,input_jacobian=IJ,transport_input_jacobian=ITJ)

    def counters(self):return self.inner.counters()
