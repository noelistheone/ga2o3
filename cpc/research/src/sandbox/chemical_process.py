"""Positive process-to-chemistry calibration with explicit concentration/flux conventions."""
from pathlib import Path
import json,hashlib
import numpy as np

KB=8.617333262145e-5
REFERENCE_K=1500.
KNOT_TSOURCE_C=np.array([1100.,1150.,1200.,1250.])
KNOT_Q=1/(KB*REFERENCE_K)-1/(KB*(KNOT_TSOURCE_C+273.15))


def source_coordinate(source_C):
    return 1/(KB*REFERENCE_K)-1/(KB*(np.asarray(source_C,float)+273.15))


def slope_basis(q):
    q=np.atleast_1d(q).astype(float);columns=[np.minimum(q,KNOT_Q[0])]
    columns += [np.clip(q-lo,0,hi-lo) for lo,hi in zip(KNOT_Q[:-1],KNOT_Q[1:])]
    columns += [np.maximum(q-KNOT_Q[-1],0.)]
    out=np.column_stack(columns)
    at_zero=np.r_[min(0.,KNOT_Q[0]),*[np.clip(-lo,0,hi-lo) for lo,hi in zip(KNOT_Q[:-1],KNOT_Q[1:])],max(-KNOT_Q[-1],0.)]
    return out-at_zero


def feature_matrix(method,source_C,velocity_um_h,rate_center):
    q=source_coordinate(source_C);r=np.log10(np.asarray(velocity_um_h,float));x=q[:,None] if method=='ArrheniusMassBalance' else slope_basis(q)
    parts=[np.ones((len(q),1)),x]
    if method in ('RegimeFlux','DirectProcessSpline'):parts.append((r-rate_center)[:,None])
    return np.column_stack(parts)


class ChemicalCalibration:
    def __init__(self,record):
        if record['version']!=1:raise ValueError('Unknown chemical calibration version')
        self.record=record;self.method=record['method'];self.coefficients=np.asarray(record['coefficients'],float)

    def predict(self,source_T_C,velocity_um_h):
        T,v=float(source_T_C),float(velocity_um_h)
        if not np.isfinite([T,v]).all() or T<=-273.15 or v<=0:raise ValueError('Positive absolute source temperature and growth rate required')
        X=feature_matrix(self.method,np.array([T]),np.array([v]),self.record['rate_center']);r=np.log10(v)
        y=float(X[0]@self.coefficients)-(r if self.method!='DirectProcessSpline' else 0.)
        slopes=self.coefficients[1:2] if self.method=='ArrheniusMassBalance' else self.coefficients[1:6]
        q=float(source_coordinate(T));s=float(slopes[0] if len(slopes)==1 else slopes[np.searchsorted(KNOT_Q,q,side='right')])
        dr=float(self.coefficients[-1]) if self.method in ('RegimeFlux','DirectProcessSpline') else 0.
        dr+=0. if self.method=='DirectProcessSpline' else -1.
        warnings=[]
        if not self.record['source_T_support_C'][0]<=T<=self.record['source_T_support_C'][1]:warnings.append('source-temperature extrapolation beyond chemical training settings')
        if not self.record['growth_rate_support_um_h'][0]<=v<=self.record['growth_rate_support_um_h'][1]:warnings.append('growth-rate extrapolation beyond two calibrated stacks')
        if not -300<y<300:raise ValueError('Concentration outside numerical exponent domain')
        return dict(log10_C_cm3=y,C_cm3=10.**y,dlog10C_dsourceC=s/(KB*(T+273.15)**2),dlog10C_dlog10v=dr,
            effective_energy_slope_eV=s*np.log(10.),effective_incorporated_flux_cm2s=10**y*v*1e-4/3600.,warnings=warnings,
            grade='learned Fig3 chemistry proxy for Hall inputs;not measured same-Hall chemistry or incidentbeam calibration',
            clock='no kinetic clock learned from steady layer interiors')


class SavedChemicalCalibrations:
    def __init__(self,bundle):
        self.bundle=Path(bundle).resolve();self.manifest=json.loads((self.bundle/'manifest.json').read_text());self.models={}
        if self.manifest['version']!=1:raise ValueError('Chemical bundle version mismatch')
        for r in self.manifest['files']:
            p=(self.bundle/r['path']).resolve()
            if self.bundle not in p.parents or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Chemical model checksum/path mismatch')

    def get(self,name):
        if name not in self.manifest['models']:raise ValueError('Unknown saved chemical model')
        if name not in self.models:
            self.models[name]=ChemicalCalibration(json.loads((self.bundle/self.manifest['models'][name]).read_text()))
        return self.models[name]
