"""Read public event logs as data; preserve headers, status labels and invalid values."""
from dataclasses import dataclass
import re
import numpy as np


@dataclass
class Monitor:
    header:str
    time:np.ndarray
    value:np.ndarray
    description:tuple
    valid:np.ndarray
    declared_range:tuple|None


def read_monitor(blob):
    lines=blob.decode('utf-8-sig').splitlines();lines=[l for l in lines if l.strip()]
    if not lines or not lines[0].startswith('Elapsed time [s]'):raise ValueError('Explicit elapsed-seconds header required')
    header=lines[0];parts=[l.split('\t') for l in lines[1:]]
    if not parts:raise ValueError('No event rows')
    t=np.array([float(r[0]) for r in parts]);v=np.array([float(r[1]) for r in parts]);desc=tuple(r[2].strip() if len(r)>2 else '' for r in parts)
    if not np.isfinite(t).all() or np.any(t<0) or np.any(np.diff(t)<0):raise ValueError('Finite nonnegative ordered event times required')
    limits=None;m=re.search(r'Value\s*\[([^\]]+)\]',header)
    if m:
        try:a,b=map(float,m[1].split(','));limits=(a,b)
        except ValueError:pass
    valid=np.isfinite(v)
    if limits is not None:valid&=(v>=limits[0])&(v<=limits[1])
    return Monitor(header,t,v,desc,valid,limits)


def at(m,t):
    index=np.searchsorted(m.time,np.asarray(t,float),side='right')-1
    valid=(index>=0);index=np.maximum(index,0)
    return m.value[index],valid&m.valid[index],index


def shutter_open(m,t):
    values,valid,index=at(m,t);state=[]
    mapping=re.search(r'Description\s*\[([^\]]+)\]',m.header)
    labels=tuple(mapping[1].split(',')) if mapping else ()
    for j,i in enumerate(index):
        label=m.description[i].strip().lower()
        if not label and values[j].is_integer() and 0<=values[j]<len(labels):label=labels[int(values[j])].strip().lower()
        if label=='open':state.append(True)
        elif label in ('close','closed'):state.append(False)
        else:state.append(False);valid[j]=False
    return np.array(state,bool),valid


def exposure_segments(monitors,end,conditions):
    selected=[monitors[k] for k in conditions]
    edges=np.unique(np.r_[0.,end,*[m.time[(m.time>0)&(m.time<end)] for m in selected]])
    left=edges[:-1];right=edges[1:];good=np.ones(len(left),bool);active=np.ones(len(left),bool)
    for name,expected in conditions.items():
        opened,valid=shutter_open(monitors[name],left);good&=valid;active&=opened==expected
    keep=good&active
    return np.c_[left[keep],right[keep]],float(np.sum((right-left)[~good]))


def duration_summary(m,segments):
    total=float(np.sum(np.diff(segments,axis=1))) if len(segments) else 0.;vals=[];weights=[];invalid=0.
    for start,end in segments:
        edges=np.r_[start,m.time[(m.time>start)&(m.time<end)],end];left=edges[:-1];dt=np.diff(edges)
        v,good,_=at(m,left);invalid+=float(np.sum(dt[~good]));vals.extend(v[good]);weights.extend(dt[good])
    v=np.asarray(vals);w=np.asarray(weights);valid=float(w.sum())
    return dict(duration_s=total,valid_duration_s=valid,invalid_duration_s=invalid,
        minimum=None if not len(v) else float(v.min()),maximum=None if not len(v) else float(v.max()),
        mean=None if not valid else float(w@v/valid),
        standard_deviation=None if not valid else float(np.sqrt(w@((v-w@v/valid)**2)/valid)),
        policy='last event held until next event or metadata end; no extrapolation before first event')
