"""Readonly access to frozen phase29 candidates and new selection definitions."""
from pathlib import Path
from itertools import combinations
import sys,json,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OLD_SCRIPTS=ROOT/'scripts/trainable_sandbox_20261017'
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(OLD_SCRIPTS))
import phase2_energy as e
import phase29_learning as learning
import phase29_data as data
import phase29_prediction_design as design
from csv_io_v2 import read_result_csv as read

BASE=ROOT/'results/predictive_redesign_20261019'
OUT=BASE/'phase30_selection'
OLD=data.OUT
METHODS=tuple(learning.METHODS)
POLICIES=('nH','rho_safe','chain_safe')
PRIMARY='JointProcess__chain_safe'


def document(path):return json.loads(Path(path).read_text())
def digest(value):return learning.digest(value)
def cases():return data.load_cases()
def roles():return data.load_roles()
def units():return learning.units()


def fit(train,method,config):return design.load_fit(train,method,config)


def group_score(rows,endpoint,*,paired=False,group_equal=True):
    """Missing/invalid rows fail the declared scope; do not improve coverage."""
    table=pd.DataFrame(rows)
    if paired:table=table[table.mobility_eligible.eq(True)]
    if table.empty:return None
    if not table.numeric.eq(True).all() or not table.valid.eq(True).all():return None
    obs={'nH':'observed','mu':'observed_log_mu','rho':'observed_log_sigma'}[endpoint]
    pred={'nH':'predicted','mu':'predicted_log_mu','rho':'predicted_log_sigma'}[endpoint]
    if not np.isfinite(table[[obs,pred]].to_numpy(float)).all():return None
    cond=table.assign(error=abs(table[pred]-table[obs])).groupby(
        ['related_cluster','family','sample_id'],sort=True).error.mean()
    perdoi=cond.groupby(level=['related_cluster','family']).mean()
    return float(perdoi.groupby(level='related_cluster').mean().mean()
        if group_equal else perdoi.mean())


def metrics(rows,*,group_equal=True):
    return {name:group_score(rows,name,paired=name!='nH',group_equal=group_equal)
        for name in ('nH','mu','rho')}


def finite(value):return value is not None and np.isfinite(value)


def safe(candidate,reference,policy):
    # This is an empirical inner-CV constraint, not a population guarantee.
    if not finite(candidate['nH']):return False
    targets=() if policy=='nH' else ('rho',) if policy=='rho_safe' else ('rho','mu')
    return all(finite(candidate[k]) and finite(reference[k]) and
        candidate[k]<=reference[k]+1e-10 for k in targets)


def select(candidates,reference,policy):
    accepted=[c for c in candidates if safe(c['metrics'],reference['metrics'],policy)]
    if policy!='nH':accepted.append(dict(reference,fallback_reference=True))
    if not accepted:
        return dict(reference,fallback_reference=True,reason='no qualified finite candidate')
    return min(accepted,key=lambda c:(round(c['metrics']['nH'],10),
        bool(c.get('fallback_reference',False)),*design.config_tie(c['config'])))


def write(path,value):e.write(Path(path),value)


def file_hashes(paths):
    return {str(Path(p).relative_to(ROOT)):e.sha(p) for p in paths}


def verify_hashes(value):
    for kind in ('code_hashes','input_hashes','evidence_sha256'):
        for name,h in value.get(kind,{}).items():
            assert e.sha(ROOT/name)==h,(kind,name)
