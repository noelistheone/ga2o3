"""Reference-consistent conditional charge-energy learning.

The task supplies one energy reference and a known charge catalogue per parent.
It does not infer unseen structures or absolute formation energies of a new
dopant. Numeric published energies remain in the separately loaded database.
Pair crossings need not be stable thermodynamic charge-transition levels.
"""
from __future__ import annotations

from dataclasses import replace
from collections import Counter
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import ExtraTreesRegressor
from .kroger_db import DefectDB

# Basic periodic-table coordinates, not fitted energetic descriptors.
PERIODIC = {
    "H": (1, 1, 1), "Si": (14, 14, 3), "Fe": (26, 8, 4),
    "Sn": (50, 14, 5), "Cr": (24, 6, 4), "Ti": (22, 4, 4),
    "Ir": (77, 9, 6), "Mg": (12, 2, 3), "Ca": (20, 2, 4),
    "Zn": (30, 12, 4), "Co": (27, 9, 4), "Zr": (40, 4, 5),
    "Hf": (72, 4, 6), "Ta": (73, 5, 6), "Ge": (32, 14, 4),
    "Pt": (78, 10, 6), "Rh": (45, 9, 5),
}
FEATURE_NAMES = (
    "atomic_number", "periodic_group", "period",
    "dm_Ga", "dm_O", *(f"site_count_{i}" for i in range(9)),
    "q", "q_reference", "delta_q", "delta_q_squared",
)


def parent_reference(db, ids):
    """Known charge nearest zero; negative charge wins a nonzero tie."""
    return min(ids, key=lambda j: (abs(int(db.charge[j])), int(db.charge[j])))


def relative_charge_records(db: DefectDB, sites):
    """One secant crossing per nonreference state of a mono-single parent."""
    records, parents = [], []
    for pid in sorted(np.unique(db.cs_ID)):
        ids = np.flatnonzero(db.cs_ID == pid)
        if not np.all(db.dm[ids] == db.dm[ids[0]]):
            raise ValueError(f"Parent {pid}: atom composition changes with charge")
        if len(np.unique(db.charge[ids])) != len(ids):
            raise ValueError(f"Parent {pid}: repeated charge states")
        dm = db.dm[ids[0]]
        if (np.count_nonzero(dm[2:]) != 1 or np.abs(dm[2:]).sum() != 1
                or dm[2:].max() != 1):
            continue
        elem = db.elements[2 + int(np.argmax(dm[2:]))]
        ref = parent_reference(db, ids)
        isolated = bool(np.abs(dm[:2]).sum() <= 1)
        parents.append(dict(parent=int(pid), element=elem, reference_index=int(ref),
            q_reference=int(db.charge[ref]), charge_states=len(ids),
            hidden_labels=len(ids)-1, isolated=isolated,
            site_signature_constant=bool(np.all(sites[ids] == sites[ids[0]]))))
        for j in ids:
            if j == ref:
                continue
            dq = int(db.charge[j] - db.charge[ref])
            record = dict(label_id=f"{int(pid)}:{int(db.charge[j])}",
                index=int(j), parent=int(pid), element=elem,
                reference_index=int(ref), q=int(db.charge[j]),
                q_reference=int(db.charge[ref]), delta_q=dq, isolated=isolated,
                crossing_eV=float(-(db.dHo[j]-db.dHo[ref])/dq),
                reference_energy_eV=float(db.dHo[ref]),
                target_energy_eV=float(db.dHo[j]),
                dm_Ga=int(dm[0]), dm_O=int(dm[1]))
            record.update({f"site_count_{i}":int(x) for i,x in enumerate(sites[j])})
            records.append(record)
    return records, parents


def features(records):
    """No A_ref, parent identity or measured property occurs in these inputs."""
    return np.asarray([
        [*PERIODIC[r["element"]], r["dm_Ga"], r["dm_O"],
         *(r[f"site_count_{i}"] for i in range(9)),
         r["q"], r["q_reference"], r["delta_q"], r["delta_q"]**2]
        for r in records], dtype=float)


def group_weights(records):
    """Equal element, then equal parent, then equal charge label; mean one."""
    parent_counts = Counter((r["element"], r["parent"]) for r in records)
    elem_parents = {}
    for e,p in parent_counts:
        elem_parents.setdefault(e,set()).add(p)
    w = np.asarray([1/(len(elem_parents[r["element"]]) *
                          parent_counts[(r["element"],r["parent"])])
                    for r in records], float)
    return w/w.mean()


def macro_error(records, prediction, target="crossing_eV"):
    """Element-equal, parent-equal absolute error."""
    by = {}
    for r,p in zip(records,prediction):
        by.setdefault(r["element"],{}).setdefault(r["parent"],[]).append(
            abs(float(p)-float(r[target])))
    return float(np.mean([np.mean([np.mean(v) for v in d.values()])
                          for d in by.values()]))


class ChargeEnergyLearner:
    """Small-data learners with train-only scaling and explicit group weights."""
    def __init__(self, method, config=None):
        self.method = method
        self.config = {} if config is None else dict(config)

    def fit(self, records):
        X = features(records)
        y = np.asarray([r["crossing_eV"] for r in records], float)
        w = group_weights(records)
        self.mean = float(np.average(y, weights=w))
        self.center = np.average(X, axis=0, weights=w)
        self.scale = np.sqrt(np.average((X-self.center)**2, axis=0, weights=w))
        self.scale[self.scale < 1e-12] = 1.
        self.n_training = len(records)
        self.training_elements = sorted({r["element"] for r in records})
        if self.method == "global_mean":
            return self
        if self.method == "charge_pair_mean":
            # Each available pair mean uses the same within-element/parent weights.
            self.pair_means = {}
            for pair in sorted({(r["q_reference"],r["q"]) for r in records}):
                use = np.asarray([(r["q_reference"],r["q"])==pair for r in records])
                self.pair_means[pair] = float(np.average(y[use],weights=w[use]))
            return self
        if self.method == "ridge":
            self.model = Ridge(alpha=self.config["alpha"])
        elif self.method == "rbf":
            gamma = 1/(2*len(FEATURE_NAMES)*self.config["length"]**2)
            self.model = KernelRidge(alpha=self.config["alpha"],kernel="rbf",gamma=gamma)
        elif self.method == "extra_trees":
            self.model = ExtraTreesRegressor(
                n_estimators=64, max_depth=self.config["max_depth"],
                min_samples_leaf=self.config["min_samples_leaf"],
                random_state=1729, n_jobs=1)
        else:
            raise ValueError(self.method)
        self.model.fit((X-self.center)/self.scale,y-self.mean,sample_weight=w)
        return self

    def predict(self, records):
        if self.method == "global_mean":
            return np.full(len(records),self.mean)
        if self.method == "charge_pair_mean":
            return np.asarray([self.pair_means.get(
                (r["q_reference"],r["q"]),self.mean) for r in records])
        return self.mean+self.model.predict((features(records)-self.center)/self.scale)


def reconstruct(db, records, crossing_predictions):
    """Replace only declared hidden labels, supplying exact known A_ref.

    The caller must account separately for every retained database energy.
    No convexity/positive-U projection or arbitrary band-gap clipping is used.
    """
    A = db.dHo.copy()
    for r,x in zip(records,crossing_predictions):
        A[r["index"]] = db.dHo[r["reference_index"]]-r["delta_q"]*float(x)
    return replace(db, dHo=A)


def stable_crossings(db, ids, low=0., high=4.9):
    """Zero-K energetic lower-envelope transitions inside a stated EF interval.

    This ignores entropy/degeneracy; these are not finite-temperature fitted
    levels. All pair crossings are tested against the entire charge catalogue.
    """
    ids=list(ids)
    out=[]
    for a,i in enumerate(ids):
        for j in ids[a+1:]:
            dq=int(db.charge[i]-db.charge[j])
            if not dq:
                continue
            x=float((db.dHo[j]-db.dHo[i])/dq)
            if not low < x < high:
                continue
            levels=db.dHo[ids]+db.charge[ids]*x
            if abs(float(db.dHo[i]+db.charge[i]*x)-float(levels.min()))>1e-8:
                continue
            qa,qb=sorted((int(db.charge[i]),int(db.charge[j])),reverse=True)
            out.append(dict(EF_eV=x,high_q=qa,low_q=qb,multi_electron=abs(dq)>1))
    return sorted(out,key=lambda r:r["EF_eV"])
