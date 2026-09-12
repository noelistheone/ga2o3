"""Phase 60 V60-DCWM — defect-chemistry world-model transition generator + state featurizer.

This is Stage-0 of the V60-DCWM design (docs/phase60_dcwm_jepa_design.md). It provides:

  1. A closed-form, DUAL-TARGET defect-chemistry simulator:
       - measured_log_vo(...)  — the *measured-proxy* V_O (XPS scale; empirical dopant sign
         acceptor->-, donor->+; matches the experimental `vacancy_concentration` column and the
         deployed BrouwerHeadVC). This is what within-DOI data robustly shows (Mg rho ~ -0.9).
       - bulk_log_vo(...)      — the *bulk-equilibrium* [V_O^++] (thermodynamic sign acceptor->+,
         donor->-, Fermi-level mediated). For the second readout head + inverse design.
     Both share the closed-form Brouwer T / pO2 structure (Rule 2).

  2. A `StateFeaturizer` that turns a synthesis recipe into a fixed-length feature vector, grounded
     with on-disk FROZEN signals (detached at use time, Rule 1):
       - element 11-d physico-chemical descriptor (src/data/element_descriptors.py)
       - continuous [log10 c, T_K/1000, log10 pO2]
       - method one-hot (6) + device-class one-hot (3)
       - phi_DFT (16-d) from data/processed/dft_features_cache_v58.npz (NaN-imputed)
       - MACE-MP-0 projection (32-d) from data/processed/mace_mp_features.npz (fixed seeded proj)

  3. A `TransitionGenerator` that samples UNLIMITED (state, action, next_state) triples by applying a
     synthesis intervention (action) to a base recipe; the physics labels for s and s' are computed by
     the simulator above. This is the data the world model's latent dynamics are trained on (Stage-A).

  4. Helpers to featurize the real experimental CSV rows with the SAME schema (Stage-B), and to extract
     within-DOI experimental concentration transitions (optional Stage-A signal; OFF by default for the
     clean go/no-go per the GNG criterion).

All math is verified against src/data/kroger_synthetic.py + src/models/brouwer_head.py + the
dft_features_cache_v58 metadata. No external API, no network, all data in the project tree.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.data.element_descriptors import ELEMENT_DESCRIPTORS, DESCRIPTOR_DIM

PROJ = Path(__file__).resolve().parents[2]

# ── physical constants (match brouwer_head.py / kroger_synthetic.py) ──────────
KB_EV = 8.617333262e-5
LN10 = math.log(10.0)
EG_GA2O3 = 4.85
N_O_SITES = 2.85e22
LOG_PREFACTOR = math.log10(N_O_SITES)        # ~22.455
C_REF = 1e-2                                  # 1 at% reference
E_F_BASELINE = 2.0                            # eV, deployed BrouwerHeadVC baseline (Rule 2)

# Atmosphere bins shared with dft_features_cache_v58 (keys "elem|atm|T_C").
ATM_KEYS = ["Ar", "Ar_O2_4_1", "Ar_O2_1_1", "O2"]
ATM_PO2 = {"Ar": 1e-5, "Ar_O2_4_1": 0.05, "Ar_O2_1_1": 0.2, "O2": 1.0}
T_C_GRID = [500, 700, 900, 1100]
N_METHODS = 6                                 # 0 sputter,1 pld,2 cvd/ald,3 wet,4 evap,5 other
N_DEVICE = 3                                  # 0 MSM, 1 heterojunction, 2 unknown

# Empirical (measured-proxy) dopant offset: + raises measured V_O, - suppresses it.
# Verbatim from kroger_synthetic.DOPANT_OFFSET (and dft_features_cache_v58 provenance).
DOPANT_OFFSET_MEASURED: dict[str, float] = {
    "Mg": -0.95, "Zn": -0.90, "Cu": -0.85, "Ni": -0.85,
    "Al": +0.05, "Fe": +0.10, "B": +0.05, "V": +0.10, "Er": +0.05,
    "Eu": +0.05, "Cr": +0.10, "Sb": +0.10, "Bi": +0.10,
    "Si": +0.95, "Sn": +0.95, "Ti": +0.80, "Ge": +0.85,
    "Ta": +0.95, "W": +0.95,
}
# Carrier sign for the BULK (equilibrium) Fermi shift: acceptor lowers E_F (-> more bulk V_O),
# donor raises E_F (-> less bulk V_O), isovalent ~0.  +1 donor, -1 acceptor, 0 isovalent.
CARRIER_SIGN: dict[str, float] = {
    "Mg": -1, "Zn": -1, "Cu": -1, "Ni": -1, "N": -1, "F": -1,
    "Al": 0, "Fe": 0, "B": 0, "Er": 0, "Eu": 0, "Cr": 0, "In": 0,
    "Sb": +1, "Bi": +1, "V": +1, "Si": +1, "Sn": +1, "Ti": +1, "Ge": +1,
    "Ta": +1, "W": +1, "H": +1, "Zr": +1,
}
# Elements the synthetic generator samples over (covered by descriptors + offsets + DFT cache).
GEN_ELEMENTS = ["Mg", "Zn", "Cu", "Al", "Fe", "B", "V", "Cr", "Sb", "Bi",
                "Si", "Sn", "Ti", "Ge", "Ta", "W", "undoped"]


def _offset(elem: str) -> float:
    return DOPANT_OFFSET_MEASURED.get(elem, 0.0)


def _carrier(elem: str) -> float:
    return CARRIER_SIGN.get(elem, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Dual-target closed-form simulator
# ──────────────────────────────────────────────────────────────────────────────
def measured_log_vo(elem: str, c_frac: float, T_K: float, log_pO2: float) -> float:
    """Measured-proxy log10[V_O] (cm^-3). Matches deployed BrouwerHeadVC with E_f=2.0 and the
    EMPIRICAL dopant sign (acceptor suppresses, donor enhances) -> consistent with the experimental
    `vacancy_concentration` within-DOI trend (Mg rho ~ -0.9)."""
    boltz = -E_F_BASELINE / (KB_EV * max(T_K, 100.0) * LN10)
    dop = _offset(elem) * math.log10(max(c_frac, 1e-6) / C_REF)
    val = LOG_PREFACTOR + boltz - 0.5 * log_pO2 + dop
    return float(min(LOG_PREFACTOR, max(9.0, val)))


PDR_BASE = 3.5          # ~ experimental log10(PDR) mean
EA_PDR = 0.05           # gentle Arrhenius dark-current term (avoids huge T-domination)


def measured_log_pdr(elem: str, c_frac: float, T_K: float, log_pO2: float) -> float:
    """Measured-proxy log10(PDR). PDR anti-correlates with V_O trap density, so the dopant sign is the
    FLIP of the V_O measured sign (acceptor->+PDR, donor->-PDR) — matches eval_counterfactual_sweep's
    PDR convention. (PDR is genuinely device-dependent; this is the device-agnostic default.)"""
    ea_term = EA_PDR * (1.0 / (KB_EV * max(T_K, 100.0) * LN10) - 1.0 / (KB_EV * 973.15 * LN10))
    dop = (-_offset(elem)) * math.log10(max(c_frac, 1e-6) / C_REF)
    return float(min(14.0, max(-1.0, PDR_BASE + ea_term + dop)))


def bulk_log_pdr(elem: str, c_frac: float, T_K: float, log_pO2: float, kappa: float = 0.15) -> float:
    """Bulk PDR = flip of bulk V_O dopant response (acceptor->-PDR via more bulk V_O; donor->+PDR)."""
    log_c = math.log10(max(c_frac, 1e-6) / C_REF)
    ea_term = EA_PDR * (1.0 / (KB_EV * max(T_K, 100.0) * LN10) - 1.0 / (KB_EV * 973.15 * LN10))
    dop = (-2.0 * kappa * _carrier(elem)) * log_c          # flip of bulk V_O Fermi response
    return float(min(14.0, max(-1.0, PDR_BASE + ea_term + dop)))


def bulk_log_vo(elem: str, c_frac: float, T_K: float, log_pO2: float,
                kappa: float = 0.15) -> float:
    """Bulk-equilibrium log10[V_O^++] (cm^-3). Thermodynamic, Fermi-mediated:
       acceptor -> E_F down -> E_f(V_O^++)=E_f0+2 dE_F down -> [V_O] UP  (self-compensation)
       donor    -> E_F up   -> E_f up                                  -> [V_O] DOWN
    E_F shift = kappa * carrier_sign * log10(c/c_ref) (clamped to gap); q=+2.
    Distinct curve from the measured head (not a mirror): magnitude set by 2*dE_F/(kT ln10)."""
    log_c = math.log10(max(c_frac, 1e-6) / C_REF)
    dE_F = max(-EG_GA2O3 / 2, min(EG_GA2O3 / 2, kappa * _carrier(elem) * log_c))
    E_f_bulk = E_F_BASELINE + 2.0 * dE_F      # q=+2 formation energy rises with E_F
    boltz = -E_f_bulk / (KB_EV * max(T_K, 100.0) * LN10)
    val = LOG_PREFACTOR + boltz - 0.5 * log_pO2
    return float(min(LOG_PREFACTOR, max(9.0, val)))


# ──────────────────────────────────────────────────────────────────────────────
# string -> categorical helpers (shared by synthetic + experimental featurization)
# ──────────────────────────────────────────────────────────────────────────────
def method_to_idx(method: str) -> int:
    m = (method or "").lower()
    if "sputter" in m:
        return 0
    if "pld" in m or "pulsed laser" in m:
        return 1
    if "mocvd" in m or "cvd" in m or "ald" in m or "mist" in m or "epitax" in m:
        return 2
    if "hydro" in m or "sol" in m or "spin" in m or "spray" in m or "wet" in m or "chemical bath" in m:
        return 3
    if "evap" in m or "e-beam" in m or "ebeam" in m or "thermal" in m or "sublim" in m:
        return 4
    return 5


def atmosphere_to_key(atm: str) -> str:
    a = (atm or "").lower()
    has_o2 = ("o2" in a) or ("oxygen" in a) or ("air" in a)
    has_inert = ("ar" in a) or ("n2" in a) or ("argon" in a) or ("nitrogen" in a) or ("vac" in a)
    if "air" in a:
        return "Ar_O2_1_1"          # air pO2 ~0.21 ~ the 0.2 bin
    if has_o2 and not has_inert:
        return "O2"
    if has_o2 and has_inert:
        # crude ratio parse: "ar:o2=4:1" / "ar:o2=1:1"
        if "4:1" in a or "4_1" in a or "9:1" in a or "3:1" in a:
            return "Ar_O2_4_1"
        return "Ar_O2_1_1"
    return "Ar"                      # inert / vacuum / unspecified -> lowest pO2 bin


def device_to_idx(text: str) -> int:
    t = (text or "").lower()
    if any(k in t for k in ("heterojunc", "p-n", "pn junction", "nio", "self-powered", "self powered",
                            "schottky")):
        return 1
    if any(k in t for k in ("msm", "interdigit", "photoconduct", "metal-semiconductor")):
        return 0
    return 2


def _nearest_T_C(T_C: float) -> int:
    return min(T_C_GRID, key=lambda g: abs(g - T_C))


# ──────────────────────────────────────────────────────────────────────────────
# 2. State featurizer (frozen DFT + MACE grounding)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class StateFeaturizer:
    """Recipe -> fixed-length feature vector. Loads frozen caches once."""
    mace_proj_dim: int = 32
    use_z_expert: bool = False
    _phi_dft: dict = field(default_factory=dict, init=False)
    _phi_dft_elemmean: dict = field(default_factory=dict, init=False)
    _phi_dft_dim: int = field(default=16, init=False)
    _mace_elems: list = field(default_factory=list, init=False)
    _mace_feat: dict = field(default_factory=dict, init=False)       # elem -> 32-d (frozen proj)
    _mace_mean: np.ndarray = field(default=None, init=False)
    _zexp: dict = field(default_factory=dict, init=False)
    _zexp_dim: int = field(default=16, init=False)

    def __post_init__(self):
        self._load_dft()
        self._load_mace()
        if self.use_z_expert:
            self._load_zexpert()

    # --- frozen caches ---
    def _load_dft(self):
        d = np.load(PROJ / "data/processed/dft_features_cache_v58.npz", allow_pickle=True)
        keys = [str(k) for k in d["keys"]]
        feats = np.nan_to_num(np.asarray(d["features"], dtype=np.float64), nan=0.0)
        self._phi_dft_dim = feats.shape[1]
        per_elem = {}
        for k, f in zip(keys, feats):
            self._phi_dft[k] = f
            el = k.split("|")[0]
            per_elem.setdefault(el, []).append(f)
        self._phi_dft_elemmean = {e: np.mean(v, axis=0) for e, v in per_elem.items()}
        self._phi_dft_global = feats.mean(axis=0)

    def _load_mace(self):
        m = np.load(PROJ / "data/processed/mace_mp_features.npz", allow_pickle=True)
        self._mace_elems = [str(e) for e in m["elements"]]
        raw = np.asarray(m["mean_dop"], dtype=np.float64)          # [17, 256] dopant-site node feats
        # standardize columns, then a FIXED seeded Gaussian projection 256 -> mace_proj_dim (frozen).
        mu, sd = raw.mean(0, keepdims=True), raw.std(0, keepdims=True) + 1e-8
        std = (raw - mu) / sd
        rng = np.random.default_rng(60)                            # deterministic projection
        P = rng.standard_normal((raw.shape[1], self.mace_proj_dim)) / math.sqrt(raw.shape[1])
        proj = std @ P                                             # [17, mace_proj_dim]
        self._mace_feat = {e: proj[i] for i, e in enumerate(self._mace_elems)}
        self._mace_mean = proj.mean(0)

    def _load_zexpert(self):
        z = np.load(PROJ / "data/processed/z_expert_v59.npz", allow_pickle=True)
        keys = [str(k) for k in z["keys"]]
        feats = np.asarray(z["features"], dtype=np.float64)
        self._zexp_dim = feats.shape[1]
        per_elem = {}
        for k, f in zip(keys, feats):
            per_elem.setdefault(k.split("|")[0], []).append(f)
        self._zexp = {e: np.mean(v, axis=0) for e, v in per_elem.items()}
        self._zexp_mean = feats.mean(axis=0)

    # --- dimension ---
    @property
    def dim(self) -> int:
        d = DESCRIPTOR_DIM + 3 + N_METHODS + N_DEVICE + self._phi_dft_dim + self.mace_proj_dim
        if self.use_z_expert:
            d += self._zexp_dim
        return d

    def _elem_desc(self, elem: str) -> np.ndarray:
        key = elem if elem in ELEMENT_DESCRIPTORS else ("undoped" if elem in ("undoped", "—", "", None)
                                                        else "other")
        return np.asarray(ELEMENT_DESCRIPTORS[key], dtype=np.float64)

    def _phi(self, elem: str, atm_key: str, T_C: float) -> np.ndarray:
        k = f"{elem}|{atm_key}|{_nearest_T_C(T_C)}"
        if k in self._phi_dft:
            return self._phi_dft[k]
        if elem in self._phi_dft_elemmean:
            return self._phi_dft_elemmean[elem]
        return self._phi_dft_global

    def _mace(self, elem: str) -> np.ndarray:
        return self._mace_feat.get(elem, self._mace_mean)

    def featurize(self, elem: str, c_frac: float, T_C: float, atm_key: str,
                  method_idx: int, device_idx: int = 2) -> np.ndarray:
        cont = np.array([
            math.log10(max(c_frac, 1e-6)),          # log10 atomic fraction (~ -4..-1)
            (T_C + 273.15) / 1000.0,                # T_K/1000
            math.log10(ATM_PO2.get(atm_key, 1e-5)), # log10 pO2
        ], dtype=np.float64)
        meth = np.zeros(N_METHODS); meth[int(method_idx) % N_METHODS] = 1.0
        dev = np.zeros(N_DEVICE); dev[int(device_idx) % N_DEVICE] = 1.0
        parts = [self._elem_desc(elem), cont, meth, dev, self._phi(elem, atm_key, T_C), self._mace(elem)]
        if self.use_z_expert:
            parts.append(self._zexp.get(elem, self._zexp_mean))
        vec = np.concatenate(parts).astype(np.float32)
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)   # never emit NaN/inf features


# ──────────────────────────────────────────────────────────────────────────────
# 3. Action representation + transition generator
# ──────────────────────────────────────────────────────────────────────────────
# Action vector layout (delta-from-source; "target onehot + changed flag" for categoricals):
#   [0]    d_log10c
#   [1]    d_T_K / 1000
#   [2]    d_log10 pO2
#   [3:9]  target method onehot (zeros if unchanged)        (6)
#   [9]    method_changed_flag
#   [10:21] target element 11-d descriptor (zeros if unchanged)
#   [21]   element_changed_flag
#   [22:25] target device onehot (zeros if unchanged)       (3)
#   [25]   device_changed_flag
ACTION_DIM = 26


def _zero_action() -> np.ndarray:
    return np.zeros(ACTION_DIM, dtype=np.float32)


@dataclass
class Recipe:
    elem: str
    c_frac: float
    T_C: float
    atm_key: str
    method_idx: int
    device_idx: int = 2

    def copy(self) -> "Recipe":
        return Recipe(self.elem, self.c_frac, self.T_C, self.atm_key, self.method_idx, self.device_idx)


class TransitionGenerator:
    """Samples (state, action, next_state) triples + dual physics labels."""

    def __init__(self, featurizer: StateFeaturizer, seed: int = 0,
                 p_conc: float = 0.55, p_T: float = 0.15, p_pO2: float = 0.15,
                 p_elem: float = 0.10, p_method: float = 0.05,
                 exp_recipes: list | None = None, p_exp: float = 0.0,
                 target: str = "vacancy_concentration"):
        self.f = featurizer
        self.target = target
        self.rng = np.random.default_rng(seed)
        ps = np.array([p_conc, p_T, p_pO2, p_elem, p_method], dtype=np.float64)
        self.p = ps / ps.sum()
        self.kinds = ["conc", "T", "pO2", "elem", "method"]
        self.exp_recipes = exp_recipes or []
        self.p_exp = float(p_exp) if self.exp_recipes else 0.0

    # --- sampling a base recipe ---
    def _sample_recipe(self) -> Recipe:
        # with prob p_exp, draw a (jittered) experimental recipe so the encoder's training
        # distribution COVERS real recipes (fixes the OOD-z collapse); else broad-uniform.
        if self.exp_recipes and self.rng.random() < self.p_exp:
            r = self.exp_recipes[int(self.rng.integers(len(self.exp_recipes)))].copy()
            if r.elem != "undoped" and r.c_frac > 0:
                r.c_frac = float(np.clip(r.c_frac * 10 ** self.rng.uniform(-0.3, 0.3), 1e-5, 0.2))
            r.T_C = float(np.clip(r.T_C + self.rng.uniform(-40, 40), 200, 1200))
            return r
        elem = str(self.rng.choice(GEN_ELEMENTS))
        c = float(10 ** self.rng.uniform(-4, -1)) if elem != "undoped" else 0.0
        T_C = float(self.rng.uniform(300, 1100))
        atm = str(self.rng.choice(ATM_KEYS))
        # bias toward sputter (the VC frontier method) but include variety
        method = int(self.rng.choice([0, 0, 0, 1, 2, 3, 4, 5]))
        return Recipe(elem, c, T_C, atm, method, device_idx=2)

    # --- applying an action -> (next recipe, action vector) ---
    def _apply(self, s: Recipe) -> tuple[Recipe, np.ndarray]:
        kind = self.kinds[int(self.rng.choice(len(self.kinds), p=self.p))]
        t = s.copy()
        a = _zero_action()
        if kind == "conc" and s.elem != "undoped":
            d = float(self.rng.uniform(-1.2, 1.2))                  # +-1.2 decades
            new_c = float(np.clip(s.c_frac * (10 ** d), 1e-5, 0.2))
            a[0] = math.log10(max(new_c, 1e-6)) - math.log10(max(s.c_frac, 1e-6))
            t.c_frac = new_c
        elif kind == "T":
            new_T = float(np.clip(s.T_C + self.rng.uniform(-300, 300), 250, 1150))
            a[1] = ((new_T + 273.15) - (s.T_C + 273.15)) / 1000.0
            t.T_C = new_T
        elif kind == "pO2":
            new_atm = str(self.rng.choice(ATM_KEYS))
            a[2] = math.log10(ATM_PO2[new_atm]) - math.log10(ATM_PO2[s.atm_key])
            t.atm_key = new_atm
        elif kind == "elem":
            new_elem = str(self.rng.choice([e for e in GEN_ELEMENTS if e != s.elem]))
            t.elem = new_elem
            if new_elem != "undoped" and s.elem == "undoped":
                t.c_frac = float(10 ** self.rng.uniform(-4, -1))
                a[0] = math.log10(max(t.c_frac, 1e-6)) - math.log10(max(s.c_frac, 1e-6))
            a[10:21] = self.f._elem_desc(new_elem)
            a[21] = 1.0
        else:  # method
            new_m = int(self.rng.choice([m for m in range(N_METHODS) if m != s.method_idx]))
            t.method_idx = new_m
            a[3:9] = np.eye(N_METHODS)[new_m]
            a[9] = 1.0
        return t, a

    def _labels(self, r: Recipe) -> tuple[float, float]:
        T_K = r.T_C + 273.15
        log_pO2 = math.log10(ATM_PO2[r.atm_key])
        c = max(r.c_frac, 1e-6)
        if self.target == "photo_dark_ratio":
            return (measured_log_pdr(r.elem, c, T_K, log_pO2), bulk_log_pdr(r.elem, c, T_K, log_pO2))
        return (measured_log_vo(r.elem, c, T_K, log_pO2), bulk_log_vo(r.elem, c, T_K, log_pO2))

    def sample_batch(self, n: int) -> dict:
        D = self.f.dim
        xs = np.zeros((n, D), np.float32); xt = np.zeros((n, D), np.float32)
        av = np.zeros((n, ACTION_DIM), np.float32)
        ym_s = np.zeros(n, np.float32); yb_s = np.zeros(n, np.float32)
        ym_t = np.zeros(n, np.float32); yb_t = np.zeros(n, np.float32)
        for i in range(n):
            s = self._sample_recipe()
            t, a = self._apply(s)
            xs[i] = self.f.featurize(s.elem, s.c_frac, s.T_C, s.atm_key, s.method_idx, s.device_idx)
            xt[i] = self.f.featurize(t.elem, t.c_frac, t.T_C, t.atm_key, t.method_idx, t.device_idx)
            av[i] = a
            ym_s[i], yb_s[i] = self._labels(s)
            ym_t[i], yb_t[i] = self._labels(t)
        return dict(x_s=xs, x_t=xt, a=av, ym_s=ym_s, yb_s=yb_s, ym_t=ym_t, yb_t=yb_t)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Experimental featurization (Stage-B) + within-DOI transitions (optional)
# ──────────────────────────────────────────────────────────────────────────────
def _parse_conc_frac(row) -> float:
    """concentration_at% column is in at% (e.g. 2.0 -> 0.02 atomic fraction)."""
    import pandas as pd
    c = pd.to_numeric(row.get("concentration_at%"), errors="coerce")
    if c is None or (isinstance(c, float) and math.isnan(c)):
        return 0.0
    return float(c) / 100.0


def load_experimental_recipes(csv_path: str) -> list:
    """All experimental recipes (labeled or not) as Recipe objects — for marginal-matched
    synthetic sampling + SIGReg-on-experimental (Stage-A representation coverage; NO labels)."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    out = []
    for _, row in df.iterrows():
        elem = str(row.get("element") or "undoped").strip()
        if elem in ("—", "-", "nan", ""):
            elem = "undoped"
        c = _parse_conc_frac(row)
        T_raw = pd.to_numeric(row.get("temperature_C"), errors="coerce")
        T_C = float(T_raw) if pd.notna(T_raw) else 25.0
        atm = atmosphere_to_key(str(row.get("atmosphere")))
        method = method_to_idx(str(row.get("method")))
        device = device_to_idx(f"{row.get('method')} {row.get('notes')}")
        out.append(Recipe(elem, c if c > 0 else (0.0 if elem == "undoped" else 1e-3),
                          T_C, atm, method, device))
    return out


def featurize_recipes(recipes: list, featurizer: StateFeaturizer) -> np.ndarray:
    return np.stack([featurizer.featurize(r.elem, max(r.c_frac, 1e-6) if r.elem != "undoped" else 1e-6,
                                          r.T_C, r.atm_key, r.method_idx, r.device_idx)
                     for r in recipes]).astype(np.float32)


def load_experimental_states(csv_path: str, target: str, featurizer: StateFeaturizer):
    """Featurize every row with a non-null target. Returns dict of arrays + meta lists."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    y = pd.to_numeric(df[target], errors="coerce")
    df = df[y.notna()].reset_index(drop=True)
    yv = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=np.float32)
    X = np.zeros((len(df), featurizer.dim), np.float32)
    dois, elems, concs = [], [], []
    for i, row in df.iterrows():
        elem = str(row.get("element") or "undoped").strip()
        if elem in ("—", "-", "nan", ""):
            elem = "undoped"
        c = _parse_conc_frac(row)
        T_raw = pd.to_numeric(row.get("temperature_C"), errors="coerce")
        T_C = float(T_raw) if pd.notna(T_raw) else 25.0   # NaN is truthy -> must use pd.notna, not `or`
        atm = atmosphere_to_key(str(row.get("atmosphere")))
        method = method_to_idx(str(row.get("method")))
        device = device_to_idx(f"{row.get('method')} {row.get('notes')}")
        X[i] = featurizer.featurize(elem, c if c > 0 else 1e-6, T_C, atm, method, device)
        dois.append(str(row.get("doi"))); elems.append(elem); concs.append(c)
    sw = pd.to_numeric(df.get("sample_weight"), errors="coerce").fillna(1.0).to_numpy(np.float32) \
        if "sample_weight" in df.columns else np.ones(len(df), np.float32)
    return dict(X=X, y=yv, doi=np.array(dois), element=np.array(elems),
                conc=np.array(concs, np.float32), sample_weight=sw)


if __name__ == "__main__":
    f = StateFeaturizer()
    print("featurizer dim:", f.dim, "(elem", DESCRIPTOR_DIM, "+cont3 +meth6 +dev3 +phi",
          f._phi_dft_dim, "+mace", f.mace_proj_dim, ")")
    g = TransitionGenerator(f, seed=1)
    b = g.sample_batch(2048)
    for k, v in b.items():
        print(f"  {k}: {v.shape} mean={v.mean():.3f}")
    # sanity: Mg conc up -> measured V_O down; bulk V_O up
    T_K, lp = 973.15, math.log10(0.2)
    print("\nMg measured  c=0.1%% -> 5%%:", round(measured_log_vo("Mg",0.001,T_K,lp),2),
          "->", round(measured_log_vo("Mg",0.05,T_K,lp),2), "(should DECREASE)")
    print("Mg bulk      c=0.1%% -> 5%%:", round(bulk_log_vo("Mg",0.001,T_K,lp),2),
          "->", round(bulk_log_vo("Mg",0.05,T_K,lp),2), "(should INCREASE)")
    print("Sn measured  c=0.1%% -> 5%%:", round(measured_log_vo("Sn",0.001,T_K,lp),2),
          "->", round(measured_log_vo("Sn",0.05,T_K,lp),2), "(should INCREASE)")
    print("Sn bulk      c=0.1%% -> 5%%:", round(bulk_log_vo("Sn",0.001,T_K,lp),2),
          "->", round(bulk_log_vo("Sn",0.05,T_K,lp),2), "(should DECREASE)")
