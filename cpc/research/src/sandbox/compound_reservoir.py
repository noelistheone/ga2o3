"""OXIDE-TARGET COMPOUND RESERVOIR for the β-Ga2O3 defect engine  (Round 20, Task 1).

═══════════════════════════════════════════════════════════════════════════════════════════
  ►►► UNGATED CAPABILITY — READ `ungated_marker()` BEFORE USING ANY NUMBER FROM THIS MODULE.
  There is NO measurement in either corpus of a dopant incorporated from a NAMED oxide target
  against which a reservoir prediction can be confronted.  See §GATE below and
  results/mathwork/PREREG_round20_reservoir.md §4.
═══════════════════════════════════════════════════════════════════════════════════════════

WHAT THIS IS
------------
Sputtering / PLD / ALD from a compound target (MgO, SnO2, ...) does not fix the dopant's
CONCENTRATION; it fixes its CHEMICAL POTENTIAL, through equilibrium with the compound:

    j·μ_X + k·μ_O = G0_{X_j O_k}(T)          (X_j O_k target in equilibrium with the film)

μ_O is NOT an unknown here: `thermo.mu_Ga_O(T, pO2)` returns it in closed form from the SGTE
O2(g) polynomial, for any (T, pO2).  So the compound constraint collapses to a SCALAR

    μ_X(T, pO2) = ( G0_{X_j O_k}(T) − k·μ_O(T, pO2) ) / j

which `engine.solve_single(..., mu_override={X: μ_X})` already accepts.  **No solver branch is
added and `engine.py` is not modified.**  This module is purely additive: with no reservoir
specified, every existing code path is untouched and bit-identical.

This is the same construction KROGER itself uses to bound impurity chemical potentials
(`external/KROGER/Ga2O3/impurity_mu_limit_imposed_by_pO2.m`), where it is read as an UPPER BOUND
on μ_X — "how much X can dissolve before the competing oxide precipitates".  Read as a
RESERVOIR it is an equality: the target IS the competing phase, so the film sits ON that bound.

THE TWO READINGS, AND WHY BOTH ARE OFFERED
------------------------------------------
  `mode="reservoir"`  : μ_X is fixed at the tie.  [X] is an OUTPUT.  Use when the compound really
                        is present as a bulk phase in equilibrium (a compound sputter target run
                        at, or annealed to, equilibrium).
  `mu_limit(...)`     : the same number read as a CEILING — the maximum μ_X for which the film
                        does not precipitate X_jO_k.  Use to test whether a REQUESTED
                        fixed-concentration doping level is thermodynamically reachable at all.

THE REFERENCE SYSTEMATIC — THE THING THAT MAKES THIS MODE FRAGILE
-----------------------------------------------------------------
Round 12e MEASURED the inter-setup systematic on Ga-containing formation energies at **~1.3 eV**
(not the 0.3 eV the project had assumed): our HSE V_O q0 O-rich ΔH is 0.80–1.53 eV off KROGER's,
and this setup gets ΔHf(Ga2O3) wrong by ~3 eV per formula unit.

The two doping modes respond to that systematic COMPLETELY DIFFERENTLY:

  * FIXED-CONCENTRATION mode is IMMUNE.  A shift δ of element X's elemental reference maps
    dHo → dHo + δ·dm[:,X]; the solver then re-solves μ_X, which moves by exactly δ, so
    dG = dHo − dm·μ is unchanged and n does not move at all.  The unknown is absorbed.
  * RESERVOIR mode is NOT.  μ_X is PINNED by the SGTE tie and cannot absorb anything, so every
    X-containing charge state moves by δ·dm[:,X] and [X] scales like exp(−δ·dm_X/kT).  There is
    no self-consistent damping of the kind Round 13a found for the E_F response.

So the reservoir mode inherits the FULL 1.3 eV systematic.  Every reservoir output from this
module therefore carries the measured slope and band as NAMED FIELDS —
`d_log10n_per_eV_reservoir`, `reservoir_n_systematic_dex_at_1p3eV`, `reservoir_n_band_dex` —
exactly as the band-edge systematic is carried.  A reservoir carrier density quoted without them
is not a claim.

WHERE THE THERMOCHEMISTRY COMES FROM
------------------------------------
`scripts/round20_extract_compound_G0.py` extracts the published FactSage/SGTE polynomial
COEFFICIENTS (thermochemical facts, not code) from the `G_*.m` files already on disk under
`external/KROGER/G_thermo_files/`, and is validated by reproducing the two polynomials this repo
had ALREADY hand-ported — `thermo.G0_Ga2O3_ls` and `thermo.G0_O2_gv` — to < 2e-15 eV.  The table
is embedded below so this module has no runtime dependency on `external/`.

Compounds are supported ONLY where a polynomial exists.  Zn (ZnO), Ta (Ta2O5), Co, Ge (GeO2) have
NO polynomial anywhere on disk and are **NOT SUPPORTED**.  Nothing is invented for them.

GATE
----
`census_gate_status()` returns the pre-registered verdict.  Short version: the transport corpus
(549 rows) has exactly ONE row naming a compound dopant source — a "1 at% Ge doped sintered
target" — which has no DOI and whose element has no oxide G0 on disk; the device corpus (488
rows) has one ALD row with a "5% MgO cycle" (a supercycle fraction, not an equilibrium reservoir)
and reports no carrier density at all.  **0 prediction blocks against a pre-registered floor of 5
=> UNTESTABLE-ON-THIS-CORPUS => this capability is UNGATED.**
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from . import engine, thermo

Q = 1.602176634e-19
AVO = 6.0221409e+23
J_PER_MOL_TO_EV = Q * AVO

# ── the MEASURED energetics systematic (Round 12e). Named, not hidden. ──────────────────────
SIGMA_ENERGETICS_MEASURED_EV = 1.300     # inter-setup systematic on Ga-containing dHo
SIGMA_ENERGETICS_SHIPPED_EV = 0.500      # the value pre-registered in Stage 3, kept for contrast

# ── embedded SGTE/FactSage coefficient table (see module docstring for provenance) ──────────
_TABLE_JSON = r"""{
"CaO_gv": {
"element": "Ca",
"j": 1,
"k": 1,
"phases": [
[
[
298.0,
1400.0,
false,
{
"T1": 205.751461,
"T2": 0.01280263453,
"T3": -1.712449235e-06,
"TlnT": -61.5491809,
"Tm1": -537947.001,
"const": 78867.7875,
"lnT": -10352.2747
}
],
[
1400.0,
3800.0,
false,
{
"T1": -406018.798,
"T2": 3.27569158,
"T3": -2.831966034e-05,
"Thalf": 5256160.29,
"TlnT": 48763.2209,
"Tm1": -329377998.0,
"const": 37566281.1,
"lnT": -14615548.9
}
],
[
3800.0,
6000.0,
false,
{
"T1": 3153.92066,
"Thalf": -178046.562,
"TlnT": -282.261951,
"Tm1": 873451717.0,
"const": -12975648.8,
"lnT": 2367789.28
}
]
]
]
},
"CaO_ls": {
"element": "Ca",
"j": 1,
"k": 1,
"phases": [
[
[
298.0,
2845.0,
false,
{
"T1": 376.676564,
"Thalf": -535.615998,
"TlnT": -58.7911706,
"Tm1": 573572.991,
"Tm2": -17163131.3,
"const": -651262.658
}
],
[
2845.0,
3500.0,
false,
{
"T1": 407.120846,
"TlnT": -62.76,
"const": -676442.671
}
]
],
[
[
298.0,
2845.0,
false,
{
"T1": 348.735802,
"Thalf": -535.615998,
"TlnT": -58.7911706,
"Tm1": 573572.991,
"Tm2": -17163131.3,
"const": -571766.658
}
],
[
2845.0,
3500.0,
false,
{
"T1": 379.180084,
"TlnT": -62.76,
"const": -596946.671
}
]
]
]
},
"Cr2O3_ls": {
"element": "Cr",
"j": 2,
"k": 3,
"phases": [
[
[
115.0,
140.0,
true,
{
"T1": -8.96023508,
"T2": 0.0757014055,
"T3": -0.0007047015,
"T4": 6.577516667e-07,
"const": -1140516.83
}
],
[
140.0,
298.0,
false,
{
"T1": 1274.14253,
"T2": -0.0411814535,
"Thalf": -6534.3991,
"TlnT": -171.344603,
"const": -1124581.89
}
],
[
298.0,
3000.0,
false,
{
"T1": 739.717283,
"T2": -0.0039,
"T3": -5e-08,
"TlnT": -121.44,
"Tm1": 1000000.0,
"const": -1169514.24
}
]
],
[
[
298.0,
1800.0,
false,
{
"T1": 914.842281,
"T2": -0.003420837616,
"Thalf": -5069.9719,
"TlnT": -141.819281,
"Tm1": 1931683.05,
"const": -1126252.44,
"lnT": 22280.8848
}
],
[
1800.0,
4500.0,
false,
{
"T1": 980.900759,
"TlnT": -156.6908,
"const": -1102615.27
}
]
]
]
},
"Cr3O4_ls": {
"element": "Cr",
"j": 3,
"k": 4,
"phases": [
[
[
298.0,
2603.0,
false,
{
"T1": 1015.79091,
"T2": 0.000266381445,
"T3": -1.3381511e-06,
"T4": 1.23628925e-10,
"TlnT": -174.54392,
"Tm1": 1422428.9,
"const": -1479091.69
}
]
]
]
},
"CrO2_ls": {
"element": "Cr",
"j": 1,
"k": 2,
"phases": [
[
[
298.0,
600.0,
true,
{
"T1": 587.380815,
"T2": -0.008577,
"TlnT": -94.558,
"const": -610530.907
}
]
]
]
},
"CrO_ls": {
"element": "Cr",
"j": 1,
"k": 1,
"phases": [
[
[
298.0,
1800.0,
false,
{
"T1": 328.167001,
"T2": -0.00136833505,
"Thalf": -2027.98876,
"TlnT": -56.727712,
"Tm1": 772673.2,
"const": -321531.193,
"lnT": 8912.3539
}
],
[
1800.0,
2000.0,
false,
{
"T1": 355.301302,
"TlnT": -62.76,
"const": -312226.95
}
]
]
]
},
"Fe2O3_ls": {
"element": "Fe",
"j": 2,
"k": 3,
"phases": [
[
[
298.0,
2500.0,
false,
{
"T1": 828.050052,
"TlnT": -137.00893,
"Tm1": 1453820.0,
"const": -861183.055
}
]
],
[
[
298.0,
956.0,
false,
{
"T1": 921.403602,
"T2": -0.002740395108,
"T3": 6.890191675e-06,
"T4": -6.496518766e-09,
"TlnT": -146.8584,
"Tm1": 2788391.03,
"Tm2": -87604647.8,
"const": -809611.086
}
],
[
956.0,
1873.0,
false,
{
"T1": 912.363938,
"TlnT": -146.8584,
"Tm1": 2788391.03,
"Tm2": -87604647.8,
"const": -802879.999
}
]
],
[
[
298.0,
1873.0,
false,
{
"T1": 911.016249,
"TlnT": -146.8584,
"Tm1": 2788391.03,
"Tm2": -87604647.8,
"const": -801591.449
}
]
]
]
},
"Fe3O4_ls": {
"element": "Fe",
"j": 3,
"k": 4,
"phases": [
[
[
298.0,
848.0,
false,
{
"T1": 1282.9585,
"T2": -0.01901728207,
"T3": 3.967903931e-05,
"T4": -3.104596095e-08,
"TlnT": -207.93083,
"Tm1": 3621668.52,
"Tm2": -110726059.0,
"const": -1200277.81
}
],
[
848.0,
1870.0,
false,
{
"T1": 1260.56231,
"TlnT": -207.93083,
"Tm1": 3621668.52,
"Tm2": -110726059.0,
"const": -1186819.15
}
]
],
[
[
848.0,
1870.0,
false,
{
"T1": 1258.7164,
"TlnT": -207.93083,
"Tm1": 3621668.52,
"Tm2": -110726059.0,
"const": -1185253.6
}
],
[
1870.0,
2500.0,
false,
{
"T1": 1304.25027,
"TlnT": -213.384,
"const": -1191672.58
}
]
],
[
[
298.0,
848.0,
false,
{
"T1": 1253.59865,
"T2": -0.01901728207,
"T3": 3.967903931e-05,
"T4": -3.104596095e-08,
"TlnT": -207.93083,
"Tm1": 3621668.31,
"Tm2": -110726059.0,
"const": -1030731.16
}
],
[
848.0,
1870.0,
false,
{
"T1": 1231.20246,
"TlnT": -207.93083,
"Tm1": 3621668.52,
"Tm2": -110726059.0,
"const": -1017272.51
}
]
],
[
[
848.0,
1870.0,
false,
{
"T1": 1229.35613,
"TlnT": -207.93083,
"Tm1": 3621668.52,
"Tm2": -110726059.0,
"const": -1015705.37
}
],
[
1870.0,
2500.0,
false,
{
"T1": 1274.89,
"TlnT": -213.384,
"const": -1022124.35
}
]
],
[
[
298.0,
1870.0,
false,
{
"T1": 1184.8811,
"TlnT": -207.93083,
"Tm1": 3621668.52,
"Tm2": -110726059.0,
"const": -1047181.6
}
],
[
1870.0,
2500.0,
false,
{
"T1": 1230.41497,
"TlnT": -213.384,
"const": -1053600.58
}
]
]
]
},
"FeO_ls": {
"element": "Fe",
"j": 1,
"k": 1,
"phases": [
[
[
298.0,
1644.0,
false,
{
"T1": -330.687435,
"T2": -0.01530402997,
"Thalf": 6003.60001,
"TlnT": 18.0244741,
"Tm1": 1266650.0,
"const": -322147.542
}
],
[
1644.0,
2000.0,
false,
{
"T1": 417.258468,
"TlnT": -68.1992,
"const": -299283.753
}
]
],
[
[
298.0,
1644.0,
false,
{
"T1": -349.657168,
"T2": -0.01530402997,
"Thalf": 6003.60001,
"TlnT": 18.0244741,
"Tm1": 1266650.0,
"const": -290958.454
}
],
[
1644.0,
2000.0,
false,
{
"T1": 398.288735,
"TlnT": -68.1992,
"const": -268094.665
}
]
]
]
},
"Ga2O3_ls": {
"element": "Ga",
"j": 2,
"k": 3,
"phases": [
[
[
0.0,
1000000000.0,
true,
{
"T1": 684.8332,
"T2": -0.00796268819,
"TlnT": -112.3935,
"Tm1": 1080114.0,
"const": -1127917.0
}
]
]
]
},
"H2O_gv": {
"element": "H",
"j": 2,
"k": 1,
"phases": [
[
[
298.0,
1100.0,
false,
{
"T1": -15.1731427,
"T2": -0.007474858139,
"T3": 9.205931575e-08,
"TlnT": -25.7816397,
"Tm1": 13999.6597,
"const": -255475.808,
"lnT": 1107.27182
}
],
[
1100.0,
4000.0,
false,
{
"T1": 164.817017,
"T2": -8.054003823e-05,
"Thalf": 5947.37003,
"TlnT": -53.1457895,
"Tm1": -12075583.2,
"const": 152152.281,
"lnT": -83128.2757
}
]
]
]
},
"H2O_ls": {
"element": "H",
"j": 2,
"k": 1,
"phases": [
[
[
0.0,
273.0,
false,
{
"T1": 198.232091,
"TlnT": -36.246,
"const": -303622.945
}
],
[
298.0,
500.0,
false,
{
"T1": -1118.62474,
"T2": -0.760342801,
"T3": 0.00053188744,
"T4": -2.059132015e-07,
"TlnT": 203.118982,
"Tm1": -1924378.83,
"const": -256638.942
}
]
]
]
},
"HfO2_ls": {
"element": "Hf",
"j": 1,
"k": 2,
"phases": [
[
[
298.0,
1973.0,
true,
{
"T1": 433.620264,
"T2": -0.004524996,
"TlnT": -72.11124,
"Tm1": 647055.6,
"const": -1170984.68
}
]
],
[
[
298.0,
1973.0,
true,
{
"T1": 428.318692,
"T2": -0.004524996,
"TlnT": -72.11124,
"Tm1": 647055.6,
"const": -1160524.68
}
],
[
1973.0,
3173.0,
false,
{
"T1": 725.217214,
"TlnT": -108.784,
"const": -1214609.54
}
]
],
[
[
298.0,
1973.0,
true,
{
"T1": 395.353045,
"T2": -0.004524996,
"TlnT": -72.11124,
"Tm1": 647055.6,
"const": -1055924.68
}
],
[
1973.0,
3300.0,
false,
{
"T1": 692.251566,
"TlnT": -108.784,
"const": -1110009.54
}
]
]
]
},
"IrO2_ls": {
"element": "Ir",
"j": 1,
"k": 2,
"phases": [
[
[
0.0,
800.0,
false,
{
"T1": 613.806359,
"Thalf": -1430.8,
"TlnT": -93.27,
"Tm1": 557000.0,
"const": -261863.998
}
],
[
800.0,
1400.0,
false,
{
"T1": 420.004152,
"T2": -0.0055,
"TlnT": -70.31,
"const": -265858.066
}
]
]
]
},
"MgO_gv": {
"element": "Mg",
"j": 1,
"k": 1,
"phases": [
[
[
298.0,
1100.0,
false,
{
"T1": 8518.40222,
"T2": 0.05878503055,
"T3": 3.784973866e-06,
"Thalf": -141244.711,
"TlnT": -898.190474,
"Tm1": 10851502.7,
"const": -1102681.16,
"lnT": 435353.536
}
],
[
1100.0,
4500.0,
false,
{
"T1": 2388.43394,
"T2": 0.003419105015,
"Thalf": -78828.9894,
"TlnT": -243.395147,
"Tm1": 70329447.0,
"const": -2483196.14,
"lnT": 582033.945
}
],
[
4500.0,
6000.0,
false,
{
"T1": 3472.65363,
"Thalf": -191013.773,
"TlnT": -305.763456,
"Tm1": 977136770.0,
"const": -13298543.7,
"lnT": 2447324.28
}
]
]
]
},
"MgO_ls": {
"element": "Mg",
"j": 1,
"k": 1,
"phases": [
[
[
298.0,
3098.0,
false,
{
"T1": 420.064762,
"Thalf": -1184.796,
"TlnT": -61.1096505,
"Tm1": 310577.002,
"Tm2": -974102.005,
"const": -611541.379
}
],
[
3098.0,
3500.0,
false,
{
"T1": 462.122764,
"TlnT": -66.944,
"const": -662387.67
}
]
],
[
[
298.0,
3098.0,
false,
{
"T1": 490.908663,
"T2": 0.001571092,
"Thalf": -1184.796,
"TlnT": -72.7955625,
"Tm1": -261375.798,
"Tm2": -974102.005,
"const": -554894.205
}
],
[
3098.0,
3500.0,
false,
{
"T1": 437.137549,
"TlnT": -66.944,
"const": -584985.686
}
]
]
]
},
"Rh2O3_ls": {
"element": "Rh",
"j": 2,
"k": 3,
"phases": [
[
[
298.0,
1300.0,
false,
{
"T1": 492.132667,
"T2": -0.0288696,
"TlnT": -86.77616,
"const": -384078.63
}
]
]
]
},
"Rh2O_ls": {
"element": "Rh",
"j": 2,
"k": 1,
"phases": [
[
[
298.0,
973.0,
false,
{
"T1": 330.304053,
"T2": -0.01353524,
"TlnT": -65.22856,
"const": -115627.889
}
]
]
]
},
"RhO2_ls": {
"element": "Rh",
"j": 1,
"k": 2,
"phases": [
[
[
298.0,
1000.0,
false,
{
"T1": 277.070194,
"T2": -0.015213029,
"T3": 1.505648333e-06,
"Thalf": -316.646634,
"TlnT": -47.7352734,
"Tm1": -131609.828,
"const": -250888.185
}
]
]
]
},
"RhO_ls": {
"element": "Rh",
"j": 1,
"k": 1,
"phases": [
[
[
298.0,
1023.0,
false,
{
"T1": 228.377327,
"T2": -0.01115036,
"TlnT": -41.17056,
"const": -104058.996
}
]
]
]
},
"SiO2_ls": {
"element": "Si",
"j": 1,
"k": 2,
"phases": [
[
[
298.0,
373.0,
false,
{
"T1": 536.025342,
"Thalf": -961.103996,
"TlnT": -80.0119918,
"Tm1": 1773342.0,
"Tm2": -81928061.6,
"const": -935388.521
}
],
[
373.0,
848.0,
false,
{
"T1": 537.075695,
"T2": -0.004220010851,
"T3": 7.535450247e-06,
"T4": -5.045870528e-09,
"Thalf": -961.103996,
"TlnT": -80.0119918,
"Tm1": 1773342.0,
"Tm2": -81928061.6,
"const": -935486.558
}
],
[
848.0,
850.0,
false,
{
"T1": -104.031919,
"TlnT": -0.04184,
"const": -876152.929
}
]
],
[
[
298.0,
1996.0,
false,
{
"T1": 533.278543,
"Thalf": -961.103996,
"TlnT": -80.0119918,
"Tm1": 1773342.0,
"Tm2": -81928061.6,
"const": -933315.349
}
],
[
1996.0,
3000.0,
false,
{
"T1": 571.62748,
"TlnT": -85.772,
"const": -964566.44
}
]
]
]
},
"SnO2_ls": {
"element": "Sn",
"j": 1,
"k": 2,
"phases": [
[
[
298.0,
1903.0,
false,
{
"T1": 474.985631,
"T2": -0.003681085,
"T3": -1.281666667e-10,
"TlnT": -76.041,
"Tm1": 1111933.5,
"const": -608087.742
}
],
[
1903.0,
4400.0,
false,
{
"T1": 584.839002,
"TlnT": -92.0,
"const": -600556.627
}
]
]
]
},
"SnO_ls": {
"element": "Sn",
"j": 1,
"k": 1,
"phases": [
[
[
298.0,
1273.0,
false,
{
"T1": 215.49932,
"T2": -0.007322,
"TlnT": -39.9572,
"const": -298331.317
}
]
]
]
},
"Ti2O3_ls": {
"element": "Ti",
"j": 2,
"k": 3,
"phases": [
[
[
298.0,
470.0,
true,
{
"T1": 6399.41546,
"Thalf": -58419.4632,
"TlnT": -730.233813,
"Tm1": -9404192.7,
"const": -1171154.54
}
],
[
470.0,
2115.0,
false,
{
"T1": 1157.75518,
"Thalf": -3000.87476,
"TlnT": -169.961109,
"Tm1": -804824.47,
"Tm2": 260920167.0,
"const": -1548686.12
}
]
],
[
[
298.0,
2115.0,
false,
{
"T1": 1155.3336,
"Thalf": -3000.87476,
"TlnT": -169.961109,
"Tm1": -804824.47,
"Tm2": 260920167.0,
"const": -1547548.01
}
]
],
[
[
298.0,
2115.0,
false,
{
"T1": 1105.68824,
"Thalf": -3000.87476,
"TlnT": -169.961109,
"Tm1": -804824.47,
"Tm2": 260920167.0,
"const": -1442547.94
}
]
]
]
},
"TiO2_ls": {
"element": "Ti",
"j": 1,
"k": 2,
"phases": [
[
[
298.0,
2130.0,
true,
{
"T1": 484.740378,
"TlnT": -77.8376214,
"Tm1": 1683920.51,
"Tm2": -67156778.7,
"const": -976986.65
}
]
],
[
[
298.0,
2100.0,
true,
{
"T1": 462.559573,
"TlnT": -75.035856,
"Tm1": 881359.6,
"const": -961316.13
}
]
],
[
[
298.0,
2130.0,
true,
{
"T1": 463.132871,
"TlnT": -77.8376214,
"Tm1": 1683920.51,
"Tm2": -67156778.7,
"const": -930962.65
}
]
]
]
},
"TiO_ls": {
"element": "Ti",
"j": 1,
"k": 1,
"phases": [
[
[
298.0,
2500.0,
true,
{
"T1": 255.475182,
"T2": -0.008898664224,
"T3": 1.105968351e-08,
"TlnT": -41.9944927,
"Tm1": 327010.589,
"const": -558169.702
}
]
],
[
[
298.0,
2500.0,
true,
{
"T1": 252.162502,
"T2": -0.008899251973,
"T3": 1.121961627e-08,
"TlnT": -41.9953178,
"Tm1": 327074.648,
"const": -553971.14
}
]
],
[
[
298.0,
1200.0,
true,
{
"T1": 230.218069,
"T2": -0.008858076521,
"TlnT": -42.013901,
"Tm1": 326982.846,
"const": -509468.783
}
],
[
1200.0,
4500.0,
false,
{
"T1": 410.41803,
"TlnT": -66.944,
"const": -526084.3
}
]
]
]
},
"ZrO2_gv": {
"element": "Zr",
"j": 1,
"k": 2,
"phases": [
[
[
298.0,
6000.0,
false,
{
"T1": 124.420897,
"Thalf": 7.18939396,
"TlnT": -58.1863462,
"Tm1": 1196235.07,
"Tm2": -64971632.0,
"const": -309428.051
}
]
]
]
},
"ZrO2_ls": {
"element": "Zr",
"j": 1,
"k": 2,
"phases": [
[
[
298.0,
1478.0,
true,
{
"T1": 652.58828,
"Thalf": -2337.90322,
"TlnT": -94.62116,
"Tm2": 20068472.3,
"const": -1106167.26
}
],
[
1478.0,
2950.0,
false,
{
"T1": 454.989704,
"TlnT": -74.4752,
"const": -1121304.04
}
]
],
[
[
298.0,
2950.0,
true,
{
"T1": 450.968857,
"TlnT": -74.4752,
"const": -1115350.77
}
]
],
[
[
298.0,
2950.0,
true,
{
"T1": 446.948857,
"TlnT": -74.4752,
"const": -1104765.78
}
]
],
[
[
298.0,
3500.0,
true,
{
"T1": 537.807469,
"TlnT": -87.864,
"const": -1057215.65
}
]
]
]
}
}"""

_TABLE = json.loads(_TABLE_JSON)

# element -> the compound polynomials available for it, in the order KROGER lists them
ELEMENT_COMPOUNDS: dict[str, list[str]] = {}
for _c, _s in _TABLE.items():
    ELEMENT_COMPOUNDS.setdefault(_s["element"], []).append(_c)
for _e in ELEMENT_COMPOUNDS:
    ELEMENT_COMPOUNDS[_e].sort()

# DB elements with NO compound polynomial anywhere on disk. Explicitly unsupported.
UNSUPPORTED: dict[str, str] = {
    "Zn": "ZnO — no G_ZnO_*.m exists; KROGER itself sets mu_Zn_lim = -30 eV (i.e. no limit)",
    "Co": "CoO / Co3O4 — no polynomial; KROGER sets mu_Co_lim = -30 eV",
    "Ta": "Ta2O5 — no polynomial; KROGER sets mu_Ta_lim = -30 eV",
    "Ge": "GeO2 / GeO — no polynomial; KROGER sets mu_Ge_lim = -30 eV",
    "Pt": "PtO2 exists only as a VAPOUR (G_PtO2_gv); no condensed target phase on disk",
}


def _alias(name: str) -> str:
    """Accept 'MgO' for 'MgO_ls', 'SnO2' for 'SnO2_ls', etc."""
    if name in _TABLE:
        return name
    for suf in ("_ls", "_gv"):
        if name + suf in _TABLE:
            return name + suf
    raise KeyError(
        f"no G0 polynomial for compound {name!r}. Available: {sorted(_TABLE)}. "
        f"Unsupported elements (no polynomial on disk, nothing invented): {sorted(UNSUPPORTED)}")


def G0_compound(name: str, T):
    """Standard Gibbs energy of the compound, **eV per formula unit**, X_i = 1 (pure phase).

    Piecewise Shomate summed over active T-segments, one accumulator per allotrope, then
    min-over-allotropes (the stable phase), exactly as the source polynomials prescribe.
    Returns +inf outside every validity range — an obvious error rather than a silent
    extrapolation.
    """
    spec = _TABLE[_alias(name)]
    T = np.asarray(T, dtype=np.float64)
    phases = []
    for ph in spec["phases"]:
        acc = np.zeros_like(T)
        for lo, hi, incl_lo, terms in ph:
            m = ((T >= lo) if incl_lo else (T > lo)) & (T <= hi)
            v = np.zeros_like(T)
            for kk, c in terms.items():
                v = v + c * _POW[kk](T)
            acc = acc + np.where(m, v, 0.0)
        phases.append(acc)
    g = np.stack(phases, axis=0).min(axis=0)
    g = np.where(g == 0.0, np.inf, g)
    return g / J_PER_MOL_TO_EV


_POW = {
    "const": lambda T: np.ones_like(T),
    "Tm2": lambda T: T ** -2.0,
    "Tm1": lambda T: T ** -1.0,
    "Thalf": lambda T: T ** 0.5,
    "T1": lambda T: T,
    "T2": lambda T: T ** 2.0,
    "T3": lambda T: T ** 3.0,
    "T4": lambda T: T ** 4.0,
    "TlnT": lambda T: T * np.log(T),
    "lnT": lambda T: np.log(T),
}


def mu_from_compound(name: str, T: float, pO2: float) -> float:
    """μ_X [eV] set by equilibrium with the compound X_jO_k at (T, pO2).

    μ_X = (G0_{X_jO_k}(T) − k·μ_O(T,pO2)) / j.  μ_O comes from `thermo.mu_Ga_O`, which is a
    CLOSED FORM in (T, pO2) — the compound constraint therefore collapses to this scalar and
    introduces no new unknown and no new solve.
    """
    spec = _TABLE[_alias(name)]
    _, mu_O = thermo.mu_Ga_O(float(T), float(pO2))
    g0 = float(G0_compound(name, float(T)))
    return (g0 - spec["k"] * float(mu_O)) / spec["j"]


def mu_limit(element: str, T: float, pO2: float) -> dict:
    """The most stringent (lowest) μ_X over every compound of `element` on disk.

    This is KROGER's `impurity_mu_limit_imposed_by_pO2` semantics: the competing oxide that
    precipitates first sets the ceiling on μ_X.
    """
    if element not in ELEMENT_COMPOUNDS:
        return {"element": element, "supported": False,
                "reason": UNSUPPORTED.get(element, "element not in the compound table"),
                "mu_limit_eV": None, "limiting_compound": None}
    per = {}
    for c in ELEMENT_COMPOUNDS[element]:
        try:
            v = mu_from_compound(c, T, pO2)
        except Exception as e:                                   # pragma: no cover
            v = float("nan")
            per[c] = {"mu_eV": None, "error": str(e)}
            continue
        per[c] = {"mu_eV": float(v), "j": _TABLE[c]["j"], "k": _TABLE[c]["k"]}
    ok = {c: d["mu_eV"] for c, d in per.items() if d.get("mu_eV") is not None
          and np.isfinite(d["mu_eV"])}
    if not ok:
        return {"element": element, "supported": False,
                "reason": f"all compounds out of their T validity range at T={T} K",
                "per_compound": per, "mu_limit_eV": None, "limiting_compound": None}
    lim = min(ok, key=lambda c: ok[c])
    return {"element": element, "supported": True, "T_K": float(T), "pO2_atm": float(pO2),
            "mu_limit_eV": float(ok[lim]), "limiting_compound": lim, "per_compound": per}


# ── the reference-shift instrument ──────────────────────────────────────────────────────────
def shifted_db(db, element: str, delta_eV: float):
    """Redefine `element`'s elemental reference by `delta_eV`:  dHo -> dHo + delta*dm[:, X].

    Identical to the definition used by `scripts/probe_verify_mu_reference.py`, so the numbers
    are comparable across rounds.
    """
    from .kroger_db import DefectDB
    col = db.col(element)
    return DefectDB(dHo=db.dHo + float(delta_eV) * db.dm[:, col].astype(float),
                    charge=db.charge.copy(), dm=db.dm.copy(), cs_ID=db.cs_ID.copy(),
                    prefactor=db.prefactor.copy(), sum_dm=db.sum_dm.copy(),
                    elements=list(db.elements), nD=db.nD, nCS=db.nCS)


@dataclass
class ReservoirResult:
    element: str
    compound: str
    T_K: float
    pO2_atm: float
    mu_X_eV: float
    stage: str
    n_cm3: float
    p_cm3: float
    EF_eV: float
    X_total_cm3: float
    max_site_occupancy: float
    exceeds_dilute_cap: bool
    d_log10n_per_eV_reservoir: float | None
    reservoir_n_systematic_dex_at_1p3eV: float | None
    meta: dict

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["reservoir_reference_systematic_eV"] = SIGMA_ENERGETICS_MEASURED_EV
        d["reservoir_n_band_dex"] = (
            None if self.reservoir_n_systematic_dex_at_1p3eV is None
            else [-abs(self.reservoir_n_systematic_dex_at_1p3eV),
                  +abs(self.reservoir_n_systematic_dex_at_1p3eV)])
        d["UNGATED"] = True
        d["ungated_reason"] = census_gate_status()["ungated_reason"]
        return d


OCC_DILUTE_CAP = 0.1     # same dilute/site-blocking cap engine.solve_single enforces in
                         # fixed-concentration mode. RESERVOIR MODE HAS NO SUCH GUARD in the
                         # engine, so it is CHECKED AND REPORTED here rather than enforced —
                         # silently capping a reservoir would change the physics being asked for.


def solve_reservoir(db, T: float, pO2: float, element: str, compound: str | None = None,
                    stage: str = "anneal", Nd_bg: float = 0.0, Na_bg: float = 0.0,
                    EcT_fraction: float = 0.40, fd: bool = True, sth_flag: float = 1.0,
                    mu_override: dict | None = None, fixed_conc: dict | None = None,
                    measure_systematic: bool = True,
                    systematic_delta_eV: float = SIGMA_ENERGETICS_MEASURED_EV,
                    strict: bool = True) -> ReservoirResult:
    # ROUND 20 REVIEW BLOCKER — DIAGNOSED, AND THE FAILURE IS PHYSICS, NOT A SOLVER BUG.
    # An adversarial reviewer found that at the shipped default strict=True only 4 of the 12
    # supported elements return at all (Mg, Zr, Hf, Ca), and that every number in
    # results/mathwork/round20_reservoir.json was produced with strict=False — i.e. by suppressing
    # the Stage-1 validity gate. Measured at 1073 K / pO2 1e-4, the reason is visible in the site
    # occupancy the reservoir demands:
    #     Mg 3.9e-6 | Zr 1.3e-8 | Hf 3.9e-8 | Ca 2.6e-8 | Sn 6.4e-7 | Si 3.8e-6 | Ti 3.4e-3
    #     H 0.122 | Ir 8.51 | Fe 1.9e4 | Rh 1.4e5 | Cr 1.2e11
    # An occupancy above 1 means the reservoir is asking for more dopant atoms than there are
    # lattice sites — up to 1e11 per site for Cr. That is the model correctly reporting that the
    # oxide is NOT an admissible competing phase at those conditions: in equilibrium with a
    # Cr2O3 / Rh2O3 / IrO2 reservoir the host would be consumed, and the dilute-defect formalism
    # the whole engine rests on no longer applies. Suppressing that with strict=False converts a
    # meaningful physical verdict into a silent number, which is the exact failure mode this
    # project spent nineteen rounds finding elsewhere.
    #
    # THEREFORE: strict=True stays the default and is the correct behaviour. Callers that want a
    # verdict instead of an exception should use reservoir_admissible() below, which classifies
    # the outcome rather than either raising opaquely or fabricating a value.
    """Solve dark equilibrium with `element` held at the compound-target chemical potential.

    `compound=None` uses the most stringent compound for that element (KROGER semantics).
    `stage` is a TAG that travels with the number — a carrier density without its stage is not
    a claim (Rounds 14/15).  This function solves ONE temperature; `stage="anneal"` means the
    value returned is the equilibrium value AT `T`, which is the well-conditioned quantity
    (Round 9).  Pass `stage="quench"` only if the caller has itself performed the quench.
    """
    if compound is None:
        lim = mu_limit(element, T, pO2)
        if not lim["supported"]:
            raise ValueError(f"compound reservoir not supported for {element}: {lim['reason']}")
        compound, mu_X = lim["limiting_compound"], lim["mu_limit_eV"]
    else:
        compound = _alias(compound)
        mu_X = mu_from_compound(compound, T, pO2)
    if not np.isfinite(mu_X):
        raise ValueError(f"G0({compound}) is out of its validity range at T={T} K")

    mo = dict(mu_override or {})
    mo[element] = float(mu_X)

    def _solve(_db):
        return engine.solve_single(_db, T, pO2=pO2, EcT_fraction=EcT_fraction,
                                   mu_override=dict(mo), fixed_conc=dict(fixed_conc or {}),
                                   fd=fd, sth_flag=sth_flag, Nd_bg=Nd_bg, Na_bg=Na_bg,
                                   strict=strict)

    r = _solve(db)
    col = db.col(element)
    tot = float((db.dm[:, col] * r.N_cs).sum())
    has = db.dm[:, col] > 0
    occ = float((r.N_cs[has] / db.prefactor[has]).max()) if has.any() else 0.0

    slope = sysdex = None
    if measure_systematic:
        d = float(systematic_delta_eV)
        lo = _solve(shifted_db(db, element, -d))
        hi = _solve(shifted_db(db, element, +d))
        l_lo, l_hi = np.log10(max(lo.n, 1e-300)), np.log10(max(hi.n, 1e-300))
        slope = float((l_hi - l_lo) / (2.0 * d))
        sysdex = float(0.5 * abs(l_hi - l_lo))     # half-width of the +-delta band, in dex

    return ReservoirResult(
        element=element, compound=compound, T_K=float(T), pO2_atm=float(pO2),
        mu_X_eV=float(mu_X), stage=stage, n_cm3=float(r.n), p_cm3=float(r.p),
        EF_eV=float(r.EF), X_total_cm3=tot, max_site_occupancy=occ,
        exceeds_dilute_cap=bool(occ > OCC_DILUTE_CAP),
        d_log10n_per_eV_reservoir=slope,
        reservoir_n_systematic_dex_at_1p3eV=sysdex,
        meta={"solver": "compound_reservoir.solve_reservoir -> engine.solve_single(mu_override)",
              "charge_residual": float(r.charge_residual),
              "Nd_bg": float(Nd_bg), "Na_bg": float(Na_bg),
              "systematic_delta_eV": float(systematic_delta_eV),
              "site_occupancy_note": ("reservoir mode has NO site-blocking guard in the engine; "
                                      "occupancy is reported, not enforced"),
              "converged": bool(r.meta.get("converged", True))})


# ── census / gate status ────────────────────────────────────────────────────────────────────
def compound_census() -> dict:
    """Per-element classification: G0 (i) ON DISK, (ii) DERIVABLE, (iii) ABSENT."""
    from .kroger_db import ELEMENTS
    out = {}
    for el in ELEMENTS:
        if el in ELEMENT_COMPOUNDS:
            out[el] = {"class": "ON_DISK", "compounds": ELEMENT_COMPOUNDS[el]}
        elif el in UNSUPPORTED:
            out[el] = {"class": "ABSENT", "reason": UNSUPPORTED[el], "compounds": []}
        else:
            out[el] = {"class": "ABSENT", "reason": "no binary-oxide polynomial on disk",
                       "compounds": []}
    out["O"] = {"class": "ON_DISK", "compounds": ["O2_gv (thermo.G0_O2_gv)"],
                "note": "mu_O is a closed form in (T,pO2) and is never an unknown"}
    out["Ga"] = {"class": "ON_DISK", "compounds": ["Ga2O3_ls (thermo.G0_Ga2O3_ls)"],
                 "note": "host; not a dopant reservoir"}
    return out


def census_gate_status() -> dict:
    """The PRE-REGISTERED gate verdict for this capability. See
    results/mathwork/PREREG_round20_reservoir.md §4."""
    return {
        "gate": "GATE-R1 (compound-source confrontation)",
        "prereg": "results/mathwork/PREREG_round20_reservoir.md",
        "prediction_blocks_found": 0,
        "power_floor_blocks": 5,
        "verdict": "UNTESTABLE-ON-THIS-CORPUS => SILENT => CAPABILITY IS UNGATED",
        "ungated_reason": (
            "No measurement exists in either corpus of a dopant incorporated from a NAMED "
            "oxide/compound target with both a DOI and a measured carrier density. Transport "
            "corpus: 1 candidate row (1 at% Ge sintered target) with no DOI and no oxide G0 for "
            "Ge. Device corpus: 1 ALD row with a '5% MgO cycle' (a supercycle fraction, not an "
            "equilibrium reservoir) and no carrier density. 0 blocks vs a pre-registered floor "
            "of 5."),
        "measurement_that_would_gate_it": (
            "A single paper reporting: (a) a NAMED binary-oxide sputter/PLD target or compound "
            "source for the dopant (e.g. MgO, SnO2, TiO2, SiO2, ZrO2, HfO2 — one whose G0 is on "
            "disk), (b) the growth/anneal temperature AND the oxygen partial pressure, and (c) "
            "the resulting dopant concentration (SIMS) AND carrier density (Hall) on the SAME "
            "film. Five such papers, or five independently-conditioned films within one paper "
            "reported as separate blocks, clears the power floor. A pO2 SERIES at fixed T on one "
            "compound target is the highest-information version, because mu_X moves with pO2 by "
            "a KNOWN closed-form amount (-k/j * dmu_O) while the DFT reference systematic is "
            "CONSTANT along that series and therefore cancels out of the SLOPE."),
    }


def ungated_marker() -> str:
    s = census_gate_status()
    return ("*** UNGATED CAPABILITY ***\n"
            f"{s['verdict']}\n{s['ungated_reason']}\n"
            f"WOULD BE GATED BY: {s['measurement_that_would_gate_it']}\n"
            f"Reservoir-mode carrier densities also carry the FULL {SIGMA_ENERGETICS_MEASURED_EV} "
            "eV Round-12e energetics systematic, undamped (fixed-concentration mode does not).")


# ======================================================================================
# ROUND 20 REVIEW FIX — classify the reservoir outcome instead of raising or suppressing.
# ======================================================================================
DILUTE_OCC_WARN = 0.1        # engine's own dilute-approximation working range
DILUTE_OCC_FAIL = 1.0        # above one dopant per lattice site the formalism does not apply


def reservoir_admissible(db, T: float, pO2: float, element: str, compound: str | None = None,
                         **kw) -> dict:
    """Is this compound an admissible reservoir for `element` at (T, pO2)? A VERDICT, not a number.

    Returns a dict that always carries a verdict and never silently returns a carrier density for
    a condition the dilute formalism cannot describe:

      ADMISSIBLE          max site occupancy <= DILUTE_OCC_WARN; the solve also passed the
                          engine's strict validity gate. The ReservoirResult is included.
      MARGINAL            occupancy between DILUTE_OCC_WARN and DILUTE_OCC_FAIL, or the strict
                          gate failed for a non-occupancy reason (typically charge-residual
                          convergence). A result is included but must not be quoted as a value.
      HOST_CONSUMED       occupancy > 1: the reservoir demands more dopant than there are lattice
                          sites, i.e. the oxide is not a stable competing phase at these
                          conditions and beta-Ga2O3 would dissolve. This is a PHYSICAL verdict and
                          is the honest output — NOT a carrier density.
      UNSUPPORTED         no G0 for that compound on disk (5 of the 19 DB elements: Zn, Ta, Co,
                          Ge, Pt — KROGER has no data for them either).

    Every branch carries the UNGATED marker: GATE-R1 found zero scorable corpus rows, so NO
    reservoir prediction has ever been confronted with a measurement.
    """
    out = {"element": element, "compound": compound, "T_K": float(T), "pO2_atm": float(pO2),
           "UNGATED": True,
           "ungated_reason": ungated_marker() if "ungated_marker" in globals() else
                             ("GATE-R1 census of both corpora found zero scorable rows: no "
                              "reservoir prediction has ever been confronted with a measurement."),
           "reference_systematic_eV": SIGMA_ENERGETICS_MEASURED_EV}
    try:
        res = solve_reservoir(db, T, pO2, element, compound=compound, strict=False, **kw)
    except Exception as e:                      # unsupported compound, bad name, missing G0
        out.update(verdict="UNSUPPORTED", reason=f"{type(e).__name__}: {e}", result=None)
        return out
    occ = float(getattr(res, "max_site_occupancy", float("nan")))
    out["max_site_occupancy"] = occ
    strict_ok = True
    strict_err = None
    try:
        solve_reservoir(db, T, pO2, element, compound=compound, strict=True, **kw)
    except Exception as e:
        strict_ok, strict_err = False, f"{type(e).__name__}: {str(e)[:200]}"
    out["strict_gate_passed"] = strict_ok
    out["strict_gate_error"] = strict_err
    if occ > DILUTE_OCC_FAIL:
        out.update(verdict="HOST_CONSUMED", result=None,
                   reason=(f"site occupancy {occ:.3g} exceeds 1 dopant per lattice site, so the "
                           f"reservoir would dissolve the host: this compound is not an admissible "
                           f"competing phase at {T:g} K / pO2 {pO2:g}. No carrier density is "
                           f"reported, because the dilute-defect formalism does not apply."))
    elif occ > DILUTE_OCC_WARN or not strict_ok:
        out.update(verdict="MARGINAL", result=res,
                   reason=(f"site occupancy {occ:.3g}" if occ > DILUTE_OCC_WARN else
                           "strict validity gate failed for a non-occupancy reason") +
                          " — a value is returned for inspection but must not be quoted.")
    else:
        out.update(verdict="ADMISSIBLE", result=res,
                   reason=f"dilute (occupancy {occ:.3g}) and the strict validity gate passed.")
    return out
