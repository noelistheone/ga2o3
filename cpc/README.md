# Differentiable defect-thermodynamic sandbox

Source code for **A differentiable defect-thermodynamic sandbox for dopant
activation prediction**.

The sandbox maps incorporated dopant concentration and preparation conditions to
frozen defect inventories, low-temperature charge balance, and Hall carrier
response. Implicit differentiation propagates learning signals through the
coupled physical roots. The implementation supports inventory learning,
variable-temperature donor activation, and property-directed acquisition of
missing energetic labels.

This directory contains source code, dependency specifications and this README.
Experimental datasets, fitted models, generated results and manuscript files are
maintained separately. All commands below run from `cpc/`.

## Installation and a first prediction

Use Python 3.11. The CPC implementation runs on a CPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/prepare_kroger_input.py
python research/scripts/trainable_sandbox_20261017/phase15_build_fermi.py
python examples/predict_example.py --element Si --dose 1e18
```

The preparation command downloads the KROGER energetic input from its original
repository, verifies its checksum and extracts it locally. The second command
builds a mathematical Fermi-integral interpolation table by quadrature; it uses
no experimental labels. The example returns a carrier prediction and its
parameter derivatives at explicit demonstration coordinates, without presenting
those coordinates as a trained model.

## Train on your input table

```bash
python scripts/sandbox_workflow.py fit --kind inventory \
  --input /path/to/inventory_inputs.csv --held HELD_OUT_GROUP \
  --max-nfev 160 --output runs/inventory
python scripts/sandbox_workflow.py predict \
  --model runs/inventory/model.json --input /path/to/query_inputs.csv \
  --output runs/prediction.csv
```

The physical predictor uses seven fields:

| Field | Meaning / units |
| --- | --- |
| `element` | `Si` or `Sn` |
| `incorporated_cm3` | Incorporated chemical dopant concentration, cm^-3 |
| `Tobs_K` | Measurement temperature, K |
| `growth_T_C` | Growth temperature, degrees C |
| `H_known` | Boolean; whether the hydrogen input is supplied |
| `H_cm3` | Hydrogen concentration, cm^-3; use zero when `H_known=False` |
| `scenario_pO2_atm` | Explicit oxygen-pressure scenario, atm |

For inventory training, add `target` (log10 Hall concentration in cm^-3),
`family` (publication ID), `sample_id`, `related_cluster` and `scenario_weight`.
The value passed to `--held` identifies the excluded `related_cluster`. Training
uses a fixed response-prior weight of 0.1 and two prescribed starts. The output
records the input checksum, protocol, all optimizer starts, fitted coordinates
and held-out predictions. Use a new output directory for each completed fit.

Thermal training uses the same physical fields plus `target`, `family`,
`sample_id` and `kind` (`RT` for room-temperature endpoints). Here `--held`
identifies the excluded `family`:

```bash
python scripts/sandbox_workflow.py fit --kind thermal \
  --input /path/to/thermal_inputs.csv --held HELD_OUT_PUBLICATION \
  --derivative IFT --output runs/thermal_ift
```

Change `--derivative` to `FD` and choose a new output directory to compare the
finite-difference derivative path under the same starts and optimizer settings.
Source identifiers and observed targets are used for fitting and evaluation;
they are not inputs to the physical prediction interface.

## Property-directed energetic acquisition

After preparing KROGER, these commands run a fresh energy-surrogate demonstration,
select masked energetic labels using the physical sensitivities, and evaluate
separate virtual query conditions:

```bash
python scripts/sandbox_workflow.py acquire --element Si --budget 2 --output runs/acquisition_si
python scripts/sandbox_workflow.py acquire --element Sn --budget 2 --output runs/acquisition_sn
```

This is retrospective acquisition from an existing energetic catalogue. The
selected labels are not new first-principles calculations. No electrical labels
enter the acquisition criterion.

## Numerical checks

```bash
python scripts/benchmark_cpc.py --output runs/numerics
python scripts/benchmark_gradient_paths.py --output runs/gradient_paths
```

The first benchmark compares analytic implicit derivatives against genuine
value-only finite differences and an independent nested-bracketing root solver.
The second separates preparation-inventory and cold-occupation gradient paths.
Both save their protocols, coordinate-level checks and timings locally.

The following commands consume the separately maintained article data and
saved-model bundle, when those directories have been placed alongside this
README:

```bash
python scripts/verify_metrics.py
python scripts/summarize_submission_validation.py
python scripts/summarize_gradient_paths.py
python scripts/replay_inventory.py
python scripts/make_tables.py
python scripts/make_figures.py
python scripts/make_cpc_figure.py
```

`verify_metrics.py` reads `source_data/`; the submission and gradient summaries
also use `verification/`; model replay uses archived fit records under
`research/results/`. These data and outputs are not included in this code-only
repository addition. The portable `sandbox_workflow.py` commands above train
on user-supplied tables without the historical stage launchers.

## Source layout

| Path | Purpose |
| --- | --- |
| `scripts/sandbox_workflow.py` | Portable fitting, prediction and acquisition CLI |
| `scripts/cpc_numerics.py` | Value-only and derivative benchmark interfaces |
| `research/src/sandbox/` | Physical solvers, charge statistics, transport and IFT |
| `research/scripts/trainable_sandbox_20261017/` | Original learning and evaluation implementations |
| `research/scripts/predictive_redesign_20261019/` | Inventory and transport model controls |
| `examples/predict_example.py` | Label-free prediction API example |

The `research/` layout preserves import paths of the article implementation.
Historical stage drivers document the full experiments and may expect their
original input/result directories; the portable entry points are listed above.

## Upstream input and license

Project code follows the repository's [MIT license](../LICENSE). The external
KROGER catalogue retains its original rights and is downloaded separately.

- Source: [KROGER](https://github.com/mikescarpulla/KROGER), commit
  `b0b8ce9d092d7208410d31b556d96ff37efd3b86`.
- Input: `Ga2O3/Ga2O3_Varley_all_defects_new_092724.mat`.
- SHA-256: `0a7508c2682d206954af16b125ca948aaf3bbfc843de252effafd3e9d79d3a43`.
- Reference: K. A. Arnab et al., *Physical Chemistry Chemical Physics* **27**,
  11129–11143 (2025), [doi:10.1039/D4CP04817B](https://doi.org/10.1039/D4CP04817B).
