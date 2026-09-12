# ga2o3

Source code and data for defect-thermodynamics simulation and machine-learning
analysis of doped β-Ga₂O₃.

| Directory | Contents |
|---|---|
| `Ga2O3-Sandbox/` | Charge-neutrality solver with kinetic quench; transport, optical and device models; MLIP structure engine; hybrid calibration; experiment designer. Library in `src/sandbox/`, runnable steps in `scripts/`. |
| `Ga2O3-Net/` | Machine-learning models and evaluation code (`src/`, `scripts/`, `config/`), measured data tables (`data/`), trained model bundles (`results/deployment/`, `checkpoints/`). |

Keep the two directories side by side: some `Ga2O3-Sandbox` scripts read `../Ga2O3-Net/`.
Some scripts still contain absolute paths or raw-data paths from the original
workstation that are not part of this repository; adjust them before running those scripts.

## Installation

```bash
conda create -n ga2o3 python=3.11 -y && conda activate ga2o3
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# Ga2O3-Net graph models also need PyG:
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
pip install torch_geometric
```

`requirements.txt` covers the Sandbox library and the examples below. Other `Ga2O3-Net`
modules and scripts need further packages: `Ga2O3-Net/requirements.txt` lists the main
ones, and `requirements-ga2o3-full.txt` is a reference `pip freeze` of the original
environment (a few conda-built packages appear there as local build paths).

MatterSim runs in its own environment:
`conda create -n mattersim python=3.11 -y && conda activate mattersim && pip install mattersim==1.2.3`.

The MLIP structure engine (`Ga2O3-Sandbox/src/sandbox/structure_engine.py`) loads the
MACE-MPA-0 medium model from the file named by `MODEL_PATH`. Download
[`mace-mpa-0-medium.model`](https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model)
and set `MODEL_PATH` to its location.

## Usage

Sandbox — run from `Ga2O3-Sandbox/`. Import the library with `PYTHONPATH=src`;
launch scripts with `PYTHONPATH=.`.

```bash
cd Ga2O3-Sandbox
PYTHONPATH=src python -c "from sandbox import mu_o; print(float(mu_o.mu_o(1073, 1e-5)))"
PYTHONPATH=src python -c "from sandbox import transport as t; print(t.mobility(300., 1e18, 2e18, film=True, E_B_eV=0.02)['mu_hall'])"
PYTHONPATH=src python -m sandbox.optical
PYTHONPATH=src python -m sandbox.tau_nmp
```

Net — run from `Ga2O3-Net/` with `PYTHONPATH=.`.

```bash
cd Ga2O3-Net
PYTHONPATH=. python -c "
from src.data.codoping_oracle import build_batch, label_with_oracle
b = build_batch([[('Sn', 0.01)], [('Mg', 0.01)]], T_list=[973.0]*2, lpo2_list=[0.0]*2)
print(label_with_oracle(b, device='cpu')['E_F'].tolist())"
```

## Third-party data (not included)

The KROGER hybrid-functional defect database (Arnab et al., *Phys. Chem. Chem. Phys.*
27, 11129 (2025), doi:10.1039/D4CP04817B) is not redistributed. Steps that read it need
the database from its original source (see
`Ga2O3-Sandbox/data/energetics/kroger_ga2o3_092724.provenance.json`), placed in
`Ga2O3-Sandbox/external/KROGER/Ga2O3/`, followed by a one-time extraction:

```bash
cd Ga2O3-Sandbox && PYTHONPATH=. python scripts/t1_extract_kroger.py
```

Quantum ESPRESSO pseudopotentials are likewise not included.

## License

MIT — see [LICENSE](LICENSE).
