# Reproducibility guide

Run commands from the package root. Separate environments are recommended.

## Reproduce included figures without retraining

```bash
python3 -m venv .venv-figures
source .venv-figures/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Code/FigureGeneration/requirements.txt
python Code/FigureGeneration/generate_all_figures.py
```

This reads the project CSV/JSON records under `Results/` and refreshes only
`Results/Figures/`. It neither imports training frameworks nor contacts IBM.

To generate one figure, run its corresponding `plot_*.py` script. Every script
supports `--help`, explicit input overrides, and `--output-dir`.

## Simulation smoke workflow

The included results were produced with Python 3.9 and the package versions
recorded in `Results/Simulation/Tables/environment.json`, which
`requirements.txt` pins. Python 3.9 is recommended:

```bash
python3.9 -m venv .venv-simulation
source .venv-simulation/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Code/Simulation/requirements.txt
python Code/Simulation/run_all.py --demo --workers 1
```

Demo files use `_demo` names in `Results/Simulation/Tables/` and do not replace
the included full tables. The demo verifies imports, data paths, training,
accounting, saving, and analysis; its metrics are not research results.

## Complete simulation

Review `Code/Simulation/config.json` and `Documentation/PARAMETERS.md`, then run:

```bash
python Code/Simulation/run_all.py
```

The full run resumes completed private/ablation configurations. For a controlled
worker count:

```bash
python Code/Simulation/run_all.py --workers 4
```

Inputs, tables, and artifacts are written to their canonical `Data/` and
`Results/` locations. The included run contains 100 non-private, 600 private,
330 completed ablation, and 30 intentionally skipped ablation records.

## Optional WESAD rebuild

Prepared WESAD features are delivered. To rebuild, obtain the official dataset,
verify it independently, and place its subject folders under
`Data/Simulation/WESAD/`. Loading pickle files is inherently unsafe; never use
an untrusted copy.

```bash
python Code/Simulation/prepare_data.py --wesad-path Data/Simulation/WESAD
```

Use `--no-trust-wesad-pickle` to make the program refuse any discovered pickle.

## IBM local validation

Use Python 3.10 or newer (required by Qiskit 2.4):

```bash
python3 -m venv .venv-ibm
source .venv-ibm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Code/IBM_QPU/requirements.txt
python Code/IBM_QPU/run_ibm_qpu.py
```

This consumes no QPU time. It refreshes the exact reference record in
`Results/IBM_QPU/ibm_local_validation.json`.

The included `ibm_qpu_results.json` is sufficient to reproduce reported
hardware quantities. Do not invoke `--hardware` unless a new real job is
intended, authorized, and budgeted.

## Static checks

From the root, syntax-check all Python files and validate JSON/CSV structure:

```bash
python3 -m compileall -q Code
python3 Code/validate_project.py
```

Refresh the per-file inventory after intentional file changes:

```bash
python3 Code/build_manifest.py
```
