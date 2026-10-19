# Unit-Aware Differentially Private Quantum Machine Learning for Clinical and Wearable Health Data

This repository contains the code, prepared data, trained checkpoints, result
records, and figure scripts for the paper:

> Md Khalid Syfullah and Alvi Ataur Khalil, "Unit-Aware Differentially Private
> Quantum Machine Learning for Clinical and Wearable Health Data."
> Computer Science, Southern Illinois University Carbondale, USA.

The study asks whether, at matched input dimensionality and privacy budget, a
quantum neural network (QNN) retains more utility than classical models when
trained with differentially private stochastic gradient descent (DP-SGD). It
defines a unit-aware DP-SGD framework that protects each clinical record, or
all windows from one wearable participant, as a single privacy unit.

The project is an academic reproducibility package. It is not a clinical
decision system and does not claim quantum advantage: no QNN-versus-MLP
difference remains significant after Holm correction.

## Research questions

The code reproduces the experiments for the paper's three research questions:

- **RQ1:** Does a capacity-matched QNN retain more utility than classical
  baselines under record-level and participant-level DP?
- **RQ2:** How do the clipping threshold, encoding gate, parameter count, and
  circuit depth affect predictive performance under DP?
- **RQ3:** Do predictions from NISQ hardware (IBM `ibm_fez`) agree with exact
  statevector inference for the same trained circuits?

## Datasets and privacy units

| Dataset | Samples | Inputs | Positive % | Privacy unit | Units (train/cal./test) | Sampling rate `q` | Noise `σ` at ε = 1 |
|---|---:|---:|---:|---|---|---:|---:|
| Pima Diabetes | 768 | 8 | 34.9 | one record | 460/154/154 | 0.067 | 5.39 |
| UCI Heart (Cleveland) | 303 | 13 | 45.9 | one record | 181/61/61 | 0.167 | 8.44 |
| Heart Failure | 299 | 12 | 32.1 | one record | 179/60/60 | 0.167 | 8.44 |
| WESAD | 438 windows | 7 | 35.4 | one participant | 9/3/3 participants | 1.000 | 20.31 |

These values correspond to Table II of the paper and can be checked in
`Results/Simulation/Tables/dataset_info.csv` and
`Results/Simulation/Tables/privacy_accounting.csv`.

- Clinical datasets use stratified 60/20/20 splits into private training,
  public calibration, and public test partitions.
- WESAD uses 15 wrist-sensor participants with 438 one-minute windows
  (baseline versus stress). For each seed, participants are split 9/3/3, and
  all windows of a participant stay in one partition. During private training,
  one participant is sampled, its mean window loss is differentiated, and its
  gradient is clipped as one unit.
- Median imputation, standardization, PCA, and `[0, π]` angle rescaling are
  fitted on the public calibration partition only, then applied unchanged to
  the private training and test partitions.

## Models

All models receive the same `d = 4` public-fitted PCA components (capacity
matching means the same input dimensionality).

| Model | Description | Trainable parameters |
|---|---|---:|
| QNN | 4 qubits; RY angle encoding; `L = 2` layers, each with four trainable RY rotations followed by a CNOT ring 0→1→2→3→0; output `pθ(x) = (1 + ⟨Z0⟩)/2` | 8 |
| MLP | 4 inputs, 2 tanh hidden units, 1 sigmoid output (smallest width with at least as many parameters as the QNN) | 13 |
| LR | Differentiable (PyTorch) logistic regression | 5 |

QNN simulation uses PennyLane `lightning.qubit` with adjoint differentiation.
The QNN, MLP, and LR are trained with mean squared error loss. Reported metrics are ROC
AUC (primary), PR-AUC, accuracy at threshold 0.5, and Brier score.

## Unit-aware DP-SGD

At each step every privacy unit is sampled independently with probability `q`
(Poisson sampling). Each sampled unit's gradient is clipped to norm `C`, the
clipped gradients are summed, Gaussian noise with standard deviation `σC` is
added, and the result is divided by the expected batch size `nq`. Opacus RDP
accounting selects `σ` for each target budget. Every run uses `δ = 1e-5`, 25
epochs, and an expected batch size of at most 32 units
(`q = 1 / ceil(n / 32)`).

**Gradient sensitivity certificate.** For the MSE loss, each coordinate of a
unit's QNN gradient is bounded by 1, so its L2 norm is at most `√P`. With
`P = 8`, `C = √8 ≈ 2.83` guarantees that clipping is inactive
(`qnn_Cbound_certified`). Thresholds `C = 0.5` and `C = 1` are empirical
choices that reduce the noise scale `σC` but may clip gradients.

## Experiment grid

| Experiment | Configuration | Runs |
|---|---|---:|
| Private training | 6 model–clipping combinations (QNN `C ∈ {0.5, 1, √8}`, MLP `C ∈ {0.5, 1}`, LR `C = 1`) × 5 budgets `ε ∈ {0.5, 1, 2, 4, 8}` × 4 datasets × 5 seeds | 600 |
| Non-private training | 5 models (QNN, MLP, PyTorch LR, scikit-learn LR, random forest) × 4 datasets × 5 seeds | 100 |
| Circuit ablation | encoding {RY, RX} × PCA components {4, 6, 8} × layers {1, 2, 3} at `ε = 1`, `C = 1`, × 4 datasets × 5 seeds, excluding 30 WESAD runs that need 8 components (WESAD has 7 inputs) | 330 |
| **Total training runs** | | **1,030** |

Notes on the non-private runs:

- The `ε = ∞` row of Table IV in the paper uses the QNN, MLP, and PyTorch LR
  non-private runs. Clipping is not used without DP, so the same non-private
  QNN result appears in the three QNN clipping columns, and the same MLP
  result appears in both MLP columns.
- The scikit-learn logistic regression and random forest runs are
  supplementary references stored in `nonprivate_results.csv`. They are not
  reported in the paper.

**Statistical evaluation.** The predeclared primary comparison is
`qnn_C1.0_empirical` versus `mlp_C1.0`, paired by seed. There are 20
per-budget tests and 4 curve-average tests (each model is first averaged across
the five budgets within a seed), giving 24 two-sided Wilcoxon signed-rank tests
with Holm correction. Descriptive 95% intervals use 10,000 paired bootstrap
resamples.

**IBM hardware experiment.** The seed-0 Pima non-private QNN checkpoint and the
private checkpoint (`C = 1`, target `ε = 1`, realized `ε = 0.995`) are run on
the first four negative and first four positive public-test cases. Both
checkpoints are submitted in one Qiskit Runtime Estimator job on `ibm_fez`
(16 bindings, 1,024 shots each, resilience level 0, gate and measurement
twirling disabled, optimization level 3, transpiler seed 2026).

## Main results included in this repository

All values below can be regenerated from the files in `Results/`.

| Result | Value | Source file |
|---|---|---|
| Curve-average QNN − MLP AUC at `C = 1` | Pima +0.104, UCI Heart +0.056, Heart Failure +0.070, WESAD −0.162 | `corrected_statistical_tests.csv` |
| Holm-adjusted p-values | 1.000 for all 24 tests | `corrected_statistical_tests.csv` |
| Largest gap between target and realized ε | 0.0082 | `privacy_accounting.csv` |
| IBM mean absolute probability error vs. exact | 0.046 (private), 0.063 (non-private) | `ibm_qpu_results.json` |
| IBM circuit | logical depth 11; compiled depth 59 with 103 instructions; 6 quantum seconds | `ibm_qpu_results.json` |
| IBM agreement | both checkpoints reproduce the eight simulated labels; 5 of 8 cases correct | `ibm_qpu_results.json` |

### Paper-to-file map

| Paper item | Repository file(s) | Generator |
|---|---|---|
| Table II | `dataset_info.csv`, `privacy_accounting.csv` | `prepare_data.py`, `analyze_results.py` |
| Table IV | `private_results.csv`, `nonprivate_results.csv`, `model_summary.csv` | `run_private.py`, `run_nonprivate.py`, `analyze_results.py` |
| Table V | `corrected_statistical_tests.csv` | `analyze_results.py` |
| Fig. 4 | `fig_privacy_utility.pdf/png` | `plot_privacy_utility.py` |
| Fig. 5 | `fig_clipping.pdf/png` | `plot_clipping.py` |
| Fig. 6 | `fig_ablation.pdf/png` | `plot_ablation.py` |
| Fig. 7 | `fig_ibm_validation.pdf/png` | `plot_ibm_validation.py` |
| Gradient-norm measurements (Sec. V-D) | `public_grad_norm_mean`, `public_grad_norm_max`, `public_clip_frac` columns of `private_results.csv` | `run_private.py` |

Figs. 1–3 (concept, circuit, and system diagrams) are not generated by this
code. The gradient-norm columns are computed with each trained private model
on the public calibration partition; they do not measure clipping during
private training.

## Project organization

```text
Unit-Aware-DP-QML-Health/
├── README.md
├── Code/
│   ├── Simulation/          data preparation, training, DP, and analysis
│   ├── IBM_QPU/             exact validation and optional IBM execution
│   ├── FigureGeneration/    one runnable script for each result figure
│   ├── build_manifest.py    file inventory and SHA-256 generator
│   └── validate_project.py  structure and result-integrity validator
├── Data/
│   └── Simulation/          raw and prepared simulation inputs
├── Results/
│   ├── Simulation/
│   │   ├── Tables/          CSV and strict-JSON experiment records
│   │   └── Artifacts/       fitted preprocessors and model checkpoints
│   ├── IBM_QPU/             local, submission, QPU, and calibration records
│   └── Figures/
│       ├── Generated/       PDF and PNG result figures
│       └── Data/            numeric data supporting the figures
└── Documentation/
    ├── PROJECT_DESCRIPTION.md
    ├── METHODOLOGY.md
    ├── ARCHITECTURE.md
    ├── PARAMETERS.md
    ├── RESULTS_GUIDE.md
    ├── FILE_REFERENCE.md
    ├── REPRODUCIBILITY.md
    └── FILE_MANIFEST.csv
```

Executable code is separate from input data and generated results. Simulation
and IBM programs write to their own `Results/` folders. Figure programs read
those result files and do not retrain models.

The included results contain 100 non-private fits, 600 private fits, 330
completed ablation fits, 30 recorded ablation skips, 1,010 fitted
preprocessors and model checkpoints, one completed IBM Runtime job with 16
parameter bindings, and four result figures in PDF and PNG.

## Installation

Use separate environments for simulation, IBM, and plotting because their
dependency stacks are independent.

### Simulation environment

The included results were produced with Python 3.9, PennyLane 0.38.0, PyTorch
2.8.0, Opacus 1.6.0, scikit-learn 1.6.1, NumPy 1.26.4, and pandas 2.3.3
(`Results/Simulation/Tables/environment.json`). `requirements.txt` pins these
versions. Python 3.9 is recommended for exact reproduction:

```bash
python3.9 -m venv .venv-simulation
source .venv-simulation/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Code/Simulation/requirements.txt
```

### IBM environment

Qiskit 2.4 requires Python 3.10 or newer. The hardware run used Qiskit 2.4.2
and Qiskit IBM Runtime 0.46.1:

```bash
python3 -m venv .venv-ibm
source .venv-ibm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Code/IBM_QPU/requirements.txt
```

### Figure environment

```bash
python3 -m venv .venv-figures
source .venv-figures/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Code/FigureGeneration/requirements.txt
```

## Run the simulation

First run the small software-validation workflow:

```bash
python Code/Simulation/run_all.py --demo --workers 1
```

Demo outputs use `_demo` filenames and do not replace the full result tables.
Demo metrics are not research results.

Run the complete experiment:

```bash
python Code/Simulation/run_all.py
```

The worker count is selected automatically from the available CPU cores
(leaving two free), with a maximum of eight. To set a smaller number:

```bash
python Code/Simulation/run_all.py --workers 4
```

The workflow runs these stages in order:

```bash
python Code/Simulation/prepare_data.py
python Code/Simulation/run_nonprivate.py --workers 4
python Code/Simulation/run_private.py --workers 4
python Code/Simulation/run_ablation.py --workers 4
python Code/Simulation/analyze_results.py
```

Simulation inputs are read from `Data/Simulation/`. Result tables are written
to `Results/Simulation/Tables/`, and fitted states are written to
`Results/Simulation/Artifacts/`. Private and ablation stages save progress
after completed jobs and resume missing configurations when restarted.

## Optional WESAD preparation

The prepared WESAD feature table (`Data/Simulation/wesad_clean.csv`) is
included. To rebuild it, place a verified official WESAD extraction at:

```text
Data/Simulation/WESAD/S2/S2.pkl
Data/Simulation/WESAD/S3/S3.pkl
...
```

or specify another location:

```bash
python Code/Simulation/prepare_data.py --wesad-path /path/to/WESAD
```

WESAD uses Python pickle files. Pickle loading can execute code, so only use a
trusted official dataset copy. `--no-trust-wesad-pickle` makes the program
refuse pickle loading.

## Run IBM validation

The default command performs exact statevector validation and uses no QPU
time:

```bash
python Code/IBM_QPU/run_ibm_qpu.py
```

It writes `Results/IBM_QPU/ibm_local_validation.json`.

The following command submits a real IBM Runtime job and may consume an IBM QPU
allocation:

```bash
python Code/IBM_QPU/run_ibm_qpu.py --hardware
```

Without `--backend`, the least busy eligible backend is selected. To repeat the
paper's setting, name the backend explicitly:

```bash
python Code/IBM_QPU/run_ibm_qpu.py \
  --hardware \
  --backend ibm_fez \
  --output-dir Results/IBM_QPU
```

The included completed QPU result is sufficient to reproduce the IBM figure.
A new hardware submission is not needed for ordinary reproduction.

## Generate figures

Each figure has one entry-point script that reads specific result files from
`Results/`:

| Paper figure | Command | Required result files |
|---|---|---|
| Fig. 4, privacy and utility | `python Code/FigureGeneration/plot_privacy_utility.py` | `private_results.csv`, `nonprivate_results.csv` |
| Fig. 5, clipping sensitivity | `python Code/FigureGeneration/plot_clipping.py` | `private_results.csv` |
| Fig. 6, QNN ablation | `python Code/FigureGeneration/plot_ablation.py` | `ablation_results.csv` |
| Fig. 7, IBM QPU validation | `python Code/FigureGeneration/plot_ibm_validation.py` | `ibm_qpu_results.json` |

Run every result figure and numeric summary:

```bash
python Code/FigureGeneration/generate_all_figures.py
```

Outputs are written to `Results/Figures/Generated/` and
`Results/Figures/Data/`.

## Main parameters

Scientific settings are stored in `Code/Simulation/config.json`.

| Parameter | Default |
|---|---|
| Datasets | Pima, UCI Heart, Heart Failure, WESAD |
| Seeds | 0, 1, 2, 3, 4 |
| Target ε | 0.5, 1, 2, 4, 8 |
| δ | `1e-5` |
| Public calibration fraction | 0.2 |
| Test fraction | 0.2 |
| Primary QNN | 4 PCA components (qubits), 2 layers, RY encoding |
| Non-private epochs | 25 |
| Private epochs | 25 |
| Expected DP batch | at most 32 units |
| Primary comparison | `qnn_C1.0_empirical` vs. `mlp_C1.0` |
| Ablation | 4/6/8 components, 1/2/3 layers, RY/RX encoding, ε = 1 (`C = 1` is fixed in `run_ablation.py`) |

Data preparation creates the default configuration only when the file is
missing and preserves user edits. `Documentation/PARAMETERS.md` describes every
configuration key, model identifier, fixed IBM constant, and command-line
option.

## Included documentation

- `PROJECT_DESCRIPTION.md` explains the motivation, research questions,
  datasets, models, experiments, and scope.
- `METHODOLOGY.md` describes splitting, public preprocessing, QNNs, baselines,
  DP training, accounting, ablation, statistics, and IBM execution.
- `ARCHITECTURE.md` explains code modules and data/result flow.
- `PARAMETERS.md` documents every configuration and CLI parameter.
- `RESULTS_GUIDE.md` describes every result table, checkpoint family, IBM file,
  and figure output.
- `FILE_REFERENCE.md` describes every source, data, result, and document file
  or file pattern.
- `REPRODUCIBILITY.md` gives step-by-step validation and rerun instructions.
- `FILE_MANIFEST.csv` lists every delivered file with category, description,
  size, and SHA-256 digest.

## Limitations

- The public calibration and test partitions are assumed to be public. If
  every record is private, end-to-end privacy requires a separately accounted
  private preprocessing method.
- Seeded pseudorandomness supports academic reproduction; deployed private
  learning requires secret randomness and protected data handling.
- WESAD has only nine private training participants per seed, so participant
  level DP needs a large noise multiplier (`σ = 20.31` at `ε = 1`). A larger
  wearable dataset is needed to assess real performance.
- The IBM experiment uses eight fixed Pima cases for each of two checkpoints on
  one device. It evaluates simulator-to-hardware agreement, not clinical
  generalization or quantum advantage.
- Results apply to the included datasets, model families, hyperparameter grid,
  and software and hardware conditions.

## Project validation

Validate paths, JSON, table counts, artifact references, and figure files:

```bash
python3 Code/validate_project.py
```

Rebuild the checksummed file inventory after intentional changes:

```bash
python3 Code/build_manifest.py
```

## AI Usage Statement

AI tools were used to assist with this repository. Specifically, AI assistance
was used to:

- validate the consistency of the code, configuration, documentation, and
  result records against the paper;
- organize the dataset files, result records, and repository structure; and
- edit and format the README and supporting documentation.

All AI-assisted outputs were reviewed and verified by the authors, who take
full responsibility for the content of this repository and the accompanying
paper.
