# Parameter reference

This document describes all user-facing configuration keys, command-line
arguments, and fixed IBM/figure parameters in this project.

## Simulation configuration

The configuration file is `Code/Simulation/config.json`. JSON keys are read at
the beginning of each simulation stage. `prepare_data.py` creates defaults only
when the file does not exist, so edits are not silently overwritten.

### Version and experiment grid

| JSON path | Default | Description |
|---|---|---|
| `code_version` | `4.0` | Must match `QPRIV_VERSION` in `qpriv_utils.py`; prevents accidental mixing of incompatible records. |
| `randomness_protocol_version` | `3.1` | Provenance label for deterministic seed derivation. |
| `datasets` | `pima`, `heart`, `heart_failure`, `wesad` | Ordered dataset names. A dataset runs only if `Data/Simulation/NAME_clean.csv` exists. |
| `eps_grid` | `0.5, 1, 2, 4, 8` | Target differential-privacy epsilon values. Lower epsilon requests stronger privacy and generally more noise. |
| `delta` | `1e-5` | Target DP failure probability passed to the RDP accountant. |
| `seeds` | `0, 1, 2, 3, 4` | Independent split, initialization, sampling, and noise repetitions. |

### Data partition

| JSON path | Default | Description |
|---|---:|---|
| `split.public_size` | `0.2` | Fraction reserved for public preprocessing and learning-rate selection. |
| `split.test_size` | `0.2` | Fraction reserved for public test evaluation. |
| `split.public_seed` | `2026` | `random_state` of the public-only PCA fit, also stored as provenance. It does not select partitions; all three partitions are drawn from the experiment seed. |

The remaining fraction is private training data. The clinical datasets use
record-level privacy units. WESAD uses subjects, so all windows for a subject
stay in the same partition and are sampled/clipped together.

### Primary QNN

| JSON path | Default | Description |
|---|---|---|
| `qnn.n_features` | `4` | Number of public-only PCA components and therefore the number of qubits. |
| `qnn.n_layers` | `2` | Number of trainable single-qubit-rotation plus ring-entanglement layers. |
| `qnn.encoding` | `ry` | Input-encoding gate; supported values are `ry` and `rx`. The trainable variational rotations are always RY. |
| `qnn.device` | `lightning.qubit` | Exact PennyLane simulator backend. |
| `qnn.diff_method` | `adjoint` | Exact gradient method used by the simulator. |

### Parallel execution

| JSON path | Default | Description |
|---|---:|---|
| `parallel.start_method` | `spawn` | Records the worker start method. |
| `parallel.maximum_workers` | `8` | Records the maximum number of spawned processes. |
| `parallel.reserved_cpu_cores` | `2` | Records the cores left free when the worker count is automatic. |
| `parallel.threads_per_worker` | `1` | Records the Torch/BLAS thread limit inside each worker. |

These `parallel.*` keys document the execution policy but are not read by the
code; editing them has no effect. The policy is fixed in the code: workers are
started with `spawn`, each worker uses one Torch/BLAS thread, and
`worker_count()` in `qpriv_utils.py` uses `min(8, max(1, CPU count - 2))`
by default, with an upper bound equal to the number of pending tasks.
`--workers N` can request a different value; requests above 8 are capped at 8.

### Training

| JSON path | Default | Description |
|---|---:|---|
| `training.nonprivate_epochs` | `25` | Epochs for QNN, matched MLP, and Torch logistic non-private fits. |
| `training.private_epochs` | `25` | Epochs for every DP training and ablation fit. |
| `training.expected_batch_units` | `32` | Upper bound on the expected privacy units per Poisson-sampled DP step; the sampling rate is `1 / ceil(n / 32)` for `n` private units. |
| `training.lr_candidates.qnn` | `0.02, 0.05, 0.08` | Public calibration learning-rate candidates for QNNs. |
| `training.lr_candidates.mlp` | `0.01, 0.03, 0.05, 0.08` | Public calibration candidates for matched MLPs. |
| `training.lr_candidates.logistic` | `0.01, 0.03, 0.05, 0.08` | Public calibration candidates for Torch logistic models. |

Learning-rate selection uses public data only. The non-private Torch models use
three restarts in a full run. Demo mode reduces epochs, candidates, restarts,
datasets, seeds, epsilon values, and model/grid scope for software validation.

### Private model identifiers

| Identifier | Family | Clipping threshold `C` | Interpretation |
|---|---|---:|---|
| `qnn_C0.5_empirical` | QNN | 0.5 | Empirical threshold below the analytic certificate. |
| `qnn_C1.0_empirical` | QNN | 1.0 | Predeclared primary QNN configuration. |
| `qnn_Cbound_certified` | QNN | `√8 ≈ 2.83` (computed as `√P`) | Certified no-clipping threshold. |
| `mlp_C0.5` | capacity-matched MLP | 0.5 | Empirical clipping. |
| `mlp_C1.0` | capacity-matched MLP | 1.0 | Predeclared primary classical comparator. |
| `logistic_C1.0` | logistic | 1.0 | Private linear baseline. |

`private_models` controls which of those names are included. The code rejects
unknown identifiers.

### Primary analysis

| JSON path | Default | Description |
|---|---|---|
| `primary_comparison.qnn` | `qnn_C1.0_empirical` | QNN used by corrected paired tests. |
| `primary_comparison.classical` | `mlp_C1.0` | Classical model used by corrected paired tests. |
| `primary_comparison.selection` | `predeclared_before_private_test_evaluation` | Provenance statement, not an execution switch. |

The analysis computes seed-paired differences, bootstrap confidence intervals,
Wilcoxon p-values, and Holm-adjusted p-values.

### Ablation grid

| JSON path | Default | Description |
|---|---|---|
| `ablation.n_features` | `4, 6, 8` | Public PCA component/qubit counts. |
| `ablation.n_layers` | `1, 2, 3` | Variational layer counts. |
| `ablation.encodings` | `ry, rx` | Rotation encodings. |
| `ablation.eps` | `1.0` | Target epsilon shared by the grid. |

The ablation clipping threshold `C = 1` is fixed in `run_ablation.py` and has
no configuration key.

Configurations requesting more PCA components than available features are
recorded in `ablation_skipped.*` rather than treated as failed fits.

## Simulation command-line arguments

| Script | Parameter | Default/behavior |
|---|---|---|
| `run_all.py` | `--demo` | Run a small smoke workflow and write `_demo` tables. |
| `run_all.py` | `--workers N` | Override automatic process count, capped by configuration. |
| `run_all.py` | `--trust-wesad-pickle` / `--no-trust-wesad-pickle` | Permit/refuse loading local WESAD pickle files; default permits a trusted local copy. |
| `prepare_data.py` | `--wesad-path PATH` | WESAD location; default is `Data/Simulation/WESAD`. Relative paths are resolved from the project root. |
| `prepare_data.py` | `--trust-wesad-pickle` / `--no-trust-wesad-pickle` | Same security control as above. |
| `run_nonprivate.py` | `--demo`, `--workers N` | Stage-specific smoke mode and worker override. |
| `run_private.py` | `--demo`, `--workers N` | Stage-specific smoke mode and worker override. |
| `run_ablation.py` | `--demo`, `--workers N` | Stage-specific smoke mode and worker override. |
| `analyze_results.py` | `--demo` | Analyze `_demo` inputs instead of full results. |

## Fixed non-private baselines

`run_nonprivate.py` always evaluates exact QNN, capacity-matched MLP, Torch
logistic, scikit-learn logistic regression, and random forest. The random
forest uses 300 estimators, minimum leaf size 3, one internal job, and the
experiment seed. Scikit-learn logistic regression uses at most 2,000
iterations. Change these constants only when intentionally defining a new
protocol version.

## IBM QPU constants

Fixed constants are declared near the top of `Code/IBM_QPU/run_ibm_qpu.py`:

| Constant | Value | Description |
|---|---:|---|
| `SHOTS` | `1024` | Shots requested for each parameter binding. |
| `MAX_QPU_SECONDS` | `540` | Runtime execution cap; it is not a completion guarantee. |
| `N_QUBITS` | `4` | Circuit and input dimension. |
| `N_LAYERS` | `2` | Trainable RY layers. |
| `TRANSPILER_SEED` | `2026` | Reproducible transpilation seed. |
| `TEST_POSITIONS` | `0,1,2,4,3,6,14,16` | Eight fixed positions in the seed-0 Pima public-test partition. |
| `PIMA_LABELS` | four 0s, four 1s | Balanced descriptive labels. |
| `PIMA_ANGLES` | embedded 8 by 4 array | Public-preprocessed input angles. |
| `NONPRIVATE_WEIGHTS` | embedded 2 by 4 array | Archived seed-0 non-private QNN weights. |
| `PRIVATE_WEIGHTS` | embedded 2 by 4 array | Archived epsilon-1 private QNN weights. |

Runtime settings are resilience level 0, gate twirling disabled, measurement
twirling disabled, preset pass-manager optimization level 3, and the logical Z
observable on qubit 0.

IBM command-line arguments are `--hardware` (real submission), `--backend NAME`
(explicit backend), and `--output-dir PATH` (default `Results/IBM_QPU`).

## Figure command-line arguments

All figure scripts accept `--output-dir`, defaulting to
`Results/Figures/Generated`.

| Script | Input parameter | Default |
|---|---|---|
| `plot_privacy_utility.py` | `--results-dir` | `Results/Simulation/Tables` |
| `plot_clipping.py` | `--input-file` | `Results/Simulation/Tables/private_results.csv` |
| `plot_ablation.py` | `--input-file` | `Results/Simulation/Tables/ablation_results.csv` |
| `plot_ibm_validation.py` | `--input-file` | `Results/IBM_QPU/ibm_qpu_results.json` |
| `generate_summaries.py` | `--results-dir`, `--ibm-json`, `--output-dir` | canonical simulation, IBM, and figure-data folders |

The plotting style fixes a 7.16-inch publication width, serif fonts, embedded
TrueType PDF fonts, a target epsilon x-grid of 0.5 through 8, and 400-DPI PNG
output. These are presentation constants and do not change experimental data.
