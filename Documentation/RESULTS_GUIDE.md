# Results guide

All generated and included outputs live under `Results/`. Simulation tables
are saved in both CSV (easy inspection/analysis) and strict JSON (structured
records and resume checkpoints). Blank CSV statistical values correspond to
JSON `null`; non-standard JSON `NaN` values are never written.

## Simulation tables

The canonical folder is `Results/Simulation/Tables/`.

| File pair or file | Rows | Description |
|---|---:|---|
| `dataset_info.csv/json` | 4 | Dataset size, input count, prevalence, missingness provenance, source hash, subject count, and privacy unit. |
| `nonprivate_results.csv/json` | 100 | One row per dataset, model, and seed for five non-private baselines. Includes split sizes, public learning rate, artifact path when applicable, and AUC/PRAUC/accuracy/Brier metrics. |
| `private_lr_selection.csv/json` | 20 | One public-only learning-rate selection per dataset and seed for QNN, MLP, and logistic families, plus preprocessor artifact provenance. |
| `private_results.csv/json` | 600 | One row per dataset, six private models, five epsilon targets, and five seeds. Includes complete clipping, sampling, accounting, public-gradient diagnostic, artifact, split, and test-metric fields. |
| `ablation_results.csv/json` | 330 | Completed epsilon-1 DP-QNN feature/layer/encoding configurations with the same accounting and test fields. |
| `ablation_skipped.csv/json` | 30 | Valid grid requests that could not run because the requested PCA component count exceeded available WESAD features. |
| `model_summary.csv/json` | 140 | Mean, standard deviation, and count grouped by training type, dataset, model, and optional epsilon. |
| `corrected_statistical_tests.csv/json` | 24 | Seed-paired QNN-versus-MLP differences, bootstrap interval, Wilcoxon p-value, and Holm-adjusted p-value by dataset/scope. |
| `privacy_accounting.csv/json` | 600 | Compact extraction of target/realized epsilon, delta, sigma, sample rate, steps, privacy units, randomness, and academic-simulation flag. |
| `environment.json` | object | Python/platform and relevant package-version fingerprint for provenance. |

The three completed-fit tables contain 100 + 600 + 330 = 1,030 rows. Included
artifact fields are project-relative and resolve from the package root.

### Important columns

- `dataset`, `model`, `seed`: unique experiment identity components.
- `eps_target` / `target_eps`: requested epsilon; `realized_eps`: accountant
  result.
- `C`: clipping threshold; `sigma`: noise multiplier; their product determines
  per-coordinate Gaussian noise scale before normalization.
- `sample_rate`, `steps`, `steps_per_epoch`, `expected_batch_units`: sampling
  and accounting schedule.
- `private_rows` / `private_units`: rows and adjacency units can differ for
  WESAD because one subject owns many windows.
- `preprocessing_fit`: provenance showing preprocessing is public-only.
- `auc`, `prauc`, `acc`, `brier`: public-test performance metrics.
- `threshold_status`, `qnn_certificate`: whether QNN clipping is empirical or
  at the analytic certificate.
- `randomness`: academic seeded-randomness provenance.

## Simulation artifacts

`Results/Simulation/Artifacts/` contains 1,010 files:

- 20 `DATASET_seedN_public_preprocessor.joblib` fitted public preprocessors;
- 60 `nonprivate/DATASET/MODEL_seedN.pt` Torch/PennyLane model states;
- 600 `private/DATASET/MODEL/epsE_seedN.pt` DP model states;
- 330 `private_ablation/DATASET/featuresF_layersL_ENCODING_seedN.pt` DP-QNN
  ablation states.

Each `.pt` file is a PyTorch checkpoint containing model state and metadata.
Only load checkpoints from trusted sources because binary serialization formats
may execute code. Every concrete artifact is listed with SHA-256 in
`FILE_MANIFEST.csv`.

## IBM QPU results

The canonical folder is `Results/IBM_QPU/`.

| File | Description |
|---|---|
| `ibm_local_validation.json` | Exact StatevectorEstimator probabilities, local metrics, circuit metadata, fixed-case provenance, and Qiskit version. No QPU time is used. |
| `ibm_qpu_submission.json` | Job ID, backend, submission time, shots, bindings, and execution cap saved immediately after submission. |
| `ibm_qpu_results.json` | Completed backend/job metadata, exact and QPU probabilities, descriptive metrics, circuit depth/size, QPU usage, and exact-agreement errors. This is the input to the IBM figure. |
| `ibm_fez_calibrations_2026-07-24T03_05_15Z.csv` | Timestamped device calibration snapshot used for hardware context. It is not required by the figure scripts. |

The hardware record contains eight fixed balanced Pima cases evaluated for two
checkpoints (16 bindings) at 1,024 shots per binding. It is a descriptive
inference-feasibility experiment, not an independently powered clinical test.

## Figure results

`Results/Figures/Generated/` contains both PDF and 400-DPI PNG versions of:

| Stem | Source data | Meaning |
|---|---|---|
| `fig_privacy_utility` | private + non-private simulation CSVs | Target epsilon against test ROC AUC, with non-private QNN and random-ranking references. |
| `fig_clipping` | private simulation CSV | Curve-average AUC and epsilon-1 injected noise scale across clipping thresholds. |
| `fig_ablation` | ablation simulation CSV | Encoding, PCA component, and variational-depth comparisons. |
| `fig_ibm_validation` | completed IBM JSON | Hardware-versus-exact probability agreement and per-case signed errors. |

`Results/Figures/Data/` contains:

- `paper_model_summary.csv`: combined non-private, private, and curve-average
  AUC summaries;
- `paper_ablation_summary.csv`: grouped ablation means/standard deviations;
- `paper_corrected_tests.csv`: copied numeric corrected-test table;
- `paper_privacy_accounting.csv`: grouped accounting values used in reporting;
- `paper_results_summary.json`: complete numeric payload including row counts
  and IBM values.

These files are generated by `Code/FigureGeneration/generate_summaries.py`.
