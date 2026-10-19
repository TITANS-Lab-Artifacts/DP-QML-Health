# File reference

This reference describes every named source/data/result/document file and every
repeating artifact pattern. The machine-readable `FILE_MANIFEST.csv` has one
row for every delivered file, including all 1,010 binary artifacts, with a
description, byte size, and SHA-256 digest.

## Root

| File | Description |
|---|---|
| `README.md` | Main project description, installation, execution, figure, parameter, result, and interpretation guide. |
| `.gitignore` | Ignores environments, caches, optional raw WESAD pickles, and demo outputs. |

## `Code/Simulation/`

| File | Description |
|---|---|
| `README.md` | Short simulation-folder guide. |
| `requirements.txt` | NumPy, pandas, PyTorch, PennyLane/Lightning, Opacus, and scikit-learn pinned to the versions that produced the included results, plus SciPy and joblib ranges. |
| `config.json` | Version, dataset, privacy, seed, split, QNN, worker, training, model, primary-comparison, and ablation parameters. |
| `prepare_data.py` | Verified clinical downloads, raw cleaning, optional trusted WESAD feature extraction, dataset summary, and manifest generation. |
| `qpriv_utils.py` | Canonical paths; atomic strict I/O; environment hashing; deterministic seeds; public-only preprocessing; QNN/MLP/logistic definitions; non-private and DP training; metrics; and privacy accounting. |
| `run_nonprivate.py` | Five non-private baselines over dataset/seed pairs, public learning-rate selection, artifact saving, resume logic, and safe workers. |
| `run_private.py` | Six DP model configurations over dataset/epsilon/seed, independent sampling/noise seeds, public family-level learning-rate selection, resume checkpoints, and artifacts. |
| `run_ablation.py` | Feature/layer/encoding DP-QNN grid with explicit skip records for unavailable dimensions. |
| `analyze_results.py` | Model summaries, paired bootstrap intervals, Wilcoxon tests, Holm correction, and compact accounting output. |
| `run_all.py` | Ordered five-stage workflow with progress, failure reporting, and adaptive time estimate. |

## `Code/IBM_QPU/`

| File | Description |
|---|---|
| `README.md` | QPU purpose, safe local command, hardware warning, outputs, and fixed defaults. |
| `requirements.txt` | NumPy, Qiskit, and Qiskit IBM Runtime constraints. |
| `run_ibm_qpu.py` | Four-qubit circuit, fixed public inputs/checkpoints, exact local validation, optional backend selection/transpilation/EstimatorV2 submission, and central result saving. |

## `Code/FigureGeneration/`

| File | Description |
|---|---|
| `README.md` | Commands and explicit absence of diagram generators. |
| `requirements.txt` | Matplotlib, NumPy, and pandas constraints. |
| `io_utils.py` | Canonical result/output paths and clear CSV/JSON input validation. |
| `figure_utils.py` | Shared names, visual style, save behavior, four quantitative plotting functions, and summary writer. It is a support module, not a command-line generator. |
| `plot_privacy_utility.py` | Loads private/non-private CSVs and creates only `fig_privacy_utility`. |
| `plot_clipping.py` | Loads the private CSV and creates only `fig_clipping`. |
| `plot_ablation.py` | Loads the ablation CSV and creates only `fig_ablation`. |
| `plot_ibm_validation.py` | Loads the completed IBM JSON and creates only `fig_ibm_validation`. |
| `generate_summaries.py` | Loads the necessary simulation/IBM files and refreshes numeric figure backing data. |
| `generate_all_figures.py` | Convenience caller for the four independent plot scripts and summary generator. |

## Project utilities under `Code/`

| File | Description |
|---|---|
| `build_manifest.py` | Creates `Documentation/FILE_MANIFEST.csv`, classifies every path, assigns a pattern-aware description, and computes SHA-256. The manifest's own row uses explicit `self` markers because a file cannot contain its own stable digest. |
| `validate_project.py` | Checks required paths, result row counts, strict JSON, artifact references, artifact count, generated figures, and exclusion of non-result diagrams. |

## `Data/Simulation/`

| File/pattern | Description |
|---|---|
| `raw/pima.csv` | Verified raw Pima source. |
| `raw/heart.csv` | Verified raw processed Cleveland Heart source. |
| `raw/heart_failure.csv` | Verified raw Heart Failure Clinical Records source. |
| `pima_clean.csv` | Named Pima columns, domain-coded missing values preserved, binary target. |
| `heart_clean.csv` | Cleaned/renamed Cleveland Heart inputs and binary target. |
| `heart_failure_clean.csv` | Cleaned Heart Failure inputs and target. |
| `wesad_clean.csv` | Subject-labelled WESAD window features and binary stress target. |
| `dataset_manifest.json` | Source URLs/hashes and WESAD subject-file provenance. |
| `WESAD/S*/S*.pkl` (optional, not delivered) | Trusted official raw WESAD extraction for rebuilding features. |

## `Results/Simulation/Tables/`

Every file is described in `RESULTS_GUIDE.md`: `dataset_info`,
`nonprivate_results`, `private_lr_selection`, `private_results`,
`ablation_results`, `ablation_skipped`, `model_summary`,
`corrected_statistical_tests`, and `privacy_accounting` each have CSV and JSON
forms; `environment.json` stores environment provenance.

## `Results/Simulation/Artifacts/`

| Pattern | Description |
|---|---|
| `DATASET_seedN_public_preprocessor.joblib` | Public-only imputer, scaler, PCA, and angle-scaling state for one dataset/seed. |
| `nonprivate/DATASET/qnn_seedN.pt` | Non-private exact-QNN checkpoint. |
| `nonprivate/DATASET/mlp_matched_seedN.pt` | Non-private capacity-matched MLP checkpoint. |
| `nonprivate/DATASET/logistic_torch_seedN.pt` | Non-private Torch logistic checkpoint. |
| `private/DATASET/MODEL/epsE_seedN.pt` | One private checkpoint identified by model, target epsilon, and seed. |
| `private_ablation/DATASET/featuresF_layersL_ENCODING_seedN.pt` | One epsilon-1 DP-QNN ablation checkpoint. |

## `Results/IBM_QPU/` and `Results/Figures/`

Every IBM record, figure, and numeric figure-data file is described in
`RESULTS_GUIDE.md`.

## `Documentation/`

| File | Description |
|---|---|
| `PROJECT_DESCRIPTION.md` | Motivation, objectives, research questions, datasets, models, experiments, outputs, and intended use. |
| `METHODOLOGY.md` | Detailed data, preprocessing, model, DP, randomness, ablation, statistical, IBM, and figure methods. |
| `ARCHITECTURE.md` | Code boundaries and data/privacy/result flows. |
| `PARAMETERS.md` | Every configuration key, CLI option, IBM constant, and figure option. |
| `RESULTS_GUIDE.md` | Every result file, key columns, artifact family, and interpretation. |
| `FILE_REFERENCE.md` | Human-readable file/pattern catalog. |
| `FILE_MANIFEST.csv` | Per-file path, category, description, size, and SHA-256. |
| `REPRODUCIBILITY.md` | Environment setup and validation workflows. |
