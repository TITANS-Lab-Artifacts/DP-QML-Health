# Simulation code

This folder contains only executable simulation code, configuration, and its
dependency list. Input data are under `Data/Simulation/`; all tables and model
artifacts are under `Results/Simulation/`.

## Files

| File | Purpose |
|---|---|
| `config.json` | Complete scientific and execution configuration. |
| `prepare_data.py` | Downloads verified clinical files, prepares optional WESAD features, and records provenance. |
| `qpriv_utils.py` | Shared I/O, preprocessing, QNN/classical models, training, metrics, DP accounting, and canonical paths. |
| `run_nonprivate.py` | Public learning-rate selection and non-private baselines. |
| `run_private.py` | DP training over every model, epsilon, dataset, and seed. |
| `run_ablation.py` | DP-QNN feature/layer/encoding grid. |
| `analyze_results.py` | Summaries, paired bootstrap intervals, Wilcoxon tests, and Holm correction. |
| `run_all.py` | Ordered end-to-end runner with progress and adaptive ETA. |
| `requirements.txt` | Package versions that produced the included results (Python 3.9 recommended). |

Run `python Code/Simulation/run_all.py --demo` from the project root for a
small smoke workflow. See the root README and `Documentation/PARAMETERS.md` for
the complete run and every configurable parameter.
