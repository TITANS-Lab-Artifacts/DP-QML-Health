# Code and data architecture

## Separation of responsibilities

The project has four layers:

1. `Data/Simulation/` stores immutable/downloaded raw data, prepared clean CSVs,
   and the source manifest.
2. `Code/Simulation/` reads those inputs, trains models, and writes only to
   `Results/Simulation/`.
3. `Code/IBM_QPU/` performs exact validation and optional hardware inference,
   writing only to `Results/IBM_QPU/`.
4. `Code/FigureGeneration/` reads included tables/JSON and writes only to
   `Results/Figures/`. It has no training dependency.

Canonical paths are defined once in `qpriv_utils.py` for simulation and once in
`io_utils.py` for figure generation. Paths are based on the files' resolved
locations, so commands work from the project root or another working directory.

## Simulation data flow

`prepare_data.py` verifies clinical downloads by SHA-256, preserves missing
values for later public-only fitting, optionally derives WESAD window features,
and writes a manifest. `qpriv_utils.split_preprocess` creates private train,
public calibration, and public test partitions. Median imputation,
standardization, PCA, and angle scaling are fitted only on the public portion.

`run_nonprivate.py` chooses learning rates on public data and evaluates five
baseline types for every dataset/seed. `run_private.py` chooses one public
learning rate per model family and dataset/seed, then executes the declared
model/epsilon grid. `run_ablation.py` executes the feature/layer/encoding grid
at epsilon 1. `analyze_results.py` produces summaries, privacy-accounting
extracts, and corrected paired tests.

Independent jobs run in spawned processes. Workers write uniquely named model
artifacts; the parent process alone updates shared CSV/JSON checkpoints. JSON
writes are atomic, strict (`NaN` is converted to `null`), and mirrored to CSV.

## Privacy and randomness flow

The private trainer uses unit-level Poisson sampling matched to the RDP
accountant. A clinical privacy unit is one row/patient. A WESAD privacy unit is
one subject and all of that subject's windows. Per-unit gradients are clipped,
Gaussian noise is added, and realized epsilon is computed from sampling rate,
steps, noise multiplier, and delta.

Stable derived seeds separate model initialization, Poisson sampling, and
Gaussian noise. This supports academic reproducibility; a production deployment
must use secret randomness and appropriate protected-data controls.

## Result/figure flow

The four `plot_*.py` programs explicitly load only the table(s) needed for one
figure. `io_utils.py` fails early with a clear message if a file or required
column/key is missing. `figure_utils.py` provides common visual style and the
four plotting functions. Generated outputs are deterministic with respect to
the included data and installed plotting stack.

No diagram, background illustration, system-overview, quantum-circuit image,
or Draw.io generator is included.
