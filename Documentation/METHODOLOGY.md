# Methodology

## Experiment sequence

The complete workflow performs five ordered stages:

1. prepare and verify data;
2. train non-private models;
3. train differentially private models;
4. run the private QNN architecture ablation;
5. compute descriptive summaries and corrected statistical tests.

`Code/Simulation/run_all.py` executes the stages and reports overall progress.
The IBM QPU experiment and figure generation are separate workflows because
they do not participate in model training.

## Data acquisition and verification

The preparation program downloads the three clinical sources only when a valid
local raw file is unavailable. Each download is checked against its declared
SHA-256 digest before use. The data source URL and digest are stored in
`Data/Simulation/dataset_manifest.json`.

WESAD is user supplied because its distribution process differs from the
clinical CSV sources. Subject pickle digests are recorded when raw WESAD
features are rebuilt. Pickle loading is allowed only through an explicit trust
control.

## Cleaning and missing data

Raw cleaning standardizes columns and creates binary targets. Missing-value
imputation is deliberately excluded from raw cleaning. This prevents private
training or public-test records from influencing imputation statistics.

For each experiment seed, the public calibration partition fits:

1. median imputation;
2. standardization;
3. PCA to the configured feature count;
4. minimum-maximum scaling into rotation-angle range.

The fitted transform is then applied unchanged to private train and public test
partitions. Fitted preprocessors are stored as joblib artifacts.

## Partitioning and privacy units

The default split reserves 20% for public calibration and 20% for public test,
leaving 60% for private training. All three partitions are drawn from the
experiment seed, so each seed defines shared splits for paired model
comparisons. Clinical data use stratified record-level splits. WESAD uses
subject-level grouping (9/3/3 of 15 participants) so a subject's windows cannot
cross partitions.

The same privacy-unit definition is used by splitting, Poisson sampling,
gradient aggregation, clipping, accounting, and reported metadata.

## Quantum neural network

Let `x` be the public-preprocessed feature vector. Each feature is encoded as a
single-qubit rotation. Every variational layer applies one trainable rotation
per qubit followed by ring CNOT connections. The output is the expectation of
Pauli Z on the first logical qubit, converted to a class-1 probability.

The default configuration has four inputs/qubits and two layers, producing
eight trainable QNN parameters. `lightning.qubit` evaluates the exact state and
the adjoint method computes exact gradients efficiently.

The loss is mean squared error between predicted probabilities and binary
labels. Public test evaluation reports ROC AUC, precision-recall AUC, accuracy
at probability 0.5, and Brier score.

## Classical baselines

The matched MLP width is the smallest hidden width whose parameter count is at
least the QNN's (two tanh hidden units and 13 parameters for four inputs and
the eight-parameter QNN). The Torch logistic model (the paper's "LR") has one
linear sigmoid output and 5 parameters.
Both use the same public learning-rate selection, train/test partitions, loss,
and metrics as the QNN.

Non-private scikit-learn logistic regression and random forest provide familiar
additional reference models. These models do not produce Torch checkpoints.

## Public hyperparameter selection

Learning rates are selected exclusively on public calibration data. Candidate
sets are declared in `config.json` for QNN, MLP, and logistic families. For
private training, one public learning rate is selected per dataset, seed, and
family, then reused across that family's clipping and epsilon configurations.

This avoids selecting hyperparameters against private training behavior or
public-test metrics.

## Non-private training

The full non-private workflow runs every configured dataset and seed. Torch
models use 25 epochs and three deterministic restarts, selecting the best fit
without using the public test set. Checkpoints and test metrics are saved after
each complete dataset/seed task.

## Differentially private training

Private training uses independent unit-level Poisson inclusion at every step.
With `n` private units, the program uses `ceil(n / 32)` steps per epoch and
sampling rate `q = 1 / ceil(n / 32)`, so the expected batch `nq` is at most 32
units (for example 30.7 for Pima and all 9 participants for WESAD). Per-unit
gradients are computed, their norms are clipped at threshold `C`, seeded
Gaussian noise with standard deviation `σC` is added to the sum, and the result
is divided by the expected batch size `nq`.

The noise multiplier is chosen to meet target epsilon for the configured delta,
sampling probability, and total number of steps. Opacus RDP accounting computes
the realized epsilon. Every private result stores target and realized epsilon,
delta, noise multiplier, clipping threshold, sample rate, steps, mean sampled
units, empty-step count, and privacy-unit count.

For the MSE loss, each coordinate of a unit's QNN gradient is bounded by one,
so its L2 norm is at most `√P` (`√8` for the primary QNN). QNN configurations
below this analytic gradient certificate are labelled empirical. The certificate-threshold QNN configuration is labelled certified
no-clipping.

## Randomness protocol

Each experiment has separate deterministic streams for model initialization,
Poisson sampling, and Gaussian noise. Derived seeds are computed from stable
experiment identifiers. Spawned workers are limited to one numerical-library
thread so multi-process execution does not oversubscribe the machine.

This protocol supports academic repeatability. A deployment privacy mechanism
would require secret cryptographically appropriate randomness rather than
published deterministic seeds.

## Checkpoint and resume behavior

Workers write only uniquely named model artifacts. The parent process writes
shared result JSON and CSV files. JSON writes use an atomic temporary-file
replacement, and non-finite numeric values become strict JSON `null`.

Private and ablation workflows load existing result identities, skip completed
configurations, and rerun only missing work. Incomplete non-private
dataset/seed groups are discarded and recomputed as deterministic units.

## Architecture ablation

At target epsilon 1 and `C = 1`, the QNN grid varies PCA components, variational layers,
and RX/RY rotation encoding. Public learning-rate selection is repeated for the
active architecture. When the requested PCA count exceeds available raw
features, the program writes an explicit `ablation_skipped` record.

## Statistical procedure

Private results are paired by independent seed. For the predeclared primary QNN
and MLP comparison, the analysis computes:

- mean seed-paired ROC-AUC difference;
- paired bootstrap confidence interval;
- Wilcoxon signed-rank p-value;
- Holm multiple-comparison adjustment.

Curve-level comparisons first average each model over epsilon for each seed,
then compare the paired seed averages. Holm correction is applied jointly to
all 24 tests (20 per-budget and 4 curve-average). Bootstrap intervals use
10,000 paired resamples. The analysis table records the number of
independent seed pairs.

## IBM execution method

The IBM script builds the same four-qubit, two-layer RY circuit used by the
primary QNN. It embeds fixed seed-0 Pima inputs and trained checkpoint weights.
The logical circuit is checked with `StatevectorEstimator` before any hardware
submission.

For hardware execution, the logical circuit is transpiled at optimization level
3 with seed 2026. Private and non-private bindings are executed together with
1,024 shots each, resilience level 0, gate twirling off, measurement twirling
off, and a 540-second QPU execution cap. The job ID is saved immediately before
waiting for completion.

## Figure methodology

Each result figure loads a fixed set of result files and performs only grouping,
summary, and plotting operations. It does not import simulation code or retrain
models. Input validators check required files, columns, and top-level JSON keys.

Figures use a publication-width serif style, vector PDF output, and 400-DPI PNG
output. Numeric grouped data are also saved as CSV/JSON so plotted values can be
inspected without reading pixels from a figure.

## Interpretation boundaries

The privacy guarantee is conditional on public calibration/test data being
truly public and on the specified unit of adjacency. Results are empirical for
the included datasets and grids. The IBM experiment measures execution
agreement on fixed cases; it does not establish hardware training performance,
clinical generalization, or quantum advantage.
