# Project description

## Overview

This project accompanies the paper "Unit-Aware Differentially Private Quantum Machine Learning for Clinical and Wearable Health Data." It studies whether a quantum neural
network (QNN) retains more predictive utility than classical models when all
are trained with unit-aware differentially private stochastic gradient descent
(DP-SGD) on clinical and wearable health data. The project implements a
controlled binary-classification workflow in which quantum and classical models
use the same input dimensionality, public/private data partitions, public-only
model selection, privacy budgets, evaluation metrics, and random seeds.

The central model is a variational quantum neural network evaluated with exact
simulation. Differential privacy is applied during training through unit-level
Poisson sampling, gradient clipping, Gaussian noise, and Rényi differential
privacy accounting. An IBM QPU experiment evaluates fixed trained checkpoints
on a small, fixed set of public Pima cases and measures agreement with exact
statevector inference.

## Motivation

Health datasets contain sensitive measurements and labels. Machine-learning
experiments may therefore require formal privacy protections in addition to
ordinary access controls. Differentially private stochastic gradient training
limits the influence of an individual privacy unit, while quantum neural
networks offer a distinct parameterized model family whose behavior under
privacy noise remains an active research question.

This project brings these topics into one reproducible workflow. It emphasizes
fair model comparison, explicit privacy units, public-only preprocessing,
realized privacy accounting, complete result provenance, and clear limits on
what a small real-hardware experiment can establish.

## Research questions

The project follows the paper's three research questions:

- **RQ1:** Does a capacity-matched QNN retain more utility than classical
  baselines under record-level and participant-level DP? This is addressed by
  privacy-utility curves for the QNN, matched MLP, and logistic regression
  across five target epsilon values, and by seed-paired QNN-versus-MLP tests
  with Holm correction.
- **RQ2:** How do the clipping threshold, encoding gate, parameter count, and
  circuit depth affect predictive performance under DP? This is addressed by
  comparing empirical clipping thresholds with the analytic gradient
  certificate and by the epsilon-1 circuit ablation.
- **RQ3:** Do predictions from NISQ hardware agree with exact statevector
  inference for the same trained circuits? This is addressed by one IBM QPU
  job on `ibm_fez`.

## Datasets

### Pima Diabetes

Pima is a patient-record binary diabetes dataset. The project preserves
domain-coded missing values during raw cleaning and fits imputation later using
only public calibration data. One row is one privacy unit.

### UCI Heart

The processed Cleveland Heart dataset is used for binary heart-disease
classification. Source values are cleaned and mapped to a binary target. One
patient row is one privacy unit.

### Heart Failure

The Heart Failure Clinical Records dataset is used for mortality
classification. One patient row is one privacy unit.

### WESAD

WESAD provides wearable physiological signals. The preparation stage derives
seven features (BVP mean/std, EDA mean/std, temperature mean, acceleration
magnitude mean/std) from one-minute wrist-sensor windows that are more than 90%
baseline (label 0) or more than 90% stress (label 1), while retaining subject
identity. This gives 438 windows from 15 participants. All windows from one
subject are grouped together, and one subject—not one window—is the privacy
unit. Subject-level splitting, sampling, and clipping prevent a subject from
being treated as many independent people.

## Data partitions and public information

Each experiment separates data into private training, public calibration, and
public test partitions. The default proportions are 60%, 20%, and 20%. Public
calibration data fit the median imputer, standardizer, PCA transform,
feature-angle scaler, and learning-rate selection. Test data are used only for
final metrics. The privacy claim is conditional on calibration and test data
being legitimately public.

## Model families

### Quantum neural network

The primary QNN uses four PCA features mapped to four qubits. Each input
feature is encoded as an RY rotation angle in `[0, π]`. Each of the two
variational layers applies four trainable RY rotations followed by a ring of
CNOT gates (0→1→2→3→0), giving eight trainable parameters. The Pauli-Z
expectation on the first qubit is converted to the positive-class probability
`(1 + ⟨Z0⟩)/2`. Exact simulation uses PennyLane `lightning.qubit` and adjoint
gradients.

### Capacity-matched MLP

Capacity matching means that every model receives the same four PCA
components. The MLP hidden width is the smallest width whose trainable
parameter count is at least the QNN's. With four inputs this gives two tanh
hidden units and one sigmoid output (13 parameters). This produces a more
informative classical comparison than an arbitrarily large neural network.

### Logistic models and random forest

The project includes a differentiable PyTorch logistic regression model
(5 parameters) for both private and non-private training; this is the "LR"
model in the paper. Non-private experiments also include scikit-learn logistic
regression and random forest references, which are not reported in the
paper.

## Differential privacy experiment

Private training evaluates target epsilon values 0.5, 1, 2, 4, and 8 at delta
1e-5. Per-unit gradients are clipped to threshold `C`; calibrated Gaussian
noise is then added. The sampler, number of steps, sampling probability, noise
multiplier, realized epsilon, and privacy unit are saved with every fit.

Three QNN thresholds distinguish empirical clipping at 0.5 and 1.0 from the
analytic QNN certificate `C = √8 ≈ 2.83`, at or above which clipping is
inactive. Matched MLPs use thresholds 0.5 and 1.0; private
logistic regression uses threshold 1.0.

## Ablation experiment

The private QNN ablation uses target epsilon 1 and clipping threshold `C = 1`,
and crosses:

- PCA components/qubits: 4, 6, and 8;
- variational layers: 1, 2, and 3;
- encodings: RY and RX;
- five independent seeds;
- every available dataset.

This gives 18 circuit settings × 4 datasets × 5 seeds = 360 requests, of which
330 run and 30 (WESAD with eight components) are skipped. If a dataset cannot support the requested PCA dimension, the configuration is
saved as an explicit skip record with a reason. It is not silently dropped or
counted as a failed fit.

## Statistical analysis

The primary comparison pairs QNN `qnn_C1.0_empirical` with matched MLP
`mlp_C1.0` by dataset, epsilon, and seed. Analysis outputs include mean paired
difference, paired bootstrap confidence interval, Wilcoxon signed-rank p-value,
and Holm-adjusted p-value. Descriptive summaries report mean, standard
deviation, and count.

There are 24 tests in total: 20 per-budget tests and 4 curve-average tests.
Evaluation metrics are ROC AUC (primary), precision-recall AUC, accuracy at
threshold 0.5, and Brier score.

## IBM QPU experiment

The hardware experiment embeds trained weights and eight public Pima inputs.
The first four selected cases have label 0 and the next four label 1. Private
and non-private checkpoints are submitted together as 16 parameter bindings,
using the same backend, transpilation process, observable, shots, and Runtime
job.

Exact statevector probabilities are computed first. Hardware outputs are then
summarized with descriptive classification metrics and agreement measures such
as mean absolute error, root mean squared error, maximum absolute error, and
signed error.

This stage tests inference feasibility and exact-to-hardware agreement. It does
not train on the QPU and does not provide an independent clinical evaluation.

## Project outputs

The project includes complete CSV and JSON experiment records, fitted public
preprocessors, model checkpoints, privacy-accounting tables, statistical-test
tables, IBM execution records, four quantitative figures, and numeric data
supporting those figures.

Every training artifact has a project-relative reference in its result table.
Every result family is documented in `RESULTS_GUIDE.md`, and every delivered
path is listed in `FILE_MANIFEST.csv`.

## Intended use

This project is intended for research reproduction, privacy-aware quantum
learning experiments, method inspection, result analysis, and quantitative
figure generation. It is not intended for direct patient-care decisions,
production handling of protected health information, or unsupported claims of
quantum performance advantage.
