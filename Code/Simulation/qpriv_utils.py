"""Shared, privacy-critical utilities for Unit-Aware Differentially Private
Quantum Machine Learning for Clinical and Wearable Health Data.

Version 4.0 uses the corrected privacy protocol, reproducible academic
randomness, and an exact Lightning/adjoint simulator:

* preprocessing is fitted only on a designated public calibration partition;
* private training uses independent Poisson sampling, matching the RDP accountant;
* WESAD is sampled and clipped by subject, not by 60-second window;
* model initialization, sampling, and DP-noise randomness are separated;
* sampling and Gaussian noise use separate, explicitly seeded Torch generators;
* private training never returns raw private-gradient or membership diagnostics;
* clipping thresholds below the analytic QNN bound are explicitly empirical.
* independent configurations use spawned processes while shared result files
  are written only by the parent process.

The formal guarantee is conditional on the public calibration/test partitions and
all preprocessing fitted from them being genuinely public. If every available
record is private, replace this preprocessing with a separately accounted DP
preprocessor before claiming end-to-end privacy.
"""

# The targeted warning filter must run before third-party imports below.
# ruff: noqa: E402

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

# Some environments contain the same optional NetworkX test backend twice.
# NetworkX reports that harmless installation issue during PennyLane import.
# Suppress only that exact warning; all other runtime warnings remain visible.
warnings.filterwarnings(
    "ignore",
    message=r"networkx backend defined more than once: nx-loopback",
    category=RuntimeWarning,
    module=r"networkx\.utils\.backends",
)

# Each spawned experiment worker gets one numerical-library thread. This avoids
# severe oversubscription when six to eight workers run simultaneously.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


QPRIV_VERSION = "4.0"
RANDOMNESS_PROTOCOL_VERSION = "3.1"
RDP_ALPHAS = [
    *[1.0 + index / 10 for index in range(1, 100)],
    *range(12, 64),
    64,
    80,
    96,
    128,
    160,
    192,
    256,
    384,
    512,
]

# Keep code, input data, and generated results separate.  All simulation
# scripts import these paths, so changing the project layout only requires a
# change here.
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "Data" / "Simulation"
RESULTS_DIR = PROJECT_ROOT / "Results" / "Simulation" / "Tables"
ARTIFACT_DIR = PROJECT_ROOT / "Results" / "Simulation" / "Artifacts"
for _directory in (DATA_DIR, RESULTS_DIR, ARTIFACT_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Atomic I/O and provenance
# ---------------------------------------------------------------------------
def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _json_safe(value):
    """Convert NumPy values and non-finite floats to strict JSON values."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def project_relative(path) -> str:
    """Return a portable path relative to the project root."""

    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        # External inputs, such as a user-supplied WESAD directory, cannot be
        # expressed relative to this project and therefore remain absolute.
        return str(resolved)


def save_json(obj, path) -> None:
    """Atomically save JSON so interrupted sweeps do not corrupt checkpoints."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(
                _json_safe(obj),
                handle,
                indent=2,
                default=_json_default,
                allow_nan=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    print(f"saved -> {path}")


def save_csv(df: pd.DataFrame, path) -> None:
    """Atomically save a DataFrame as CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".csv",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            df.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    print(f"saved -> {path}")


def load_json(path, default=None):
    """Load JSON, returning ``default`` when the file does not exist."""

    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_records(
    records: list[dict], json_path, sort_by: Sequence[str]
) -> pd.DataFrame:
    """Save deterministic record checkpoints in both JSON and CSV formats."""

    frame = pd.DataFrame(records)
    if frame.empty and len(frame.columns) == 0:
        frame = pd.DataFrame(columns=sort_by)
    available_sort_columns = [column for column in sort_by if column in frame.columns]
    if available_sort_columns and not frame.empty:
        frame = frame.sort_values(available_sort_columns, kind="stable").reset_index(
            drop=True
        )
    normalized = frame.to_dict(orient="records")
    save_json(normalized, json_path)
    save_csv(frame, Path(json_path).with_suffix(".csv"))
    return frame


def save_preprocessor(preprocessor: "PublicPreprocessor", path) -> None:
    """Persist a fitted public preprocessor.

    Joblib artifacts can execute code when loaded. Only load artifacts created by
    this project in a trusted environment.
    """

    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    joblib.dump(preprocessor, temp_path)
    os.replace(temp_path, path)
    print(f"saved -> {path}")


def load_preprocessor(path) -> "PublicPreprocessor":
    """Load a trusted preprocessor artifact created by :func:`save_preprocessor`."""

    import joblib

    return joblib.load(path)


def save_model_state(model: nn.Module, path, metadata: dict) -> None:
    """Atomically save a model state and reconstruction metadata.

    PyTorch checkpoints use pickle internally. Only load checkpoints created by
    this project in a trusted environment.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "metadata": metadata,
            },
            temp_path,
        )
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    print(f"saved -> {path}")


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_fingerprint() -> dict:
    """Return the runtime versions stored beside each result set."""

    import opacus
    import pennylane_lightning
    import sklearn

    return {
        "qpriv_version": QPRIV_VERSION,
        "randomness_protocol_version": RANDOMNESS_PROTOCOL_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "pennylane": qml.__version__,
        "pennylane_lightning": getattr(
            pennylane_lightning,
            "__version__",
            "unknown",
        ),
        "opacus": opacus.__version__,
        "scikit_learn": sklearn.__version__,
        "parallel_worker_policy": "spawned_processes_one_torch_thread_each",
    }


def set_experiment_seed(seed: int) -> None:
    """Seed reproducible academic experiment randomness."""

    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_worker() -> None:
    """Limit one spawned worker to one Torch thread."""

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting inter-op threads only once per process.
        pass


def worker_count(requested: Optional[int], task_count: int) -> int:
    """Return a safe worker count, defaulting to CPU count minus two, capped at 8."""

    if task_count < 1:
        return 1
    available = os.cpu_count() or 1
    automatic = min(8, max(1, available - 2))
    selected = automatic if requested is None else min(requested, 8)
    if selected < 1:
        raise ValueError("workers must be at least 1")
    return min(selected, task_count)


# ---------------------------------------------------------------------------
# Public preprocessing and privacy-unit-aware data splits
# ---------------------------------------------------------------------------
@dataclass
class PublicPreprocessor:
    """Transforms fitted exclusively on a designated public calibration set."""

    feature_columns: list[str]
    imputer: SimpleImputer
    standardizer: StandardScaler
    pca: PCA
    angle_scaler: MinMaxScaler
    n_output_features: int
    public_seed: int

    def transform_frame(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, self.feature_columns].to_numpy(dtype=float)
        values = self.imputer.transform(values)
        values = self.standardizer.transform(values)
        values = self.pca.transform(values)
        values = self.angle_scaler.transform(values)
        return np.clip(values, 0.0, np.pi)


@dataclass
class PreparedData:
    """Public calibration, private training, and public test partitions."""

    X_train: torch.Tensor
    y_train: torch.Tensor
    unit_ids_train: np.ndarray
    X_public: torch.Tensor
    y_public: torch.Tensor
    unit_ids_public: np.ndarray
    X_test: torch.Tensor
    y_test: torch.Tensor
    unit_ids_test: np.ndarray
    preprocessor: PublicPreprocessor
    split_info: dict

    @property
    def n_private_units(self) -> int:
        return len(pd.unique(self.unit_ids_train))


def load_clean(name: str) -> pd.DataFrame:
    """Load a raw-cleaned dataset produced by ``prepare_data.py``."""

    return pd.read_csv(DATA_DIR / f"{name}_clean.csv")


def _validate_partition_labels(y: np.ndarray, name: str) -> None:
    if len(np.unique(y)) < 2:
        raise ValueError(
            f"{name} partition has only one class. Choose another split seed or "
            "increase the partition size."
        )


def _subject_partitions(
    subjects: np.ndarray,
    public_size: float,
    test_size: float,
    seed: int,
) -> tuple[set, set, set]:
    if len(subjects) < 3:
        raise ValueError("subject-level splitting needs at least three subjects")
    shuffled = np.asarray(sorted(subjects), dtype=object)
    np.random.RandomState(seed).shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_size)))
    n_public = max(1, int(round(len(shuffled) * public_size)))
    if n_test + n_public >= len(shuffled):
        n_public = 1
        n_test = 1
    test_subjects = set(shuffled[:n_test])
    public_subjects = set(shuffled[n_test : n_test + n_public])
    train_subjects = set(shuffled[n_test + n_public :])
    return train_subjects, public_subjects, test_subjects


def split_preprocess(
    df: pd.DataFrame,
    n_features: int = 4,
    split_seed: int = 0,
    public_seed: int = 2026,
    public_size: float = 0.2,
    test_size: float = 0.2,
) -> PreparedData:
    """Create disjoint public-calibration, private-train, and public-test data.

    All fitted transforms (median imputation, standardization, PCA, and angle
    scaling) see only the designated public calibration partition. The privacy
    guarantee for training is conditional on this partition and the test set
    being public and fixed before private training.

    For WESAD, partitions and privacy units are subjects. For clinical datasets,
    partitions and privacy units are rows/patients.
    """

    if not 0 < public_size < 1 or not 0 < test_size < 1:
        raise ValueError("public_size and test_size must be in (0, 1)")
    if public_size + test_size >= 1:
        raise ValueError("public_size + test_size must be below 1")
    if "target" not in df.columns:
        raise ValueError("dataframe must contain a target column")

    y = df["target"].to_numpy(dtype=float)
    indices = np.arange(len(df))
    subject_level = "subject" in df.columns

    if subject_level:
        train_subjects, public_subjects, test_subjects = _subject_partitions(
            df["subject"].unique(), public_size, test_size, split_seed
        )
        subject_values = df["subject"].to_numpy()
        idx_train = indices[np.isin(subject_values, list(train_subjects))]
        idx_public = indices[np.isin(subject_values, list(public_subjects))]
        idx_test = indices[np.isin(subject_values, list(test_subjects))]
        unit_ids = subject_values.astype(str)
        privacy_unit = "subject"
    else:
        idx_pool, idx_test = train_test_split(
            indices,
            test_size=test_size,
            stratify=y,
            random_state=split_seed,
        )
        relative_public_size = public_size / (1.0 - test_size)
        idx_train, idx_public = train_test_split(
            idx_pool,
            test_size=relative_public_size,
            stratify=y[idx_pool],
            random_state=split_seed + 10_000,
        )
        unit_ids = np.asarray([f"row-{i}" for i in indices], dtype=object)
        privacy_unit = "row"

    for part_name, part_indices in (
        ("private train", idx_train),
        ("public calibration", idx_public),
        ("public test", idx_test),
    ):
        if len(part_indices) == 0:
            raise ValueError(f"{part_name} partition is empty")
        _validate_partition_labels(y[part_indices], part_name)

    feature_columns = [
        column for column in df.columns if column not in {"target", "subject"}
    ]
    public_values = df.iloc[idx_public][feature_columns].to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median").fit(public_values)
    public_values = imputer.transform(public_values)
    standardizer = StandardScaler().fit(public_values)
    public_values = standardizer.transform(public_values)
    k = min(n_features, public_values.shape[1], public_values.shape[0])
    if k < 1:
        raise ValueError("no usable features remain after preprocessing")
    pca = PCA(n_components=k, random_state=public_seed).fit(public_values)
    public_values = pca.transform(public_values)
    angle_scaler = MinMaxScaler((0.0, np.pi)).fit(public_values)
    preprocessor = PublicPreprocessor(
        feature_columns=feature_columns,
        imputer=imputer,
        standardizer=standardizer,
        pca=pca,
        angle_scaler=angle_scaler,
        n_output_features=k,
        public_seed=public_seed,
    )

    def tensors(part_indices):
        X = preprocessor.transform_frame(df.iloc[part_indices])
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y[part_indices], dtype=torch.float32),
            unit_ids[part_indices].copy(),
        )

    X_train, y_train, units_train = tensors(idx_train)
    X_public, y_public, units_public = tensors(idx_public)
    X_test, y_test, units_test = tensors(idx_test)

    split_info = {
        "split_seed": split_seed,
        "public_seed": public_seed,
        "privacy_unit": privacy_unit,
        "private_rows": len(idx_train),
        "private_units": len(pd.unique(units_train)),
        "public_rows": len(idx_public),
        "public_units": len(pd.unique(units_public)),
        "test_rows": len(idx_test),
        "test_units": len(pd.unique(units_test)),
        "n_features": k,
        "preprocessing_fit": "public_calibration_only",
    }
    return PreparedData(
        X_train=X_train,
        y_train=y_train,
        unit_ids_train=units_train,
        X_public=X_public,
        y_public=y_public,
        unit_ids_public=units_public,
        X_test=X_test,
        y_test=y_test,
        unit_ids_test=units_test,
        preprocessor=preprocessor,
        split_info=split_info,
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class QNN(nn.Module):
    """Angle-encoded QNN with an explicitly requested differentiation method."""

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        encoding: str = "ry",
        diff_method: str = "adjoint",
        device_name: str = "lightning.qubit",
    ):
        super().__init__()
        if encoding not in {"ry", "rx"}:
            raise ValueError("encoding must be 'ry' or 'rx'")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding = encoding
        self.diff_method = diff_method
        self.device_name = device_name
        device = qml.device(device_name, wires=n_qubits)

        @qml.qnode(device, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            for wire in range(n_qubits):
                if encoding == "ry":
                    qml.RY(inputs[wire], wires=wire)
                else:
                    qml.RX(inputs[wire], wires=wire)
            for layer in range(n_layers):
                for wire in range(n_qubits):
                    qml.RY(weights[layer, wire], wires=wire)
                for wire in range(n_qubits):
                    qml.CNOT(wires=[wire, (wire + 1) % n_qubits])
            return qml.expval(qml.PauliZ(0))

        self.q = qml.qnn.TorchLayer(
            circuit,
            {"weights": (n_layers, n_qubits)},
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = torch.stack([self.q(row) for row in x])
        return (outputs + 1.0) / 2.0


class MLP(nn.Module):
    """Small one-hidden-layer classical comparator."""

    def __init__(self, n_in: int = 4, hidden: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LogisticModel(nn.Module):
    """Strong linear-sigmoid baseline compatible with the manual DP loop."""

    def __init__(self, n_in: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def matched_mlp_width(n_in: int, target_params: int) -> int:
    """Smallest MLP width whose parameter count is at least ``target_params``."""

    for hidden in range(1, 128):
        if count_params(MLP(n_in, hidden)) >= target_params:
            return hidden
    raise ValueError("target parameter count is unexpectedly large")


def build_matched_mlp(n_in: int, target_params: int) -> MLP:
    """Build an approximately size-matched MLP and report both counts in results."""

    return MLP(n_in, matched_mlp_width(n_in, target_params))


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def mse_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((pred - y) ** 2).mean()


def train_nonprivate(
    model_factory: Callable[[], nn.Module],
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 40,
    lr: float = 0.05,
    batch: int = 32,
    seed: int = 0,
    n_restarts: int = 3,
) -> nn.Module:
    """Train fresh, deterministically initialized restarts."""

    best_model, best_loss = None, float("inf")
    n_rows = len(X)
    for restart in range(max(1, n_restarts)):
        set_experiment_seed(seed * 10_000 + restart)
        model = model_factory()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            permutation = torch.randperm(n_rows)
            for start in range(0, n_rows, batch):
                idx = permutation[start : start + batch]
                optimizer.zero_grad()
                loss = mse_loss(model(X[idx]), y[idx])
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            final_loss = mse_loss(model(X), y).item()
        if final_loss < best_loss:
            best_model, best_loss = model, final_loss
    return best_model


def tune_lr_on_public_data(
    model_factory: Callable[[], nn.Module],
    X_public: torch.Tensor,
    y_public: torch.Tensor,
    candidates: Sequence[float],
    seed: int,
    epochs: int = 20,
) -> float:
    """Select a model-family learning rate using public data only."""

    indices = np.arange(len(X_public))
    idx_fit, idx_val = train_test_split(
        indices,
        test_size=0.3,
        stratify=y_public.numpy(),
        random_state=seed,
    )
    scored = []
    for lr in candidates:
        model = train_nonprivate(
            model_factory,
            X_public[idx_fit],
            y_public[idx_fit],
            epochs=epochs,
            lr=lr,
            seed=seed,
            n_restarts=1,
        )
        with torch.no_grad():
            score = mse_loss(model(X_public[idx_val]), y_public[idx_val]).item()
        scored.append((score, lr))
    return min(scored)[1]


def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        probabilities = model(X).detach().cpu().numpy()
    labels = y.detach().cpu().numpy()
    return {
        "auc": roc_auc_score(labels, probabilities),
        "prauc": average_precision_score(labels, probabilities),
        "acc": accuracy_score(labels, (probabilities >= 0.5).astype(int)),
        "brier": brier_score_loss(labels, probabilities),
    }


# ---------------------------------------------------------------------------
# Differential privacy
# ---------------------------------------------------------------------------
def qnn_gradient_certificate(model: nn.Module) -> float:
    """Return the proven QNN MSE-gradient bound ``sqrt(p)``.

    This is a no-clipping certificate only for thresholds C greater than or
    equal to the returned value. Smaller thresholds are empirical choices.
    """

    if not isinstance(model, QNN):
        raise TypeError("the sqrt(p) certificate is specific to QNN")
    return math.sqrt(count_params(model))


def get_sigma(target_eps: float, delta: float, sample_rate: float, steps: int) -> float:
    from opacus.accountants.utils import get_noise_multiplier

    return get_noise_multiplier(
        target_epsilon=target_eps,
        target_delta=delta,
        sample_rate=sample_rate,
        steps=steps,
        accountant="rdp",
        alphas=RDP_ALPHAS,
    )


def _unit_row_groups(
    unit_ids: Optional[Iterable], n_rows: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    if unit_ids is None:
        ids = np.asarray([f"row-{i}" for i in range(n_rows)], dtype=object)
    else:
        ids = np.asarray(list(unit_ids), dtype=object)
        if len(ids) != n_rows:
            raise ValueError("unit_ids must have one entry per row")
    unique_units = pd.unique(ids)
    groups = [np.flatnonzero(ids == unit) for unit in unique_units]
    return unique_units, groups


def _make_seeded_generator(seed: Optional[int], purpose: str):
    if seed is None:
        raise ValueError(f"{purpose} requires an explicit seed")
    return torch.Generator(device="cpu").manual_seed(int(seed))


def _generate_gaussian_noise(
    std: float,
    reference: torch.Tensor,
    generator,
) -> torch.Tensor:
    """Generate reproducible Gaussian noise for academic experiments."""

    if std == 0:
        return torch.zeros_like(reference)
    return torch.normal(
        mean=0.0,
        std=std,
        size=reference.shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )


def train_dp_manual(
    model_factory: Callable[[], nn.Module],
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    unit_ids: Optional[Iterable] = None,
    target_eps: float = 1.0,
    delta: float = 1e-5,
    C: float = 1.0,
    epochs: int = 25,
    lr: float = 0.08,
    expected_batch_units: int = 32,
    init_seed: int = 0,
    sampling_seed: Optional[int] = None,
    noise_seed: Optional[int] = None,
) -> tuple[nn.Module, dict]:
    """Train with unit-level Poisson-sampled DP-SGD.

    Each privacy unit is independently included at every step. For WESAD a unit
    is one subject; its gradient is the gradient of its mean window loss and is
    clipped once. The averaging denominator is the expected number of sampled
    units, matching Opacus's Poisson DP-SGD convention.

    Sampling and noise have separate explicit seeds so academic experiments can
    be reproduced. This implementation is intended for simulations on public
    benchmark data, not deployment over genuinely sensitive records.
    """

    from opacus.accountants import RDPAccountant

    if target_eps <= 0 or not 0 < delta < 1 or C <= 0:
        raise ValueError("target_eps and C must be positive; delta must be in (0, 1)")
    if X.device.type != "cpu":
        raise ValueError("the current QNN training path requires CPU tensors")

    set_experiment_seed(init_seed)
    model = model_factory()
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    flat_size = sum(parameter.numel() for parameter in parameters)

    unique_units, groups = _unit_row_groups(unit_ids, len(X))
    n_units = len(unique_units)
    if n_units < 2:
        raise ValueError("DP training needs at least two privacy units")
    if delta >= 1.0 / n_units:
        raise ValueError(
            f"delta={delta} must be below 1/n_private_units={1.0 / n_units:.6g}"
        )

    steps_per_epoch = max(1, math.ceil(n_units / expected_batch_units))
    sample_rate = 1.0 / steps_per_epoch
    expected_batch_size = n_units * sample_rate
    total_steps = epochs * steps_per_epoch
    sigma = get_sigma(target_eps, delta, sample_rate, total_steps)

    sampler_generator = _make_seeded_generator(sampling_seed, "Poisson sampling")
    noise_generator = _make_seeded_generator(noise_seed, "Gaussian noise")

    accountant = RDPAccountant()
    sampled_unit_total = 0
    empty_steps = 0

    for _ in range(total_steps):
        selected_mask = torch.rand(n_units, generator=sampler_generator) < sample_rate
        selected_units = selected_mask.nonzero(as_tuple=False).flatten().tolist()
        sampled_unit_total += len(selected_units)
        clipped_gradients = []

        for unit_position in selected_units:
            row_indices = torch.as_tensor(groups[unit_position], dtype=torch.long)
            model.zero_grad(set_to_none=True)
            unit_loss = mse_loss(model(X[row_indices]), y[row_indices])
            unit_loss.backward()
            gradient = torch.cat(
                [parameter.grad.reshape(-1) for parameter in parameters]
            )
            norm = gradient.norm()
            clipped_gradients.append(
                gradient * torch.clamp(C / (norm + 1e-12), max=1.0)
            )

        if clipped_gradients:
            gradient_sum = torch.stack(clipped_gradients).sum(dim=0)
        else:
            empty_steps += 1
            gradient_sum = torch.zeros(flat_size, dtype=parameters[0].dtype)

        noise = _generate_gaussian_noise(
            sigma * C,
            gradient_sum,
            noise_generator,
        )
        private_gradient = (gradient_sum + noise) / expected_batch_size

        with torch.no_grad():
            offset = 0
            for parameter in parameters:
                count = parameter.numel()
                parameter -= lr * private_gradient[offset : offset + count].view_as(
                    parameter
                )
                offset += count
        accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)

    realized_epsilon = accountant.get_epsilon(delta=delta, alphas=RDP_ALPHAS)
    info = {
        "target_eps": target_eps,
        "realized_eps": realized_epsilon,
        "delta": delta,
        "sigma": sigma,
        "C": C,
        "sample_rate": sample_rate,
        "steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "expected_batch_units": expected_batch_size,
        "mean_sampled_units": sampled_unit_total / total_steps,
        "empty_steps": empty_steps,
        "n_private_units": n_units,
        "sampler": "independent_poisson",
        "accountant": "opacus_rdp",
        "randomness": "separately_seeded_torch_generators",
        "academic_simulation": True,
        "code_version": QPRIV_VERSION,
    }
    return model, info


def public_gradient_diagnostics(
    model: nn.Module,
    X_public: torch.Tensor,
    y_public: torch.Tensor,
    C: float,
    unit_ids: Optional[Iterable] = None,
) -> dict:
    """Compute diagnostics on public data only.

    This is safe post-processing of a DP model because the queried records are
    explicitly public. The returned key names retain the ``public_`` prefix to
    prevent them from being mistaken for private-training diagnostics.
    """

    _, groups = _unit_row_groups(unit_ids, len(X_public))
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    norms = []
    for rows in groups:
        row_indices = torch.as_tensor(rows, dtype=torch.long)
        model.zero_grad(set_to_none=True)
        loss = mse_loss(model(X_public[row_indices]), y_public[row_indices])
        loss.backward()
        gradient = torch.cat([parameter.grad.reshape(-1) for parameter in parameters])
        norms.append(float(gradient.norm().item()))
    model.zero_grad(set_to_none=True)
    values = np.asarray(norms)
    return {
        "public_unit_count": len(values),
        "public_grad_norm_mean": float(values.mean()),
        "public_grad_norm_max": float(values.max()),
        "public_clip_frac": float((values > C).mean()),
    }


def train_dp_opacus_row_level(
    model_factory: Callable[[], nn.Module],
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    target_eps: float,
    delta: float,
    C: float,
    epochs: int,
    lr: float,
    expected_batch_rows: int,
    init_seed: int,
):
    """Reference MLP path for row-level cross-checks.

    This function intentionally rejects grouped subjects; the manual path is
    required for subject-level WESAD training.
    """

    from opacus import PrivacyEngine
    from torch.utils.data import DataLoader, TensorDataset

    set_experiment_seed(init_seed)
    model = model_factory()
    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=expected_batch_rows,
        shuffle=True,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"opacus\.privacy_engine",
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"opacus\.accountants\.analysis\.rdp",
        )
        engine = PrivacyEngine(accountant="rdp")
        model, optimizer, loader = engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            target_epsilon=target_eps,
            target_delta=delta,
            epochs=epochs,
            max_grad_norm=C,
            poisson_sampling=True,
        )
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model, {
        "target_eps": target_eps,
        "realized_eps": engine.accountant.get_epsilon(
            delta=delta,
            alphas=RDP_ALPHAS,
        ),
        "delta": delta,
        "C": C,
        "sampler": "opacus_independent_poisson",
        "randomness": "torch_default",
        "academic_simulation": True,
        "code_version": QPRIV_VERSION,
    }
