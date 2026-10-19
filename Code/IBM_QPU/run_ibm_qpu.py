"""Compare full v4 private and non-private Pima QNNs on one IBM QPU.

Training was completed by the v4.0 academic simulation workflow. IBM hardware
is used only for inference on eight fixed, balanced public-test examples. Both
checkpoints are submitted together in one EstimatorV2 job so the comparison
uses the same backend, transpilation, observable, inputs, and shot setting.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Results" / "IBM_QPU"

SHOTS = 1024
MAX_QPU_SECONDS = 540
N_QUBITS = 4
N_LAYERS = 2
TRANSPILER_SEED = 2026

SIMULATION_CODE_VERSION = "4.0"
RANDOMNESS_PROTOCOL_VERSION = "3.1"
DATASET_SOURCE_SHA256 = (
    "6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af"
)

# Positions within the full v4 seed-0 Pima public-test partition. The selection
# rule was fixed before QPU execution: take the first four negative and first
# four positive examples. These are descriptive hardware-validation cases, not
# an independently powered clinical test set.
TEST_POSITIONS = np.array([0, 1, 2, 4, 3, 6, 14, 16], dtype=int)
PIMA_LABELS = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)

# The corresponding inputs after the v4 public-only median imputation,
# standardization, four-component PCA, and angle scaling. Both checkpoints use
# these exact inputs.
PIMA_ANGLES = np.array(
    [
        [0.3417709768, 2.3290631771, 2.1851043701, 1.6883946657],
        [0.4990563393, 2.1308593750, 1.7937645912, 1.9951997995],
        [0.4513387978, 2.4820897579, 1.4368503094, 1.9354919195],
        [1.7539356947, 1.7826480865, 1.8731423616, 1.8704657555],
        [1.8435709476, 2.1656727791, 2.3648653030, 2.3594217300],
        [2.7757458687, 2.3744647503, 2.3788189888, 2.3499121666],
        [1.9522427320, 2.1362729073, 1.8551278114, 1.8670727015],
        [1.3506504297, 1.3900259733, 3.1415927410, 1.1304181814],
    ],
    dtype=float,
)

# Full v4.0 Pima non-private QNN checkpoint, seed 0.
# Original artifact SHA-256:
# c471f4d7ce0c29291ea4e0e2e4e91e9c44afeea411d5861b7535321acc99ee09
NONPRIVATE_WEIGHTS = np.array(
    [
        [4.3021593094, 1.5700134039, 2.6000690460, 4.4756708145],
        [4.4836425781, 4.7882905006, 2.0056462288, 1.1443382502],
    ],
    dtype=float,
)

# Full v4.0 Pima private QNN checkpoint, seed 0:
# target epsilon=1, realized epsilon=0.9945662909471514, delta=1e-5,
# empirical clipping threshold C=1.
# Original artifact SHA-256:
# bbb4ee16de5a52d457de9fc24441989f7b84284d784ba7abe0072a42ffbe11af
PRIVATE_WEIGHTS = np.array(
    [
        [3.9626879692, 4.7062959671, 1.5246973038, 0.8933698535],
        [1.2248102427, 5.2193789482, 4.0125646591, 5.2810826302],
    ],
    dtype=float,
)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def save_json(payload: dict, path: Path) -> None:
    """Save a readable result record in the central Results directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str, allow_nan=False)
    print(f"Saved {path.name}")


def build_circuit():
    """Build the same four-qubit, two-layer RY QNN used in v4 simulations."""

    inputs = ParameterVector("x", N_QUBITS)
    weights = ParameterVector("theta", N_QUBITS * N_LAYERS)
    circuit = QuantumCircuit(N_QUBITS)
    for qubit in range(N_QUBITS):
        circuit.ry(inputs[qubit], qubit)
    for layer in range(N_LAYERS):
        for qubit in range(N_QUBITS):
            circuit.ry(weights[layer * N_QUBITS + qubit], qubit)
        for qubit in range(N_QUBITS):
            circuit.cx(qubit, (qubit + 1) % N_QUBITS)
    return circuit, inputs, weights


def parameter_row(circuit, inputs, weights, angles, checkpoint):
    """Return values in the parameter order of a logical or transpiled circuit."""

    lookup = {
        **{inputs[index].name: float(angles[index]) for index in range(N_QUBITS)},
        **{
            weights[index].name: float(checkpoint.reshape(-1)[index])
            for index in range(N_QUBITS * N_LAYERS)
        },
    }
    return [lookup[parameter.name] for parameter in circuit.parameters]


def parameter_matrix(circuit, inputs, weights):
    """Bind all non-private rows first, followed by all private rows."""

    rows = []
    for checkpoint in (NONPRIVATE_WEIGHTS, PRIVATE_WEIGHTS):
        for angles in PIMA_ANGLES:
            rows.append(parameter_row(circuit, inputs, weights, angles, checkpoint))
    return np.asarray(rows, dtype=float)


def split_probabilities(expectation_values):
    """Convert qubit-0 Z expectations into class-one probabilities."""

    values = np.asarray(expectation_values, dtype=float).reshape(-1)
    expected = 2 * len(PIMA_ANGLES)
    if values.shape != (expected,):
        raise ValueError(f"expected {expected} values, received {values.shape}")
    probabilities = (values + 1.0) / 2.0
    midpoint = len(PIMA_ANGLES)
    return probabilities[:midpoint], probabilities[midpoint:]


def descriptive_metrics(probabilities) -> dict:
    """Return descriptive metrics for the eight balanced hardware cases."""

    values = np.asarray(probabilities, dtype=float)
    predictions = (values >= 0.5).astype(int)
    return {
        "correct": int((predictions == PIMA_LABELS).sum()),
        "total": int(len(PIMA_LABELS)),
        "accuracy": float((predictions == PIMA_LABELS).mean()),
        "brier": float(np.mean((values - PIMA_LABELS) ** 2)),
    }


def agreement_metrics(exact, observed) -> dict:
    """Measure QPU probability disagreement relative to exact statevector."""

    residual = np.asarray(observed, dtype=float) - np.asarray(exact, dtype=float)
    return {
        "mean_signed_error": float(residual.mean()),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "max_absolute_error": float(np.abs(residual).max()),
    }


def experiment_metadata() -> dict:
    """Return fixed provenance shared by local and hardware result files."""

    return {
        "purpose": "QPU inference feasibility and exact-to-hardware agreement",
        "dataset": "pima",
        "dataset_source_sha256": DATASET_SOURCE_SHA256,
        "simulation_code_version": SIMULATION_CODE_VERSION,
        "randomness_protocol_version": RANDOMNESS_PROTOCOL_VERSION,
        "split_seed": 0,
        "public_preprocessing_seed": 2026,
        "preprocessing_fit": "public_calibration_only",
        "test_positions": TEST_POSITIONS.tolist(),
        "labels": PIMA_LABELS.tolist(),
        "selection_rule": "first four negative and first four positive test cases",
        "private_checkpoint": {
            "model": "qnn_C1.0_empirical",
            "seed": 0,
            "target_epsilon": 1.0,
            "realized_epsilon": 0.9945662909471514,
            "delta": 1e-5,
            "clipping_threshold": 1.0,
            "artifact_sha256": (
                "bbb4ee16de5a52d457de9fc24441989f7b84284d784ba7abe0072a42ffbe11af"
            ),
        },
        "nonprivate_checkpoint": {
            "model": "qnn",
            "seed": 0,
            "artifact_sha256": (
                "c471f4d7ce0c29291ea4e0e2e4e91e9c44afeea411d5861b7535321acc99ee09"
            ),
        },
    }


def local_check(circuit, observable, values, output_dir: Path):
    """Validate the circuit and save exact reference probabilities."""

    estimator = StatevectorEstimator()
    result = estimator.run([(circuit, observable, values)]).result()
    nonprivate, private = split_probabilities(result[0].data.evs)
    print("Exact non-private probabilities:", np.round(nonprivate, 4))
    print("Exact private probabilities:    ", np.round(private, 4))
    output = {
        "created_utc": utc_now(),
        "experiment": experiment_metadata(),
        "runtime": {"qiskit": version("qiskit")},
        "circuit": {
            "qubits": N_QUBITS,
            "layers": N_LAYERS,
            "logical_depth": circuit.depth(),
            "parameter_bindings": int(len(values)),
        },
        "exact_statevector": {
            "nonprivate_probabilities": nonprivate.tolist(),
            "private_probabilities": private.tolist(),
            "nonprivate_descriptive_metrics": descriptive_metrics(nonprivate),
            "private_descriptive_metrics": descriptive_metrics(private),
        },
    }
    save_json(output, output_dir / "ibm_local_validation.json")
    return nonprivate, private


def backend_name_string(backend) -> str:
    """Return a backend name across supported IBM Runtime backend versions."""

    name = backend.name
    return name() if callable(name) else str(name)


def run_hardware(
    circuit,
    inputs,
    weights,
    observable,
    exact_nonprivate,
    exact_private,
    output_dir: Path,
    backend_name=None,
):
    """Submit one cost-limited EstimatorV2 job to an IBM QPU."""

    from qiskit_ibm_runtime import (
        EstimatorV2 as RuntimeEstimator,
        QiskitRuntimeService,
    )

    service = QiskitRuntimeService()
    backend = (
        service.backend(backend_name)
        if backend_name
        else service.least_busy(operational=True, simulator=False)
    )
    if backend.num_qubits < N_QUBITS:
        raise RuntimeError("the selected backend has fewer than four qubits")
    selected_backend = backend_name_string(backend)

    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=3,
        seed_transpiler=TRANSPILER_SEED,
    )
    isa_circuit = pass_manager.run(circuit)
    isa_observable = observable.apply_layout(isa_circuit.layout)
    values = parameter_matrix(isa_circuit, inputs, weights)

    estimator = RuntimeEstimator(mode=backend)
    estimator.options.resilience_level = 0
    estimator.options.default_shots = SHOTS
    estimator.options.max_execution_time = MAX_QPU_SECONDS
    estimator.options.twirling.enable_gates = False
    estimator.options.twirling.enable_measure = False

    print("Backend:", selected_backend)
    print("ISA depth:", isa_circuit.depth())
    print("Parameter bindings:", len(values))
    print("Requested shot executions:", len(values) * SHOTS)

    job = estimator.run([(isa_circuit, isa_observable, values)])
    submission = {
        "submitted_utc": utc_now(),
        "backend": selected_backend,
        "job_id": job.job_id(),
        "shots_per_binding": SHOTS,
        "parameter_bindings": int(len(values)),
        "maximum_qpu_seconds": MAX_QPU_SECONDS,
    }
    save_json(submission, output_dir / "ibm_qpu_submission.json")
    print("Job ID:", job.job_id())

    result = job.result()
    nonprivate, private = split_probabilities(result[0].data.evs)
    metrics = job.metrics()
    output = {
        "completed_utc": utc_now(),
        "experiment": experiment_metadata(),
        "runtime": {
            "qiskit": version("qiskit"),
            "qiskit_ibm_runtime": version("qiskit-ibm-runtime"),
            "backend": selected_backend,
            "job_id": job.job_id(),
            "job_metrics": metrics,
        },
        "circuit": {
            "qubits": N_QUBITS,
            "layers": N_LAYERS,
            "logical_depth": circuit.depth(),
            "isa_depth": isa_circuit.depth(),
            "isa_size": isa_circuit.size(),
            "transpiler_optimization_level": 3,
            "transpiler_seed": TRANSPILER_SEED,
            "observable": "Z on logical qubit 0",
            "shots_per_binding": SHOTS,
            "parameter_bindings": int(len(values)),
            "requested_shot_executions": int(len(values) * SHOTS),
            "maximum_qpu_seconds": MAX_QPU_SECONDS,
        },
        "exact_statevector": {
            "nonprivate_probabilities": np.asarray(exact_nonprivate).tolist(),
            "private_probabilities": np.asarray(exact_private).tolist(),
        },
        "qpu": {
            "nonprivate_probabilities": nonprivate.tolist(),
            "private_probabilities": private.tolist(),
            "nonprivate_descriptive_metrics": descriptive_metrics(nonprivate),
            "private_descriptive_metrics": descriptive_metrics(private),
        },
        "agreement_to_exact": {
            "nonprivate": agreement_metrics(exact_nonprivate, nonprivate),
            "private": agreement_metrics(exact_private, private),
        },
    }
    save_json(output, output_dir / "ibm_qpu_results.json")
    print("QPU non-private probabilities:", np.round(nonprivate, 4))
    print("QPU private probabilities:    ", np.round(private, 4))
    print(
        "Reported QPU seconds:",
        metrics.get("usage", {}).get("quantum_seconds"),
    )


def main(
    run_on_hardware: bool,
    backend_name: Optional[str],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Run the exact local check and optionally submit the single IBM job."""

    print("qiskit", version("qiskit"))
    if run_on_hardware:
        print("qiskit-ibm-runtime", version("qiskit-ibm-runtime"))
    circuit, inputs, weights = build_circuit()
    observable = SparsePauliOp("IIIZ")
    values = parameter_matrix(circuit, inputs, weights)
    assert values.shape == (16, 12)
    exact_nonprivate, exact_private = local_check(
        circuit, observable, values, output_dir
    )

    if run_on_hardware:
        run_hardware(
            circuit,
            inputs,
            weights,
            observable,
            exact_nonprivate,
            exact_private,
            output_dir,
            backend_name,
        )
    else:
        print("Local check passed. Re-run with --hardware to use IBM QPU time.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hardware",
        action="store_true",
        help="submit one real IBM Estimator job after the exact local check",
    )
    parser.add_argument(
        "--backend",
        help="optional IBM backend name; default selects the least busy QPU",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="result destination (default: Results/IBM_QPU)",
    )
    arguments = parser.parse_args()
    main(arguments.hardware, arguments.backend, arguments.output_dir)
