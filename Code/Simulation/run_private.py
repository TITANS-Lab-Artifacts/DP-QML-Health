"""Run the private sweep with safe spawned workers and parent-only checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import qpriv_utils as qp
from run_nonprivate import available_datasets, qnn_factory


def derived_seed(*parts) -> int:
    """Derive a stable independent seed from experiment identifiers."""

    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def model_spec(model_name: str, n_features: int, config: dict):
    """Return model constructor, clipping threshold, and provenance."""

    q_factory = qnn_factory(n_features, config)
    qnn_params = qp.count_params(q_factory())
    qnn_bound = qp.qnn_gradient_certificate(q_factory())
    if model_name == "qnn_C0.5_empirical":
        return q_factory, 0.5, "empirical_below_certificate", "qnn", qnn_bound
    if model_name == "qnn_C1.0_empirical":
        return q_factory, 1.0, "empirical_below_certificate", "qnn", qnn_bound
    if model_name == "qnn_Cbound_certified":
        return q_factory, qnn_bound, "certified_no_clipping", "qnn", qnn_bound
    if model_name == "mlp_C0.5":
        return (
            lambda: qp.build_matched_mlp(n_features, qnn_params),
            0.5,
            "empirical",
            "mlp",
            None,
        )
    if model_name == "mlp_C1.0":
        return (
            lambda: qp.build_matched_mlp(n_features, qnn_params),
            1.0,
            "empirical",
            "mlp",
            None,
        )
    if model_name == "logistic_C1.0":
        return (
            lambda: qp.LogisticModel(n_features),
            1.0,
            "empirical",
            "logistic",
            None,
        )
    raise ValueError(f"unknown private model: {model_name}")


def _prepared_data(dataset: str, seed: int, config: dict):
    return qp.split_preprocess(
        qp.load_clean(dataset),
        n_features=config["qnn"]["n_features"],
        split_seed=seed,
        public_seed=config["split"]["public_seed"],
        public_size=config["split"]["public_size"],
        test_size=config["split"]["test_size"],
    )


def _tune_private_task(task: tuple[str, int, dict, bool]) -> dict:
    """Tune all three model families once for one dataset/seed."""

    dataset, seed, config, demo = task
    qp.configure_worker()
    prepared = _prepared_data(dataset, seed, config)
    n_features = prepared.X_train.shape[1]
    representative_models = {
        "qnn": model_spec("qnn_C1.0_empirical", n_features, config)[0],
        "mlp": model_spec("mlp_C1.0", n_features, config)[0],
        "logistic": model_spec("logistic_C1.0", n_features, config)[0],
    }
    tune_epochs = 1 if demo else 20
    family_lrs = {}
    for family, factory in representative_models.items():
        candidates = config["training"]["lr_candidates"][family]
        if demo:
            candidates = candidates[:2]
        family_lrs[family] = qp.tune_lr_on_public_data(
            factory,
            prepared.X_public,
            prepared.y_public,
            candidates,
            seed=seed,
            epochs=tune_epochs,
        )
    preprocessor_path = (
        qp.ARTIFACT_DIR / f"{dataset}_seed{seed}_public_preprocessor.joblib"
    )
    qp.save_preprocessor(prepared.preprocessor, preprocessor_path)
    return {
        "dataset": dataset,
        "seed": seed,
        "qnn_lr": family_lrs["qnn"],
        "mlp_lr": family_lrs["mlp"],
        "logistic_lr": family_lrs["logistic"],
        "preprocessor_artifact": qp.project_relative(preprocessor_path),
        "quantum_device": config["qnn"]["device"],
        "diff_method": config["qnn"]["diff_method"],
        "code_version": qp.QPRIV_VERSION,
    }


def _private_task(task: tuple[str, int, float, str, dict, bool, dict]) -> dict:
    """Train and evaluate one private configuration in a spawned worker."""

    dataset, seed, eps, model_name, config, demo, lr_record = task
    qp.configure_worker()
    prepared = _prepared_data(dataset, seed, config)
    n_features = prepared.X_train.shape[1]
    factory, threshold, threshold_status, family, qnn_bound = model_spec(
        model_name,
        n_features,
        config,
    )
    sampling_seed = derived_seed(
        "sample",
        qp.RANDOMNESS_PROTOCOL_VERSION,
        dataset,
        model_name,
        eps,
        seed,
    )
    noise_seed = derived_seed(
        "noise",
        qp.RANDOMNESS_PROTOCOL_VERSION,
        dataset,
        model_name,
        eps,
        seed,
    )
    if demo:
        X_private = prepared.X_train[:64]
        y_private = prepared.y_train[:64]
        units_private = prepared.unit_ids_train[:64]
        epochs = 1
    else:
        X_private = prepared.X_train
        y_private = prepared.y_train
        units_private = prepared.unit_ids_train
        epochs = config["training"]["private_epochs"]
    learning_rate = lr_record[f"{family}_lr"]
    model, privacy = qp.train_dp_manual(
        factory,
        X_private,
        y_private,
        unit_ids=units_private,
        target_eps=eps,
        delta=config["delta"],
        C=threshold,
        epochs=epochs,
        lr=learning_rate,
        expected_batch_units=config["training"]["expected_batch_units"],
        init_seed=seed,
        sampling_seed=sampling_seed,
        noise_seed=noise_seed,
    )
    diagnostics = qp.public_gradient_diagnostics(
        model,
        prepared.X_public,
        prepared.y_public,
        threshold,
        unit_ids=prepared.unit_ids_public,
    )
    artifact_path = (
        qp.ARTIFACT_DIR / "private" / dataset / model_name / f"eps{eps}_seed{seed}.pt"
    )
    quantum_device = config["qnn"]["device"] if family == "qnn" else None
    diff_method = config["qnn"]["diff_method"] if family == "qnn" else None
    qp.save_model_state(
        model,
        artifact_path,
        {
            "dataset": dataset,
            "model": model_name,
            "family": family,
            "eps_target": eps,
            "seed": seed,
            "n_features": n_features,
            "C": threshold,
            "threshold_status": threshold_status,
            "quantum_device": quantum_device,
            "diff_method": diff_method,
            "randomness": "separately_seeded_torch_generators",
            "randomness_protocol_version": qp.RANDOMNESS_PROTOCOL_VERSION,
            "academic_simulation": True,
            "code_version": qp.QPRIV_VERSION,
        },
    )
    return {
        "dataset": dataset,
        "model": model_name,
        "family": family,
        "eps_target": eps,
        "seed": seed,
        "demo": demo,
        "selected_lr_public": learning_rate,
        "threshold_status": threshold_status,
        "qnn_certificate": qnn_bound,
        "params": qp.count_params(model),
        "model_artifact": qp.project_relative(artifact_path),
        "quantum_device": quantum_device,
        "diff_method": diff_method,
        "code_version": qp.QPRIV_VERSION,
        "randomness_protocol_version": qp.RANDOMNESS_PROTOCOL_VERSION,
        **prepared.split_info,
        **privacy,
        **diagnostics,
        **qp.evaluate(model, prepared.X_test, prepared.y_test),
    }


def _run_tasks(tasks: list, worker, workers: int):
    """Yield results using spawn, with a restricted-environment fallback."""

    if workers == 1:
        for task in tasks:
            yield worker(task)
        return
    context = multiprocessing.get_context("spawn")
    try:
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=context)
    except (OSError, PermissionError) as error:
        print(
            f"parallel workers unavailable ({error}); continuing safely with "
            "one worker",
            flush=True,
        )
        for task in tasks:
            yield worker(task)
        return
    with executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for future in as_completed(futures):
            yield future.result()


def main(
    demo: bool = False,
    progress_callback=None,
    workers: Optional[int] = None,
) -> None:
    """Tune once per dataset/seed, then run missing private configurations."""

    config = qp.load_json(qp.BASE_DIR / "config.json")
    if config["code_version"] != qp.QPRIV_VERSION:
        raise ValueError("config.json was created for another code version")
    datasets = available_datasets(config)
    eps_grid = config["eps_grid"]
    seeds = config["seeds"]
    model_names = config["private_models"]
    if demo:
        datasets = datasets[:1]
        eps_grid = [1.0]
        seeds = seeds[:1]
        model_names = [
            "qnn_C1.0_empirical",
            "mlp_C1.0",
            "logistic_C1.0",
        ]

    suffix = "_demo" if demo else ""
    results_path = qp.RESULTS_DIR / f"private_results{suffix}.json"
    lr_path = qp.RESULTS_DIR / f"private_lr_selection{suffix}.json"
    results = qp.load_json(results_path, default=[])
    results = [row for row in results if row.get("code_version") == qp.QPRIV_VERSION]
    lr_records = qp.load_json(lr_path, default=[])
    lr_records = [
        row for row in lr_records if row.get("code_version") == qp.QPRIV_VERSION
    ]
    lr_by_pair = {(row["dataset"], int(row["seed"])): row for row in lr_records}

    all_keys = [
        (dataset, model_name, eps, seed)
        for dataset in datasets
        for seed in seeds
        for eps in eps_grid
        for model_name in model_names
    ]
    done = {
        (row["dataset"], row["model"], row["eps_target"], int(row["seed"]))
        for row in results
    }
    pending_keys = [key for key in all_keys if key not in done]
    needed_pairs = sorted({(key[0], key[3]) for key in pending_keys})
    tune_tasks = [
        (dataset, seed, config, demo)
        for dataset, seed in needed_pairs
        if (dataset, seed) not in lr_by_pair
    ]
    tune_workers = qp.worker_count(workers, len(tune_tasks))
    print(
        f"private LR selection: {len(tune_tasks)} pending job(s), "
        f"{tune_workers} worker(s)",
        flush=True,
    )
    for lr_record in _run_tasks(tune_tasks, _tune_private_task, tune_workers):
        pair = (lr_record["dataset"], int(lr_record["seed"]))
        lr_by_pair[pair] = lr_record
        lr_records = list(lr_by_pair.values())
        qp.save_records(lr_records, lr_path, ["dataset", "seed"])

    tasks = [
        (
            dataset,
            seed,
            eps,
            model_name,
            config,
            demo,
            lr_by_pair[(dataset, seed)],
        )
        for dataset, model_name, eps, seed in pending_keys
    ]
    completed = len(all_keys) - len(tasks)
    for index in range(completed):
        if progress_callback:
            progress_callback(index + 1, len(all_keys), "private runs (resumed)")
    effective_workers = qp.worker_count(workers, len(tasks))
    print(
        f"private stage: {len(tasks)} pending job(s), {effective_workers} worker(s)",
        flush=True,
    )
    for row in _run_tasks(tasks, _private_task, effective_workers):
        results.append(row)
        result_frame = qp.save_records(
            results,
            results_path,
            ["dataset", "seed", "eps_target", "model"],
        )
        completed += 1
        print(
            f"{row['dataset']:14s} {row['model']:24s} "
            f"eps={row['eps_target']:<4} seed={row['seed']} "
            f"AUC={row['auc']:.3f} realized_eps={row['realized_eps']:.3f}",
            flush=True,
        )
        if progress_callback:
            progress_callback(
                completed,
                len(all_keys),
                f"private runs ({row['dataset']}, {row['model']})",
            )

    result_frame = qp.save_records(
        results,
        results_path,
        ["dataset", "seed", "eps_target", "model"],
    )
    qp.save_records(lr_records, lr_path, ["dataset", "seed"])
    qp.save_json(qp.environment_fingerprint(), qp.RESULTS_DIR / "environment.json")
    if not result_frame.empty:
        print(
            result_frame.groupby(["dataset", "model", "eps_target"])
            .auc.agg(["mean", "std"])
            .round(3),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    main(demo=args.demo, workers=args.workers)
