"""Run non-private ceilings and classical baselines with safe parallel workers."""

from __future__ import annotations

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import qpriv_utils as qp


MODEL_NAMES = {
    "qnn",
    "mlp_matched",
    "logistic_torch",
    "logreg",
    "rf",
}


def available_datasets(config: dict) -> list[str]:
    """Return configured datasets that have a prepared CSV file."""

    return [
        name
        for name in config["datasets"]
        if (qp.DATA_DIR / f"{name}_clean.csv").exists()
    ]


def qnn_factory(n_features: int, config: dict):
    """Return the configured exact Lightning QNN constructor."""

    qnn = config["qnn"]
    return lambda: qp.QNN(
        n_features,
        qnn["n_layers"],
        encoding=qnn["encoding"],
        diff_method=qnn["diff_method"],
        device_name=qnn["device"],
    )


def _nonprivate_task(task: tuple[str, int, dict, bool]) -> list[dict]:
    """Run one dataset/seed job inside a spawned worker."""

    dataset, seed, config, demo = task
    qp.configure_worker()
    prepared = qp.split_preprocess(
        qp.load_clean(dataset),
        n_features=config["qnn"]["n_features"],
        split_seed=seed,
        public_seed=config["split"]["public_seed"],
        public_size=config["split"]["public_size"],
        test_size=config["split"]["test_size"],
    )
    n_features = prepared.X_train.shape[1]
    q_factory = qnn_factory(n_features, config)
    qnn_parameter_count = qp.count_params(q_factory())
    factories = {
        "qnn": q_factory,
        "mlp_matched": lambda: qp.build_matched_mlp(
            n_features,
            qnn_parameter_count,
        ),
        "logistic_torch": lambda: qp.LogisticModel(n_features),
    }
    family_key = {
        "qnn": "qnn",
        "mlp_matched": "mlp",
        "logistic_torch": "logistic",
    }
    train_epochs = 1 if demo else config["training"]["nonprivate_epochs"]
    tune_epochs = 1 if demo else 20
    restart_count = 1 if demo else 3
    records = []

    for model_name, factory in factories.items():
        candidates = config["training"]["lr_candidates"][family_key[model_name]]
        if demo:
            candidates = candidates[:2]
        selected_lr = qp.tune_lr_on_public_data(
            factory,
            prepared.X_public,
            prepared.y_public,
            candidates,
            seed=seed,
            epochs=tune_epochs,
        )
        X_train = prepared.X_train[:64] if demo else prepared.X_train
        y_train = prepared.y_train[:64] if demo else prepared.y_train
        model = qp.train_nonprivate(
            factory,
            X_train,
            y_train,
            epochs=train_epochs,
            lr=selected_lr,
            batch=config["training"]["expected_batch_units"],
            seed=seed,
            n_restarts=restart_count,
        )
        artifact_path = (
            qp.ARTIFACT_DIR / "nonprivate" / dataset / f"{model_name}_seed{seed}.pt"
        )
        qp.save_model_state(
            model,
            artifact_path,
            {
                "dataset": dataset,
                "model": model_name,
                "seed": seed,
                "n_features": n_features,
                "params": qp.count_params(model),
                "quantum_device": (
                    config["qnn"]["device"] if model_name == "qnn" else None
                ),
                "diff_method": (
                    config["qnn"]["diff_method"] if model_name == "qnn" else None
                ),
                "code_version": qp.QPRIV_VERSION,
            },
        )
        records.append(
            {
                "dataset": dataset,
                "model": model_name,
                "seed": seed,
                "selected_lr_public": selected_lr,
                "params": qp.count_params(model),
                "model_artifact": qp.project_relative(artifact_path),
                "quantum_device": (
                    config["qnn"]["device"] if model_name == "qnn" else None
                ),
                "diff_method": (
                    config["qnn"]["diff_method"] if model_name == "qnn" else None
                ),
                "code_version": qp.QPRIV_VERSION,
                **prepared.split_info,
                **qp.evaluate(model, prepared.X_test, prepared.y_test),
            }
        )

    sklearn_models = {
        "logreg": LogisticRegression(max_iter=2000, random_state=seed),
        "rf": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=1,
        ),
    }
    for model_name, model in sklearn_models.items():
        model.fit(prepared.X_train.numpy(), prepared.y_train.numpy())
        probabilities = model.predict_proba(prepared.X_test.numpy())[:, 1]
        labels = prepared.y_test.numpy()
        records.append(
            {
                "dataset": dataset,
                "model": model_name,
                "seed": seed,
                "selected_lr_public": None,
                "params": None,
                "model_artifact": None,
                "quantum_device": None,
                "diff_method": None,
                "code_version": qp.QPRIV_VERSION,
                **prepared.split_info,
                "auc": qp.roc_auc_score(labels, probabilities),
                "prauc": qp.average_precision_score(labels, probabilities),
                "acc": qp.accuracy_score(
                    labels,
                    (probabilities >= 0.5).astype(int),
                ),
                "brier": qp.brier_score_loss(labels, probabilities),
            }
        )

    qp.save_preprocessor(
        prepared.preprocessor,
        qp.ARTIFACT_DIR / f"{dataset}_seed{seed}_public_preprocessor.joblib",
    )
    return records


def _completed_pairs(records: list[dict]) -> set[tuple[str, int]]:
    """Return dataset/seed pairs containing all expected model rows."""

    frame = pd.DataFrame(records)
    if frame.empty:
        return set()
    return {
        (dataset, int(seed))
        for (dataset, seed), group in frame.groupby(["dataset", "seed"])
        if set(group["model"]) == MODEL_NAMES
    }


def main(
    demo: bool = False,
    progress_callback=None,
    workers: Optional[int] = None,
) -> None:
    """Run missing dataset/seed jobs and checkpoint results in the parent."""

    config = qp.load_json(qp.BASE_DIR / "config.json")
    if config["code_version"] != qp.QPRIV_VERSION:
        raise ValueError("config.json was created for another code version")
    datasets = available_datasets(config)
    seeds = config["seeds"][:1] if demo else config["seeds"]
    if demo:
        datasets = datasets[:1]

    results_path = qp.RESULTS_DIR / (
        "nonprivate_results_demo.json" if demo else "nonprivate_results.json"
    )
    records = qp.load_json(results_path, default=[])
    records = [row for row in records if row.get("code_version") == qp.QPRIV_VERSION]
    completed_pairs = _completed_pairs(records)
    # Remove incomplete pairs so they are recomputed as one deterministic unit.
    records = [
        row for row in records if (row["dataset"], int(row["seed"])) in completed_pairs
    ]
    all_tasks = [
        (dataset, seed, config, demo) for dataset in datasets for seed in seeds
    ]
    pending = [task for task in all_tasks if (task[0], task[1]) not in completed_pairs]
    completed = len(all_tasks) - len(pending)
    for index in range(completed):
        if progress_callback:
            progress_callback(index + 1, len(all_tasks), "non-private runs (resumed)")

    effective_workers = qp.worker_count(workers, len(pending))
    print(
        f"non-private stage: {len(pending)} pending job(s), "
        f"{effective_workers} worker(s)",
        flush=True,
    )

    def accept(task_records: list[dict]) -> None:
        nonlocal completed, records
        records.extend(task_records)
        result_frame = qp.save_records(
            records,
            results_path,
            ["dataset", "seed", "model"],
        )
        completed += 1
        first = task_records[0]
        if progress_callback:
            progress_callback(
                completed,
                len(all_tasks),
                f"non-private runs ({first['dataset']}, seed {first['seed']})",
            )
        print(
            result_frame.groupby(["dataset", "model"]).auc.mean().round(3),
            flush=True,
        )

    if effective_workers == 1:
        for task in pending:
            accept(_nonprivate_task(task))
    elif pending:
        context = multiprocessing.get_context("spawn")
        try:
            executor = ProcessPoolExecutor(
                max_workers=effective_workers,
                mp_context=context,
            )
        except (OSError, PermissionError) as error:
            print(
                f"parallel workers unavailable ({error}); continuing safely with "
                "one worker",
                flush=True,
            )
            for task in pending:
                accept(_nonprivate_task(task))
        else:
            with executor:
                futures = [executor.submit(_nonprivate_task, task) for task in pending]
                for future in as_completed(futures):
                    accept(future.result())

    qp.save_records(records, results_path, ["dataset", "seed", "model"])
    qp.save_json(qp.environment_fingerprint(), qp.RESULTS_DIR / "environment.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    main(demo=args.demo, workers=args.workers)
