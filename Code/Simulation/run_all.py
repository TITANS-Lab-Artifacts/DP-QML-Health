"""Run the complete workflow with stage progress and an adaptive time estimate."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from analyze_results import main as analyze
from prepare_data import main as prepare_data
from run_ablation import main as run_ablation
from run_nonprivate import main as run_nonprivate
from run_private import main as run_private

ProgressCallback = Callable[[int, int, str], None]


def format_time(seconds: float) -> str:
    """Return a compact, human-readable duration."""

    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


@dataclass
class Stage:
    """One workflow stage and its approximate share of compute."""

    name: str
    weight: float
    action: Callable[[ProgressCallback], None]


class Progress:
    """Print progress and estimate the remaining runtime from completed work."""

    def __init__(self, stages: list[Stage]) -> None:
        self.stages = stages
        self.started = time.perf_counter()
        self.completed_weight = 0.0
        self.timed_weight = 0.0
        self.timed_seconds = 0.0
        self.total_weight = sum(stage.weight for stage in stages)
        self.active_stage: Optional[Stage] = None
        self.active_started = 0.0
        self.active_completed = 0
        self.active_total = 1

    def active_weight(self) -> float:
        if self.active_stage is None:
            return 0.0
        return self.active_stage.weight * self.active_completed / self.active_total

    def eta(self) -> str:
        current_weight = self.active_weight()
        remaining_weight = self.total_weight - self.completed_weight - current_weight
        measured_weight = self.timed_weight
        measured_seconds = self.timed_seconds
        if (
            self.active_stage is not None
            and self.active_stage.weight >= 1
            and current_weight > 0
        ):
            measured_weight += current_weight
            measured_seconds += time.perf_counter() - self.active_started
        if measured_weight == 0:
            return "calculating"
        seconds_per_weight = measured_seconds / measured_weight
        return f"about {format_time(remaining_weight * seconds_per_weight)}"

    def update(self, completed: int, total: int, detail: str) -> None:
        """Report progress within the currently running stage."""

        if total <= 0 or not 0 <= completed <= total:
            raise ValueError("progress must satisfy 0 <= completed <= total")
        self.active_completed = completed
        self.active_total = total
        effective_weight = self.completed_weight + self.active_weight()
        percent = 100 * effective_weight / self.total_weight
        elapsed = time.perf_counter() - self.started
        print(
            f"Progress: {percent:5.1f}% overall | {completed}/{total} {detail}\n"
            f"Elapsed: {format_time(elapsed)} | Expected time left: {self.eta()}",
            flush=True,
        )

    def run(self) -> None:
        for number, stage in enumerate(self.stages, start=1):
            self.active_stage = stage
            self.active_started = time.perf_counter()
            self.active_completed = 0
            self.active_total = 1
            elapsed = time.perf_counter() - self.started
            percent = 100 * self.completed_weight / self.total_weight
            print(
                f"\n[{number}/{len(self.stages)} | {percent:5.1f}%] "
                f"{stage.name}\n"
                f"Elapsed: {format_time(elapsed)} | Expected time left: {self.eta()}",
                flush=True,
            )
            stage_started = time.perf_counter()
            try:
                stage.action(self.update)
            except Exception:
                stage_elapsed = time.perf_counter() - stage_started
                print(
                    f"\nFAILED: {stage.name} after {format_time(stage_elapsed)}",
                    flush=True,
                )
                raise

            stage_elapsed = time.perf_counter() - stage_started
            self.completed_weight += stage.weight
            self.active_stage = None
            # Data download and final analysis do not predict training speed.
            if stage.weight >= 1:
                self.timed_weight += stage.weight
                self.timed_seconds += stage_elapsed
            elapsed = time.perf_counter() - self.started
            percent = 100 * self.completed_weight / self.total_weight
            print(
                f"Completed: {stage.name} in {format_time(stage_elapsed)} "
                f"({percent:.1f}% overall)\n"
                f"Elapsed: {format_time(elapsed)} | Expected time left: {self.eta()}",
                flush=True,
            )


def main(
    demo: bool,
    trust_wesad_pickle: bool = True,
    workers: Optional[int] = None,
) -> None:
    """Prepare data, train models, run the ablation, and analyze results."""

    # Weights reflect the relative number of expensive quantum training jobs.
    # They make the ETA more useful than treating every stage as equally costly.
    weights = [0.1, 5, 3, 2, 0.1] if demo else [0.1, 20, 300, 360, 0.1]
    stages = [
        Stage(
            "Prepare datasets",
            weights[0],
            lambda _report: prepare_data(
                trust_wesad_pickle=trust_wesad_pickle,
            ),
        ),
        Stage(
            "Train non-private models",
            weights[1],
            lambda report: run_nonprivate(
                demo,
                progress_callback=report,
                workers=workers,
            ),
        ),
        Stage(
            "Train private models",
            weights[2],
            lambda report: run_private(
                demo,
                progress_callback=report,
                workers=workers,
            ),
        ),
        Stage(
            "Run quantum ablation",
            weights[3],
            lambda report: run_ablation(
                demo,
                progress_callback=report,
                workers=workers,
            ),
        ),
        Stage(
            "Analyze and save results",
            weights[4],
            lambda _report: analyze(demo),
        ),
    ]
    mode = "demo" if demo else "complete"
    print(
        f"Unit-Aware DP-QML Health {mode} workflow: {len(stages)} stages.\n"
        "The ETA is calculated after the first training stage and adapts to this "
        "computer.\n"
        f"Parallel workers: {'automatic (up to 8)' if workers is None else workers}.",
        flush=True,
    )
    progress = Progress(stages)
    progress.run()
    print(
        f"\nWorkflow completed successfully in "
        f"{format_time(time.perf_counter() - progress.started)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo",
        action="store_true",
        help="quick one-dataset smoke test instead of the complete sweep",
    )
    parser.add_argument(
        "--trust-wesad-pickle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "trust Data/Simulation/WESAD/S*/S*.pkl files (default); use "
            "--no-trust-wesad-pickle to refuse loading them"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="spawn this many workers (default: CPU count minus 2, maximum 8)",
    )
    args = parser.parse_args()
    main(
        demo=args.demo,
        trust_wesad_pickle=args.trust_wesad_pickle,
        workers=args.workers,
    )
