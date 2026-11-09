from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import math
import os
from pathlib import Path
import random
import subprocess
import sys

from src.concert.tree_csv import load_edges, load_nodes
from src.concert.infer_parameters import find_tree_file_pairs
from src.analyses.crc11.schedule_search import (
    DEFAULT_ELITE_COUNT,
    DEFAULT_MAX_COHORTS,
    DEFAULT_STAGNATION_COHORTS,
    DEFAULT_COHORT_SIZE,
    GRID_SIZE,
    MINIMUM_LL_IMPROVEMENT,
    RANDOM_IMMIGRANT_FRACTION,
    ScheduleFailure,
    ScheduleResult,
    _read_log_likelihood,
    _schedule_times,
    _write_cohort_summary,
    _write_summary,
    candidate_grid_indices,
    genetic_schedules,
    random_schedules,
)
from src.concert.tree_utils import prepare_nx_tree


def _run_schedule(
    schedule_number: int,
    cohort: int,
    times: tuple[float, float, float],
    input_dir: Path,
    output_dir: Path,
    project_root: Path,
) -> ScheduleResult:
    schedule_id = f"schedule_{schedule_number:04d}"
    schedule_dir = output_dir / schedule_id
    schedule_dir.mkdir(parents=True, exist_ok=True)
    result_path = schedule_dir / "inference.txt"
    optimizer_log = schedule_dir / "optimizer.log"
    stdout_log = schedule_dir / "stdout.log"
    (schedule_dir / "transition_times.txt").write_text(
        "".join(f"{time_value:.17g}\n" for time_value in times)
    )
    (schedule_dir / "transition_events.txt").write_text(
        f"PT->LN {times[0]:.17g}\n"
        f"PT->LN {times[1]:.17g}\n"
        f"NC->PT {times[2]:.17g}\n"
    )
    command = [
        sys.executable,
        "-m",
        "src.concert",
        "--state-specific-transitions",
        str(input_dir),
        "--nc-to-pt-time",
        f"{times[2]:.17g}",
        "--pt-to-ln-time",
        f"{times[0]:.17g}",
        "--second-pt-to-ln-time",
        f"{times[1]:.17g}",
        "--output",
        str(result_path),
        "--log",
        str(optimizer_log),
        "--device",
        "cpu",
    ]
    environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = "1"
    with stdout_log.open("w") as log:
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{schedule_id} failed with exit code {completed.returncode}; "
            f"see {stdout_log}."
        )
    return ScheduleResult(
        schedule_id=schedule_id,
        transition_times=times,
        log_likelihood=_read_log_likelihood(result_path),
        result_path=result_path,
        cohort=cohort,
    )


def run_state_specific_schedule_search(
    input_dir: Path,
    output_dir: Path,
    workers: int = 4,
    *,
    summary_dir: Path | None = None,
    cohorts: int = DEFAULT_MAX_COHORTS,
    cohort_size: int = DEFAULT_COHORT_SIZE,
    elite_count: int = DEFAULT_ELITE_COUNT,
    seed: int = 42,
) -> list[ScheduleResult]:
    if workers < 1:
        raise ValueError("workers must be at least one.")
    if cohorts < 1 or cohort_size < 1:
        raise ValueError("cohorts and cohort_size must be at least one.")
    if not 0 <= elite_count < cohort_size:
        raise ValueError("elite_count must be nonnegative and less than cohort_size.")

    project_root = Path(__file__).resolve().parents[3]
    input_dir = input_dir if input_dir.is_absolute() else project_root / input_dir
    output_dir = output_dir if output_dir.is_absolute() else project_root / output_dir
    summary_dir = summary_dir or (
        Path("outputs/sgootr/state_specific_schedule_search") / output_dir.name
    )
    summary_dir = (
        summary_dir if summary_dir.is_absolute() else project_root / summary_dir
    )
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Tree input directory does not exist: {input_dir}")
    edge_file = input_dir / "transition_edges.txt"
    if not edge_file.is_file():
        raise FileNotFoundError(f"Transition graph file does not exist: {edge_file}")
    tree_pairs = find_tree_file_pairs(input_dir)
    if len(tree_pairs) != 1:
        raise ValueError(
            f"CRC11 schedule search expects one tree; found {len(tree_pairs)}."
        )
    node_path, edge_path = tree_pairs[0]
    tree = prepare_nx_tree(load_nodes(str(node_path)), load_edges(str(edge_path)))
    root_time = float(tree.nodes[tree.graph["root"]]["T"])
    candidate_indices = candidate_grid_indices(root_time, exclude_zero_time=True)
    if len(candidate_indices) < 3:
        raise ValueError("At least three candidate transition times are required.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Schedule-search output directory must be empty: {output_dir}"
        )
    if summary_dir.exists() and any(summary_dir.iterdir()):
        raise FileExistsError(
            f"Schedule-search summary directory must be empty: {summary_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    seen: set[tuple[int, int, int]] = set()
    results: list[ScheduleResult] = []
    failures: list[ScheduleFailure] = []
    elite_results: list[ScheduleResult] = []
    cohort_rows: list[dict[str, object]] = []
    schedule_number = 1
    best_likelihood = -math.inf
    stagnant_cohorts = 0
    effective_workers = min(workers, os.cpu_count() or 1, cohort_size)

    for cohort in range(1, cohorts + 1):
        if cohort == 1:
            new_schedules = random_schedules(
                candidate_indices,
                cohort_size,
                rng,
                seen,
            )
            cohort_elites: list[ScheduleResult] = []
        else:
            cohort_elites = sorted(
                elite_results,
                key=lambda result: result.log_likelihood,
                reverse=True,
            )[:elite_count]
            new_schedules = genetic_schedules(
                cohort_elites,
                candidate_indices,
                cohort_size - len(cohort_elites),
                rng,
                seen,
                random_immigrant_count=math.ceil(
                    (cohort_size - len(cohort_elites))
                    * RANDOM_IMMIGRANT_FRACTION
                ),
            )

        cohort_results = list(cohort_elites)
        cohort_failures: list[ScheduleFailure] = []
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures: dict[
                Future[ScheduleResult],
                tuple[str, tuple[float, float, float]],
            ] = {}
            for schedule in new_schedules:
                times = _schedule_times(schedule)
                this_schedule_number = schedule_number
                schedule_number += 1
                schedule_id = f"schedule_{this_schedule_number:04d}"
                future = executor.submit(
                    _run_schedule,
                    this_schedule_number,
                    cohort,
                    times,
                    input_dir,
                    output_dir,
                    project_root,
                )
                futures[future] = (schedule_id, times)
            for future in as_completed(futures):
                schedule_id, times = futures[future]
                try:
                    cohort_results.append(future.result())
                except (OSError, RuntimeError, ValueError) as error:
                    cohort_failures.append(
                        ScheduleFailure(
                            schedule_id=schedule_id,
                            transition_times=times,
                            error=str(error),
                            cohort=cohort,
                        )
                    )

        results.extend(
            result for result in cohort_results if result.cohort == cohort
        )
        failures.extend(cohort_failures)
        ranked_cohort = sorted(
            cohort_results,
            key=lambda result: result.log_likelihood,
            reverse=True,
        )
        if ranked_cohort:
            cohort_best = ranked_cohort[0]
            if cohort_best.log_likelihood > best_likelihood + MINIMUM_LL_IMPROVEMENT:
                best_likelihood = cohort_best.log_likelihood
                stagnant_cohorts = 0
            else:
                stagnant_cohorts += 1
            elite_results = ranked_cohort[:elite_count]
            cohort_rows.append(
                {
                    "cohort": cohort,
                    "population_size": len(cohort_elites) + len(new_schedules),
                    "new_fits": len(new_schedules),
                    "successful_fits": len(cohort_results) - len(cohort_elites),
                    "failed_fits": len(cohort_failures),
                    "best_log_likelihood": f"{cohort_best.log_likelihood:.15g}",
                    "best_schedule_id": cohort_best.schedule_id,
                    "best_transition_times": " ".join(
                        f"{time_value:.17g}"
                        for time_value in cohort_best.transition_times
                    ),
                }
            )
            print(
                f"Cohort {cohort}: {len(cohort_results) - len(cohort_elites)} "
                f"new fits succeeded, {len(cohort_failures)} failed; "
                f"best LL={cohort_best.log_likelihood:.15g} "
                f"at {cohort_best.transition_times}.",
                flush=True,
            )
        else:
            elite_results = []
            stagnant_cohorts += 1
            cohort_rows.append(
                {
                    "cohort": cohort,
                    "population_size": len(cohort_elites) + len(new_schedules),
                    "new_fits": len(new_schedules),
                    "successful_fits": 0,
                    "failed_fits": len(cohort_failures),
                    "best_log_likelihood": "",
                    "best_schedule_id": "",
                    "best_transition_times": "",
                }
            )
        _write_cohort_summary(summary_dir, cohort_rows)
        _write_summary(
            summary_dir,
            results,
            failures,
            results_dir=project_root,
        )
        if stagnant_cohorts >= DEFAULT_STAGNATION_COHORTS:
            print(
                f"Stopping after {stagnant_cohorts} cohorts without a global "
                f"improvement greater than {MINIMUM_LL_IMPROVEMENT:g}.",
                flush=True,
            )
            break

    best_result = _write_summary(
        summary_dir,
        results,
        failures,
        results_dir=project_root,
    )
    if best_result is None:
        raise RuntimeError("No state-specific schedules completed successfully.")
    print(
        f"Best schedule: {best_result.schedule_id}; "
        f"cohort={best_result.cohort}; "
        f"LL={best_result.log_likelihood:.15g}; "
        f"times={best_result.transition_times}",
        flush=True,
    )
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Genetically search S15-style state-specific CRC11 schedules with "
            "one NC->PT and two PT->LN epochs on the 0.001 time grid."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cohorts", type=int, default=DEFAULT_MAX_COHORTS)
    parser.add_argument("--cohort-size", type=int, default=DEFAULT_COHORT_SIZE)
    parser.add_argument("--elite-count", type=int, default=DEFAULT_ELITE_COUNT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_state_specific_schedule_search(
        args.input_dir,
        args.output_dir,
        args.workers,
        summary_dir=args.summary_dir,
        cohorts=args.cohorts,
        cohort_size=args.cohort_size,
        elite_count=args.elite_count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
