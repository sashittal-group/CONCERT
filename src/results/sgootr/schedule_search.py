from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import csv
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
import math
import os
from pathlib import Path
import random
import subprocess
import sys

from src.dataprocessor import load_edges, load_nodes
from src.main2 import find_tree_file_pairs
from src.tree_utils import prepare_nx_tree


GRID_SIZE = 1000
SCHEDULE_SIZE = 3
DEFAULT_COHORT_SIZE = 500
DEFAULT_ELITE_COUNT = 50
DEFAULT_MAX_COHORTS = 10
DEFAULT_STAGNATION_COHORTS = 3
MINIMUM_LL_IMPROVEMENT = 1e-6
MUTATION_RATE = 0.15
RANDOM_IMMIGRANT_FRACTION = 0.20
LOCAL_MUTATION_RADIUS = 50
GLOBAL_MUTATION_PROBABILITY = 0.25
TOURNAMENT_SIZE = 3

ScheduleIndices = tuple[int, int, int]


@dataclass(frozen=True)
class ScheduleResult:
    schedule_id: str
    transition_times: tuple[float, float, float]
    log_likelihood: float
    result_path: Path
    cohort: int


@dataclass(frozen=True)
class ScheduleFailure:
    schedule_id: str
    transition_times: tuple[float, float, float]
    error: str
    cohort: int


def random_schedules(
    candidate_indices: list[int],
    count: int,
    rng: random.Random,
    excluded: set[ScheduleIndices] | None = None,
) -> list[ScheduleIndices]:
    candidate_indices = sorted(set(candidate_indices))
    if len(candidate_indices) < SCHEDULE_SIZE:
        raise ValueError("At least three candidate grid times are required.")
    if count < 0:
        raise ValueError("Schedule count must be nonnegative.")

    seen = set() if excluded is None else excluded
    valid_seen_count = sum(
        1
        for schedule in seen
        if all(index in candidate_indices for index in schedule)
    )
    available_schedules = math.comb(len(candidate_indices), SCHEDULE_SIZE)
    if count > available_schedules - valid_seen_count:
        raise ValueError("Requested more unique schedules than remain available.")
    schedules: list[ScheduleIndices] = []
    while len(schedules) < count:
        schedule = tuple(sorted(rng.sample(candidate_indices, SCHEDULE_SIZE)))
        if schedule not in seen:
            seen.add(schedule)
            schedules.append(schedule)
    return schedules


def _tournament_parent(
    elites: list[ScheduleResult],
    rng: random.Random,
) -> ScheduleResult:
    contestants = rng.sample(elites, min(TOURNAMENT_SIZE, len(elites)))
    return max(contestants, key=lambda result: result.log_likelihood)


def _mutate_transition_time(
    current_index: int,
    candidate_indices: list[int],
    rng: random.Random,
) -> int:
    alternatives = [index for index in candidate_indices if index != current_index]
    if rng.random() < GLOBAL_MUTATION_PROBABILITY:
        return rng.choice(alternatives)

    local_alternatives = [
        index
        for index in alternatives
        if abs(index - current_index) <= LOCAL_MUTATION_RADIUS
    ]
    return rng.choice(local_alternatives or alternatives)


def genetic_schedules(
    elites: list[ScheduleResult],
    candidate_indices: list[int],
    count: int,
    rng: random.Random,
    seen: set[ScheduleIndices],
    *,
    random_immigrant_count: int = 0,
) -> list[ScheduleIndices]:
    candidate_indices = sorted(set(candidate_indices))
    if not elites:
        return random_schedules(candidate_indices, count, rng, seen)
    if len(candidate_indices) < SCHEDULE_SIZE:
        raise ValueError("At least three candidate grid times are required.")
    if not 0 <= random_immigrant_count <= count:
        raise ValueError("random_immigrant_count must be between zero and count.")
    valid_seen_count = sum(
        1
        for schedule in seen
        if all(index in candidate_indices for index in schedule)
    )
    available_schedules = math.comb(len(candidate_indices), SCHEDULE_SIZE)
    if count > available_schedules - valid_seen_count:
        raise ValueError("Requested more unique schedules than remain available.")

    schedules = random_schedules(
        candidate_indices,
        random_immigrant_count,
        rng,
        seen,
    )
    while len(schedules) < count:
        parent_a = _tournament_parent(elites, rng)
        parent_b = _tournament_parent(elites, rng)
        parent_a_indices = tuple(round(time * GRID_SIZE) for time in parent_a.transition_times)
        parent_b_indices = tuple(round(time * GRID_SIZE) for time in parent_b.transition_times)

        child = [
            rng.choice((parent_a_indices[locus], parent_b_indices[locus]))
            for locus in range(SCHEDULE_SIZE)
        ]
        for locus in range(SCHEDULE_SIZE):
            if rng.random() < MUTATION_RATE:
                child[locus] = _mutate_transition_time(
                    child[locus],
                    candidate_indices,
                    rng,
                )

        child = sorted(set(child))
        if len(child) < SCHEDULE_SIZE:
            missing = [
                index for index in candidate_indices
                if index not in child
            ]
            child.extend(rng.sample(missing, SCHEDULE_SIZE - len(child)))
            child.sort()

        schedule = tuple(child)
        if schedule in seen:
            continue
        seen.add(schedule)
        schedules.append(schedule)
    return schedules


def candidate_grid_indices(
    root_time: float,
    *,
    exclude_zero_time: bool = False,
) -> list[int]:
    if not math.isfinite(root_time) or root_time <= 0.0:
        raise ValueError("Root time must be finite and positive.")
    return [
        index
        for index in range(1 if exclude_zero_time else 0, GRID_SIZE)
        if index / GRID_SIZE < root_time
    ]


def _schedule_times(schedule: ScheduleIndices) -> tuple[float, float, float]:
    return (
        schedule[0] / GRID_SIZE,
        schedule[1] / GRID_SIZE,
        schedule[2] / GRID_SIZE,
    )


def _read_log_likelihood(result_path: Path) -> float:
    values = [
        line.partition("=")[2]
        for line in result_path.read_text().splitlines()
        if line.startswith("LL=")
    ]
    if len(values) != 1:
        raise ValueError(
            f"Expected exactly one LL entry in {result_path}; found {len(values)}."
        )
    try:
        value = float(values[0])
    except ValueError as error:
        raise ValueError(f"Invalid log likelihood in {result_path}.") from error
    if not math.isfinite(value):
        raise ValueError(f"Log likelihood in {result_path} must be finite.")
    return value


def _run_schedule(
    schedule_number: int,
    cohort: int,
    transition_times: tuple[float, float, float],
    input_dir: Path,
    transition_edges_file: Path,
    output_dir: Path,
    project_root: Path,
) -> ScheduleResult:
    schedule_id = f"schedule_{schedule_number:04d}"
    schedule_dir = output_dir / schedule_id
    schedule_dir.mkdir(parents=True, exist_ok=True)

    schedule_path = schedule_dir / "transition_times.txt"
    schedule_path.write_text(
        "".join(f"{time:.17g}\n" for time in transition_times)
    )
    result_path = schedule_dir / "inference.txt"
    optimizer_log = schedule_dir / "optimizer.log"
    stdout_log = schedule_dir / "stdout.log"

    command = [
        sys.executable,
        "-m",
        "src.main2",
        str(input_dir),
        "--num-states",
        "3",
        "--num-trees",
        "1",
        "--transition-edges-file",
        str(transition_edges_file),
        "--transition-times-file",
        str(schedule_path),
        "--preserve-zero-transition-time",
        "--output",
        str(result_path),
        "--log",
        str(optimizer_log),
        "--device",
        "cpu",
        "--optimizer",
        "lbfgs",
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
        transition_times=transition_times,
        log_likelihood=_read_log_likelihood(result_path),
        result_path=result_path,
        cohort=cohort,
    )


def _write_summary(
    output_dir: Path,
    results: list[ScheduleResult],
    failures: list[ScheduleFailure],
    *,
    results_dir: Path | None = None,
) -> ScheduleResult | None:
    ranked_results = sorted(
        results,
        key=lambda result: result.log_likelihood,
        reverse=True,
    )
    summary_path = output_dir / "schedule_search_summary.csv"
    with summary_path.open("w", newline="") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=[
                "rank",
                "schedule_id",
                "cohort",
                "status",
                "log_likelihood",
                "transition_times",
                "result_path",
                "error",
            ],
        )
        writer.writeheader()
        for rank, result in enumerate(ranked_results, start=1):
            try:
                result_path = result.result_path.relative_to(
                    results_dir or output_dir
                )
            except ValueError:
                result_path = result.result_path
            writer.writerow(
                {
                    "rank": rank,
                    "schedule_id": result.schedule_id,
                    "cohort": result.cohort,
                    "status": "complete",
                    "log_likelihood": f"{result.log_likelihood:.15g}",
                    "transition_times": " ".join(
                        f"{time:.17g}" for time in result.transition_times
                    ),
                    "result_path": result_path,
                    "error": "",
                }
            )
        for failure in sorted(failures, key=lambda item: item.schedule_id):
            writer.writerow(
                {
                    "rank": "",
                    "schedule_id": failure.schedule_id,
                    "cohort": failure.cohort,
                    "status": "failed",
                    "log_likelihood": "",
                    "transition_times": " ".join(
                        f"{time:.17g}" for time in failure.transition_times
                    ),
                    "result_path": Path(failure.schedule_id) / "inference.txt",
                    "error": failure.error,
                }
            )

    if not ranked_results:
        return None
    best_result = ranked_results[0]
    (output_dir / "best_schedule.txt").write_text(
        "".join(f"{time:.17g}\n" for time in best_result.transition_times)
    )
    return best_result


def _write_cohort_summary(
    output_dir: Path,
    rows: list[dict[str, object]],
) -> None:
    summary_path = output_dir / "cohort_summary.csv"
    with summary_path.open("w", newline="") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=[
                "cohort",
                "population_size",
                "new_fits",
                "successful_fits",
                "failed_fits",
                "best_log_likelihood",
                "best_schedule_id",
                "best_transition_times",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_schedule_search(
    input_dir: Path,
    transition_edges_file: Path,
    output_dir: Path,
    workers: int,
    *,
    cohorts: int = DEFAULT_MAX_COHORTS,
    cohort_size: int = DEFAULT_COHORT_SIZE,
    elite_count: int = DEFAULT_ELITE_COUNT,
    seed: int = 42,
    exclude_zero_time: bool = False,
    summary_dir: Path | None = None,
) -> list[ScheduleResult]:
    if workers < 1:
        raise ValueError("workers must be at least one.")
    if cohorts < 1:
        raise ValueError("cohorts must be at least one.")
    if cohort_size < 1:
        raise ValueError("cohort_size must be at least one.")
    if not 0 <= elite_count < cohort_size:
        raise ValueError("elite_count must be nonnegative and less than cohort_size.")

    project_root = Path(__file__).resolve().parents[3]
    input_dir = input_dir if input_dir.is_absolute() else project_root / input_dir
    transition_edges_file = (
        transition_edges_file
        if transition_edges_file.is_absolute()
        else project_root / transition_edges_file
    )
    output_dir = (
        output_dir if output_dir.is_absolute() else project_root / output_dir
    )
    summary_dir = summary_dir or (
        Path("outputs/sgootr/schedule_search") / output_dir.name
    )
    summary_dir = (
        summary_dir if summary_dir.is_absolute() else project_root / summary_dir
    )
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Tree input directory does not exist: {input_dir}")
    if not transition_edges_file.is_file():
        raise FileNotFoundError(
            f"Transition graph file does not exist: {transition_edges_file}"
        )

    tree_file_pairs = find_tree_file_pairs(input_dir)
    if len(tree_file_pairs) != 1:
        raise ValueError(
            f"CRC11 schedule search expects one tree; found {len(tree_file_pairs)}."
        )
    node_path, edge_path = tree_file_pairs[0]
    with redirect_stdout(StringIO()):
        tree = prepare_nx_tree(
            load_nodes(str(node_path)),
            load_edges(str(edge_path)),
        )
    root_time = float(tree.nodes[tree.graph["root"]]["T"])
    candidate_indices = candidate_grid_indices(
        root_time,
        exclude_zero_time=exclude_zero_time,
    )
    if len(candidate_indices) < SCHEDULE_SIZE:
        raise ValueError(
            f"Only {len(candidate_indices)} grid times precede the root; "
            "at least three are required."
        )

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
    seen: set[ScheduleIndices] = set()
    results: list[ScheduleResult] = []
    failures: list[ScheduleFailure] = []
    cohort_rows: list[dict[str, object]] = []
    elite_results: list[ScheduleResult] = []
    next_schedule_number = 1
    best_likelihood = -math.inf
    stagnant_cohorts = 0
    available_cpus = os.cpu_count() or 1
    effective_workers = min(workers, available_cpus, cohort_size)
    first_grid_index = 1 if exclude_zero_time else 0
    print(
        f"Grid: i/{GRID_SIZE} for i={first_grid_index}..{GRID_SIZE - 1}; "
        f"{len(candidate_indices)} times precede root time {root_time:.15g}. "
        f"Evaluating cohorts of {cohort_size}, retaining {elite_count} elites, "
        f"up to {cohorts} cohorts with {effective_workers} workers."
    )

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
                schedule_number = next_schedule_number
                next_schedule_number += 1
                schedule_id = f"schedule_{schedule_number:04d}"
                times = _schedule_times(schedule)
                futures[
                    executor.submit(
                        _run_schedule,
                        schedule_number,
                        cohort,
                        times,
                        input_dir,
                        transition_edges_file,
                        output_dir,
                        project_root,
                    )
                ] = (schedule_id, times)

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
                        f"{time:.17g}" for time in cohort_best.transition_times
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
        raise RuntimeError(
            "No schedules completed successfully; see the schedule summary."
        )
    print(
        f"Best schedule: {best_result.schedule_id}; "
        f"cohort={best_result.cohort}; "
        f"LL={best_result.log_likelihood:.15g}; "
        f"times={best_result.transition_times}"
    )
    print(f"Summary: {output_dir / 'schedule_search_summary.csv'}")
    print(f"Cohort summary: {summary_dir / 'cohort_summary.csv'}")
    print(f"Best schedule file: {summary_dir / 'best_schedule.txt'}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Search CRC11 three-time schedules with a genetic algorithm over "
            "the 0.001-spaced grid."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--transition-edges-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cohorts", type=int, default=DEFAULT_MAX_COHORTS)
    parser.add_argument("--cohort-size", type=int, default=DEFAULT_COHORT_SIZE)
    parser.add_argument("--elite-count", type=int, default=DEFAULT_ELITE_COUNT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-zero-time",
        action="store_true",
        help="Exclude the zero-time grid point; the smallest candidate is 0.001.",
    )
    args = parser.parse_args()

    run_schedule_search(
        args.input_dir,
        args.transition_edges_file,
        args.output_dir,
        args.workers,
        cohorts=args.cohorts,
        cohort_size=args.cohort_size,
        elite_count=args.elite_count,
        seed=args.seed,
        exclude_zero_time=args.exclude_zero_time,
        summary_dir=args.summary_dir,
    )


if __name__ == "__main__":
    main()
