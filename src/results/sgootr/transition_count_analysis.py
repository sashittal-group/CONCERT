from __future__ import annotations

import argparse
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import csv
from dataclasses import dataclass
from io import StringIO
import math
from pathlib import Path
import random
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import torch

from src.dataprocessor import load_edges, load_nodes
from src.main2 import find_tree_file_pairs
from src.main2_state_specific_transitions import _write_result
from src.mle_state_specific_transitions import (
    EPOCH_EDGES,
    optimize_parameters,
)
from src.results.sgootr.schedule_search import (
    DEFAULT_COHORT_SIZE,
    DEFAULT_ELITE_COUNT,
    DEFAULT_MAX_COHORTS,
    DEFAULT_STAGNATION_COHORTS,
    GRID_SIZE,
    MINIMUM_LL_IMPROVEMENT,
    RANDOM_IMMIGRANT_FRACTION,
)
from src.tree_utils import (
    cut_tree,
    get_datastructure_for_likelihood_calculation,
    prepare_nx_tree,
)


_WORKER_TREE: nx.DiGraph | None = None


@dataclass(frozen=True)
class ScheduleCandidate:
    schedule_id: str
    indices: tuple[int, ...]
    times: tuple[float, ...]
    log_likelihood: float
    cohort: int


@dataclass(frozen=True)
class ScheduleFailure:
    schedule_id: str
    times: tuple[float, ...]
    error: str
    cohort: int


def _epoch_edges(num_ptln_times: int) -> tuple[tuple[int, int], ...]:
    if num_ptln_times < 1:
        raise ValueError("At least one PT->LN transition time is required.")
    return (EPOCH_EDGES[1],) * num_ptln_times + (EPOCH_EDGES[0],)


def _cohort_label(input_dir: Path) -> str:
    parts = input_dir.name.split("_")
    if (
        len(parts) < 3
        or parts[0] not in {"S10", "S15"}
        or not parts[1].startswith("CRC")
    ):
        raise ValueError(
            "The input directory must identify a cohort and CRC, "
            "for example S10_CRC04_transition_chain_input."
        )
    return f"{parts[0]} {parts[1]}"


def _initialize_worker(input_dir: str) -> None:
    global _WORKER_TREE
    torch.set_num_threads(1)
    with redirect_stdout(StringIO()):
        tree_pairs = find_tree_file_pairs(Path(input_dir), num_trees=1)
        if len(tree_pairs) != 1:
            raise ValueError(
                f"Transition-count analysis expects one tree; "
                f"found {len(tree_pairs)}."
            )
        node_path, edge_path = tree_pairs[0]
        _WORKER_TREE = prepare_nx_tree(
            load_nodes(str(node_path)),
            load_edges(str(edge_path)),
        )


def _fit_schedule(
    times: tuple[float, ...],
    num_ptln_times: int,
) -> tuple[float | None, str]:
    if _WORKER_TREE is None:
        raise RuntimeError("The transition-count worker has no loaded tree.")
    epoch_edges = _epoch_edges(num_ptln_times)
    try:
        with redirect_stdout(StringIO()):
            levels, hierarchy = cut_tree(_WORKER_TREE, times)
            datastructure = get_datastructure_for_likelihood_calculation(
                levels,
                hierarchy,
                times,
                num_states=3,
            )
        result = optimize_parameters(
            [_WORKER_TREE],
            [datastructure],
            list(times),
            epoch_edges=epoch_edges,
            device="cpu",
            lbfgs_max_iter=20,
        )
        likelihood = float(result.log_likelihood.item())
        if not math.isfinite(likelihood):
            raise FloatingPointError("The fitted likelihood is not finite.")
        return likelihood, ""
    except (ArithmeticError, RuntimeError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}"


def _random_schedules(
    candidate_indices: Sequence[int],
    schedule_size: int,
    count: int,
    rng: random.Random,
    seen: set[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    if len(candidate_indices) < schedule_size:
        raise ValueError("Not enough grid times for this transition count.")
    available = math.comb(len(candidate_indices), schedule_size)
    if count > available - len(seen):
        raise ValueError("Requested more unique schedules than remain available.")
    schedules = []
    while len(schedules) < count:
        schedule = tuple(sorted(rng.sample(candidate_indices, schedule_size)))
        if schedule not in seen:
            seen.add(schedule)
            schedules.append(schedule)
    return schedules


def _tournament(
    elites: list[ScheduleCandidate],
    rng: random.Random,
) -> ScheduleCandidate:
    contestants = rng.sample(elites, min(3, len(elites)))
    return max(contestants, key=lambda candidate: candidate.log_likelihood)


def _genetic_schedules(
    elites: list[ScheduleCandidate],
    candidate_indices: Sequence[int],
    schedule_size: int,
    count: int,
    rng: random.Random,
    seen: set[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    if not elites:
        return _random_schedules(
            candidate_indices,
            schedule_size,
            count,
            rng,
            seen,
        )
    random_immigrant_count = math.ceil(count * RANDOM_IMMIGRANT_FRACTION)
    schedules = _random_schedules(
        candidate_indices,
        schedule_size,
        random_immigrant_count,
        rng,
        seen,
    )
    while len(schedules) < count:
        parent_a = _tournament(elites, rng)
        parent_b = _tournament(elites, rng)
        child = [
            rng.choice((parent_a.indices[locus], parent_b.indices[locus]))
            for locus in range(schedule_size)
        ]
        for locus in range(schedule_size):
            if rng.random() >= 0.15:
                continue
            alternatives = [index for index in candidate_indices if index != child[locus]]
            if rng.random() < 0.25:
                child[locus] = rng.choice(alternatives)
            else:
                local = [
                    index for index in alternatives
                    if abs(index - child[locus]) <= 50
                ]
                child[locus] = rng.choice(local or alternatives)
        child = sorted(set(child))
        if len(child) < schedule_size:
            available = [index for index in candidate_indices if index not in child]
            child.extend(rng.sample(available, schedule_size - len(child)))
            child.sort()
        schedule = tuple(child)
        if schedule not in seen:
            seen.add(schedule)
            schedules.append(schedule)
    return schedules


def _write_schedule_summary(
    path: Path,
    candidates: list[ScheduleCandidate],
    failures: list[ScheduleFailure],
) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "schedule_id",
                "cohort",
                "status",
                "log_likelihood",
                "transition_times",
                "error",
            ],
        )
        writer.writeheader()
        for candidate in sorted(
            candidates,
            key=lambda item: item.log_likelihood,
            reverse=True,
        ):
            writer.writerow(
                {
                    "schedule_id": candidate.schedule_id,
                    "cohort": candidate.cohort,
                    "status": "complete",
                    "log_likelihood": f"{candidate.log_likelihood:.15g}",
                    "transition_times": " ".join(
                        f"{value:.17g}" for value in candidate.times
                    ),
                    "error": "",
                }
            )
        for failure in sorted(failures, key=lambda item: item.schedule_id):
            writer.writerow(
                {
                    "schedule_id": failure.schedule_id,
                    "cohort": failure.cohort,
                    "status": "failed",
                    "log_likelihood": "",
                    "transition_times": " ".join(
                        f"{value:.17g}" for value in failure.times
                    ),
                    "error": failure.error,
                }
            )


def _tree_root_time(input_dir: Path) -> float:
    with redirect_stdout(StringIO()):
        pairs = find_tree_file_pairs(input_dir, num_trees=1)
        if len(pairs) != 1:
            raise ValueError(
                f"Transition-count analysis expects one tree; "
                f"found {len(pairs)}."
            )
        node_path, edge_path = pairs[0]
        tree = prepare_nx_tree(load_nodes(str(node_path)), load_edges(str(edge_path)))
    return float(tree.nodes[tree.graph["root"]]["T"])


def _save_best_fit(
    input_dir: Path,
    output_dir: Path,
    times: tuple[float, ...],
    num_ptln_times: int,
) -> Path:
    with redirect_stdout(StringIO()):
        pairs = find_tree_file_pairs(input_dir, num_trees=1)
        if len(pairs) != 1:
            raise ValueError(
                f"Transition-count analysis expects one tree; "
                f"found {len(pairs)}."
            )
        node_path, edge_path = pairs[0]
        tree = prepare_nx_tree(load_nodes(str(node_path)), load_edges(str(edge_path)))
        levels, hierarchy = cut_tree(tree, times)
        datastructure = get_datastructure_for_likelihood_calculation(
            levels,
            hierarchy,
            times,
            num_states=3,
        )
    fit_dir = output_dir / "best"
    result = optimize_parameters(
        [tree],
        [datastructure],
        list(times),
        epoch_edges=_epoch_edges(num_ptln_times),
        logfile=fit_dir / "optimizer.log",
        device="cpu",
        lbfgs_max_iter=20,
    )
    inference_path = fit_dir / "inference.txt"
    _write_result(inference_path, result, _epoch_edges(num_ptln_times))
    (fit_dir / "transition_times.txt").write_text(
        "".join(f"{time_value:.17g}\n" for time_value in times)
    )
    return inference_path


def _search_count(
    input_dir: Path,
    output_dir: Path,
    num_ptln_times: int,
    *,
    workers: int,
    cohorts: int,
    cohort_size: int,
    elite_count: int,
    seed: int,
) -> tuple[float, tuple[float, ...], Path]:
    root_time = _tree_root_time(input_dir)
    candidate_indices = [
        index
        for index in range(1, GRID_SIZE)
        if index / GRID_SIZE < root_time
    ]
    schedule_size = num_ptln_times + 1
    if len(candidate_indices) < schedule_size:
        raise ValueError("The tree does not contain enough distinct grid times.")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = set()
    candidates: list[ScheduleCandidate] = []
    failures: list[ScheduleFailure] = []
    elites: list[ScheduleCandidate] = []
    next_schedule_number = 1
    best_likelihood = -math.inf
    stagnant_cohorts = 0
    summary_path = output_dir / "schedule_search_summary.csv"

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(str(input_dir),),
    ) as executor:
        for cohort in range(1, cohorts + 1):
            if cohort == 1:
                new_schedules = _random_schedules(
                    candidate_indices,
                    schedule_size,
                    cohort_size,
                    rng,
                    seen,
                )
                cohort_elites: list[ScheduleCandidate] = []
            else:
                cohort_elites = sorted(
                    elites,
                    key=lambda item: item.log_likelihood,
                    reverse=True,
                )[:elite_count]
                new_schedules = _genetic_schedules(
                    cohort_elites,
                    candidate_indices,
                    schedule_size,
                    cohort_size - len(cohort_elites),
                    rng,
                    seen,
                )

            cohort_results = list(cohort_elites)
            cohort_failures: list[ScheduleFailure] = []
            futures: dict[
                Future[tuple[float | None, str]],
                tuple[str, tuple[int, ...], tuple[float, ...]],
            ] = {}
            for indices in new_schedules:
                schedule_id = f"schedule_{next_schedule_number:04d}"
                next_schedule_number += 1
                times = tuple(index / GRID_SIZE for index in indices)
                future = executor.submit(_fit_schedule, times, num_ptln_times)
                futures[future] = (schedule_id, indices, times)

            for future in as_completed(futures):
                schedule_id, indices, times = futures[future]
                try:
                    likelihood, error = future.result()
                except (ArithmeticError, OSError, RuntimeError, ValueError) as error:
                    cohort_failures.append(
                        ScheduleFailure(
                            schedule_id,
                            times,
                            f"{type(error).__name__}: {error}",
                            cohort,
                        )
                    )
                    continue
                if likelihood is None:
                    cohort_failures.append(
                        ScheduleFailure(schedule_id, times, error, cohort)
                    )
                    continue
                cohort_results.append(
                    ScheduleCandidate(
                        schedule_id,
                        indices,
                        times,
                        likelihood,
                        cohort,
                    )
                )

            new_candidates = [
                candidate
                for candidate in cohort_results
                if candidate.cohort == cohort
            ]
            candidates.extend(new_candidates)
            failures.extend(cohort_failures)
            ranked = sorted(
                cohort_results,
                key=lambda item: item.log_likelihood,
                reverse=True,
            )
            if ranked:
                cohort_best = ranked[0]
                if cohort_best.log_likelihood > best_likelihood + MINIMUM_LL_IMPROVEMENT:
                    best_likelihood = cohort_best.log_likelihood
                    stagnant_cohorts = 0
                else:
                    stagnant_cohorts += 1
                elites = ranked[:elite_count]
                print(
                    f"{num_ptln_times} PT->LN times; cohort {cohort}: "
                    f"{len(new_candidates)} successful, {len(cohort_failures)} "
                    f"failed; best LL={cohort_best.log_likelihood:.15g} "
                    f"at {cohort_best.times}.",
                    flush=True,
                )
            else:
                elites = []
                stagnant_cohorts += 1
                print(
                    f"{num_ptln_times} PT->LN times; cohort {cohort}: "
                    f"no successful schedules.",
                    flush=True,
                )
            _write_schedule_summary(summary_path, candidates, failures)
            if stagnant_cohorts >= DEFAULT_STAGNATION_COHORTS:
                break

    if not candidates:
        raise RuntimeError(
            f"No successful schedules for {num_ptln_times} PT->LN times."
        )
    best = max(candidates, key=lambda item: item.log_likelihood)
    inference_path = _save_best_fit(
        input_dir,
        output_dir,
        best.times,
        num_ptln_times,
    )
    return best.log_likelihood, best.times, inference_path


def _read_likelihood(path: Path) -> float:
    matching = [
        line.partition("=")[2]
        for line in path.read_text().splitlines()
        if line.startswith("LL=")
    ]
    if len(matching) != 1:
        raise ValueError(f"Expected one LL entry in {path}.")
    value = float(matching[0])
    if not math.isfinite(value):
        raise ValueError(f"Likelihood in {path} is not finite.")
    return value


def _write_analysis(
    output_dir: Path,
    results: dict[int, tuple[float, tuple[float, ...], str]],
    cohort: str,
) -> tuple[Path, Path, Path]:
    rows = []
    previous_likelihood: float | None = None
    for count in sorted(results):
        likelihood, times, source = results[count]
        rows.append(
            {
                "num_ptln_transition_times": count,
                "num_ncpt_transition_times": 1,
                "log_likelihood": likelihood,
                "change_from_previous_count": (
                    ""
                    if previous_likelihood is None
                    else likelihood - previous_likelihood
                ),
                "transition_times": " ".join(f"{value:.17g}" for value in times),
                "source": source,
            }
        )
        previous_likelihood = likelihood

    csv_path = output_dir / "likelihood_by_ptln_transition_count.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    png_path = output_dir / "likelihood_by_ptln_transition_count.png"
    pdf_path = output_dir / "likelihood_by_ptln_transition_count.pdf"
    counts = [row["num_ptln_transition_times"] for row in rows]
    likelihoods = [row["log_likelihood"] for row in rows]
    figure, axis = plt.subplots(figsize=(7.5, 5))
    axis.plot(counts, likelihoods, marker="o", linewidth=2)
    axis.set_xticks(counts)
    axis.set_xlabel("Number of PT→LN transition times")
    axis.set_ylabel("Best log-likelihood")
    axis.set_title(
        f"{cohort} likelihood by PT→LN transition count\n"
        "(one NC→PT time; shared transition matrix per source state)"
    )
    axis.grid(True, linestyle=":", alpha=0.5)
    figure.tight_layout()
    figure.savefig(png_path, dpi=180)
    figure.savefig(pdf_path)
    plt.close(figure)
    return csv_path, png_path, pdf_path


def run_analysis(
    input_dir: Path,
    output_dir: Path,
    existing_two_count_inference: Path | None = None,
    *,
    counts: tuple[int, ...] = (1, 2, 3, 4, 5),
    workers: int = 4,
    cohorts: int = DEFAULT_MAX_COHORTS,
    cohort_size: int = DEFAULT_COHORT_SIZE,
    elite_count: int = DEFAULT_ELITE_COUNT,
    seed: int = 42,
) -> dict[int, tuple[float, tuple[float, ...], str]]:
    if not counts or any(count < 1 for count in counts):
        raise ValueError("Counts must be positive.")
    if len(set(counts)) != len(counts):
        raise ValueError("PT->LN transition counts must be unique.")
    if existing_two_count_inference is None and 2 not in counts:
        raise ValueError("Include count 2 unless an existing two-count fit is supplied.")
    if workers < 1 or cohorts < 1 or cohort_size < 1:
        raise ValueError("workers, cohorts, and cohort_size must be positive.")
    if not 0 <= elite_count < cohort_size:
        raise ValueError("elite_count must be nonnegative and less than cohort_size.")
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    cohort = _cohort_label(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Tree input directory does not exist: {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Analysis output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[int, tuple[float, tuple[float, ...], str]] = {}
    if existing_two_count_inference is not None:
        existing_two_count_inference = existing_two_count_inference.resolve()
        if not existing_two_count_inference.is_file():
            raise FileNotFoundError(
                "The existing two-PT->LN shared-source result was not found: "
                f"{existing_two_count_inference}"
            )
        existing_two_likelihood = _read_likelihood(existing_two_count_inference)
        existing_two_schedule = (
            existing_two_count_inference.parent.parent / "best_schedule.txt"
        )
        if not existing_two_schedule.is_file():
            raise FileNotFoundError(
                f"Best schedule for the existing two-time result is missing: "
                f"{existing_two_schedule}"
            )
        existing_two_times = tuple(
            float(value)
            for value in existing_two_schedule.read_text().split()
        )
        results[2] = (
            existing_two_likelihood,
            existing_two_times,
            str(existing_two_count_inference),
        )

    project_root = Path(__file__).resolve().parents[3]
    for count in counts:
        if count in results:
            continue
        count_output = output_dir / f"ptln_{count}"
        likelihood, times, inference_path = _search_count(
            input_dir,
            count_output,
            count,
            workers=workers,
            cohorts=cohorts,
            cohort_size=cohort_size,
            elite_count=elite_count,
            seed=seed + count,
        )
        try:
            inference_source = str(inference_path.relative_to(project_root))
        except ValueError:
            inference_source = str(inference_path)
        results[count] = (
            likelihood,
            times,
            inference_source,
        )
        _write_analysis(output_dir, results, cohort)
    output_paths = _write_analysis(output_dir, results, cohort)
    for path in output_paths:
        print(f"Wrote {path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze likelihood versus PT->LN transition-time count "
            "with one NC->PT time and shared source-state transition matrices."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--existing-two-count-inference", type=Path)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cohorts", type=int, default=DEFAULT_MAX_COHORTS)
    parser.add_argument("--cohort-size", type=int, default=DEFAULT_COHORT_SIZE)
    parser.add_argument("--elite-count", type=int, default=DEFAULT_ELITE_COUNT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_analysis(
        args.input_dir,
        args.output_dir,
        args.existing_two_count_inference,
        counts=tuple(args.counts),
        workers=args.workers,
        cohorts=args.cohorts,
        cohort_size=args.cohort_size,
        elite_count=args.elite_count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
