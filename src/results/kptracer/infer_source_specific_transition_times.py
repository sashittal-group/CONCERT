from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import time

import torch

from src.dtos import OptimizationResult
from src.main2 import read_free_edges, write_result
from src.main2_joint_transition_times import _load_trees
from src.mle_closed_form2 import (
    PhylodynamicParameters,
    calculate_total_log_likelihood,
    optimize_parameters,
)
from src.progressive_joint_transition_times import _build_datastructures
from src.results.tls.data import load_inference_parameters


STATE_NAMES = ("T1", "L1", "L2", "L3", "S1")
SOURCE_STATES = (0, 2, 3, 4)
SOURCE_TIMES_FILENAME = "transition_times_by_source.tsv"
SAMPLING_PROBABILITY = 0.05


def _group_source_times(
    source_times: dict[int, float],
) -> tuple[list[float], list[list[int]]]:
    if set(source_times) != set(SOURCE_STATES):
        raise ValueError(f"Expected transition times for sources {SOURCE_STATES}")
    if any(
        not math.isfinite(value) or not 0.0 < value < 1.0
        for value in source_times.values()
    ):
        raise ValueError("Source-specific transition times must be in (0, 1)")
    times = sorted(set(source_times.values()))
    active_sources = [
        [
            source
            for source in SOURCE_STATES
            if source_times[source] == transition_time
        ]
        for transition_time in times
    ]
    return times, active_sources


def _result_from_seed(
    inference_path: Path,
    source_times: dict[int, float],
) -> OptimizationResult:
    parameters = load_inference_parameters(inference_path, STATE_NAMES)
    times, _ = _group_source_times(source_times)
    likelihood_line = next(
        (
            line.partition("=")[2]
            for line in inference_path.read_text().splitlines()
            if line.startswith("LL=")
        ),
        None,
    )
    if likelihood_line is None:
        raise ValueError(f"No log likelihood found in {inference_path}")
    return OptimizationResult(
        log_likelihood=torch.tensor(float(likelihood_line), dtype=torch.float64),
        birth_rates=parameters.birth_rates,
        death_rates=parameters.death_rates,
        sampling_probabilities=torch.full(
            (len(STATE_NAMES),),
            SAMPLING_PROBABILITY,
            dtype=torch.float64,
        ),
        transition_probabilities=parameters.transition_probabilities,
        transition_times=torch.tensor(times, dtype=torch.float64),
    )


def _initial_time_starts(
    seed_times: list[float],
    num_starts: int,
) -> list[dict[int, float]]:
    if not 1 <= num_starts <= len(SOURCE_STATES):
        raise ValueError(
            f"num_starts must be between 1 and {len(SOURCE_STATES)}"
        )
    if len(seed_times) < len(SOURCE_STATES):
        raise ValueError(
            f"Need at least {len(SOURCE_STATES)} seed transition times"
        )
    anchors = [
        float(seed_times[round(index * (len(seed_times) - 1) / 3)])
        for index in range(len(SOURCE_STATES))
    ]
    if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in anchors):
        raise ValueError("Seed transition times must be finite values in (0, 1)")
    return [
        {
            source: anchors[(index + offset) % len(SOURCE_STATES)]
            for index, source in enumerate(SOURCE_STATES)
        }
        for offset in range(num_starts)
    ]


def _fit_parameters(
    trees,
    source_times: dict[int, float],
    free_edges: list[tuple[int, int]],
    result: OptimizationResult,
    max_steps: int,
    patience: int,
    logfile: Path,
) -> OptimizationResult:
    times, active_sources = _group_source_times(source_times)
    datastructures = _build_datastructures(
        trees,
        times,
        len(STATE_NAMES),
    )
    return optimize_parameters(
        trees=trees,
        datastructure=datastructures,
        num_states=len(STATE_NAMES),
        given_transition_times=times,
        free_edges=free_edges,
        initial_birth_rates=result.birth_rates.detach().cpu().tolist(),
        initial_death_rates=result.death_rates.detach().cpu().tolist(),
        initial_transition_probabilities=(
            result.transition_probabilities.detach().cpu().tolist()
        ),
        given_sampling_probabilities=[
            SAMPLING_PROBABILITY
        ] * len(STATE_NAMES),
        transition_sources_by_time=active_sources,
        logfile=str(logfile),
        optimizer_name="adam",
        max_steps=max_steps,
        patience=patience,
        device=torch.device("cpu"),
        normalize_root_likelihood=True,
    )


def _fixed_parameter_likelihood(
    trees,
    source_times: dict[int, float],
    free_edges: list[tuple[int, int]],
    result: OptimizationResult,
) -> float:
    times, active_sources = _group_source_times(source_times)
    parameters = PhylodynamicParameters(
        num_states=len(STATE_NAMES),
        free_edges=free_edges,
        given_transition_times=times,
        device=torch.device("cpu"),
        initial_birth_rates=result.birth_rates.detach().cpu().tolist(),
        initial_death_rates=result.death_rates.detach().cpu().tolist(),
        initial_transition_probabilities=(
            result.transition_probabilities.detach().cpu().tolist()
        ),
        given_sampling_probabilities=[
            SAMPLING_PROBABILITY
        ] * len(STATE_NAMES),
        transition_sources_by_time=active_sources,
    )
    datastructures = _build_datastructures(
        trees,
        times,
        len(STATE_NAMES),
    )
    with torch.no_grad():
        likelihood = calculate_total_log_likelihood(
            parameters,
            trees,
            datastructures,
            static_transition_times=times,
            normalize_root_likelihood=True,
            transition_sources_by_time=active_sources,
        )
    return float(likelihood.item())


def _refine_source_times(
    trees,
    source_times: dict[int, float],
    free_edges: list[tuple[int, int]],
    result: OptimizationResult,
    grid_size: int,
) -> tuple[dict[int, float], float]:
    candidates = [index / grid_size for index in range(1, grid_size)]
    best_likelihood = _fixed_parameter_likelihood(
        trees,
        source_times,
        free_edges,
        result,
    )
    refined = dict(source_times)
    for source in SOURCE_STATES:
        best_time = refined[source]
        for candidate in candidates:
            proposal = dict(refined)
            proposal[source] = candidate
            likelihood = _fixed_parameter_likelihood(
                trees,
                proposal,
                free_edges,
                result,
            )
            if math.isfinite(likelihood) and likelihood > best_likelihood:
                best_time = candidate
                best_likelihood = likelihood
        refined[source] = best_time
        print(
            f"SOURCE_TIME source={STATE_NAMES[source]} time={best_time:.6f} "
            f"LL={best_likelihood:.12f}",
            flush=True,
        )
    return refined, best_likelihood


def _write_source_times(path: Path, source_times: dict[int, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output_file:
        writer = csv.writer(output_file, delimiter="\t")
        writer.writerow(("source_index", "source_state", "transition_time"))
        for source in SOURCE_STATES:
            writer.writerow(
                (source, STATE_NAMES[source], f"{source_times[source]:.15g}")
            )


def run_inference(
    tree_directory: Path,
    seed_path: Path,
    transition_edges_path: Path,
    output_path: Path,
    source_times_path: Path,
    max_steps: int = 50000,
    patience: int = 500,
    grid_size: int = 100,
    rounds: int = 1,
    num_starts: int = 4,
) -> OptimizationResult:
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    torch.set_num_threads(1)
    free_edges = read_free_edges(str(transition_edges_path))
    if {source for source, _ in free_edges} != set(SOURCE_STATES):
        raise ValueError(
            "The transition graph must have outgoing edges exactly from "
            f"{SOURCE_STATES}"
        )
    trees = _load_trees(str(tree_directory), num_trees=1)
    seed_parameters = load_inference_parameters(seed_path, STATE_NAMES)
    initial_time_starts = _initial_time_starts(
        seed_parameters.transition_times,
        num_starts,
    )
    start_time = time.perf_counter()
    best_result = None
    best_source_times = None
    for start_index, initial_source_times in enumerate(initial_time_starts, start=1):
        source_times = dict(initial_source_times)
        result = _result_from_seed(seed_path, source_times)
        print(
            f"START={start_index} initial_source_times={source_times}",
            flush=True,
        )
        for round_index in range(1, rounds + 1):
            result = _fit_parameters(
                trees,
                source_times,
                free_edges,
                result,
                max_steps,
                patience,
                output_path.with_name(
                    f"{output_path.stem}_start_{start_index}_"
                    f"round_{round_index}_parameters.log"
                ),
            )
            source_times, _ = _refine_source_times(
                trees,
                source_times,
                free_edges,
                result,
                grid_size,
            )
            print(
                f"START={start_index} ROUND={round_index} "
                f"source_times={source_times}",
                flush=True,
            )
        result = _fit_parameters(
            trees,
            source_times,
            free_edges,
            result,
            max_steps,
            patience,
            output_path.with_name(
                f"{output_path.stem}_start_{start_index}_final_parameters.log"
            ),
        )
        times, _ = _group_source_times(source_times)
        result.transition_times = torch.tensor(times, dtype=torch.float64)
        start_result_path = output_path.with_name(
            f"{output_path.stem}_start_{start_index}.out"
        )
        write_result(str(start_result_path), result, len(STATE_NAMES))
        _write_source_times(
            output_path.with_name(
                f"{output_path.stem}_start_{start_index}_source_times.tsv"
            ),
            source_times,
        )
        if (
            best_result is None
            or float(result.log_likelihood) > float(best_result.log_likelihood)
        ):
            best_result = result
            best_source_times = source_times
    if best_result is None or best_source_times is None:
        raise RuntimeError("No source-specific fit completed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_result(str(output_path), best_result, len(STATE_NAMES))
    _write_source_times(source_times_path, best_source_times)
    print(f"Final log likelihood: {float(best_result.log_likelihood):.12f}")
    print(f"Source-specific transition times: {best_source_times}")
    print(f"Result written to {output_path}")
    print(f"Source-time mapping written to {source_times_path}")
    print(f"Total execution time: {time.perf_counter() - start_time:.2f} seconds")
    return best_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Jointly fit KPTracer parameters with one transition time per "
            "active source state."
        )
    )
    parser.add_argument("--tree-directory", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--transition-edges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-times-output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--starts", type=int, default=4)
    arguments = parser.parse_args()
    run_inference(
        tree_directory=arguments.tree_directory,
        seed_path=arguments.seed,
        transition_edges_path=arguments.transition_edges,
        output_path=arguments.output,
        source_times_path=arguments.source_times_output,
        max_steps=arguments.max_steps,
        patience=arguments.patience,
        grid_size=arguments.grid_size,
        rounds=arguments.rounds,
        num_starts=arguments.starts,
    )


if __name__ == "__main__":
    main()
