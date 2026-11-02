from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import time

import torch

from src.main2 import read_free_edges, read_transition_times
from src.main2_joint_transition_times import (
    _fixed_parameter_likelihood,
    _load_trees,
)
from src.progressive_joint_transition_times import (
    _fit_parameters,
    _load_initial_result,
    _write_count_checkpoint,
)


def _fine_grid_candidates(
    transition_times: list[float],
    coordinate: int,
    max_time: float,
    grid_step: float,
) -> list[float]:
    if not 0 <= coordinate < len(transition_times):
        raise IndexError("coordinate is outside the transition-time schedule")
    if not math.isfinite(max_time) or max_time <= 0:
        raise ValueError("max_time must be positive and finite")
    if not math.isfinite(grid_step) or grid_step <= 0:
        raise ValueError("grid_step must be positive and finite")
    if any(
        not math.isfinite(value) or not 0.0 < value < max_time
        for value in transition_times
    ) or any(
        right <= left for left, right in zip(transition_times, transition_times[1:])
    ):
        raise ValueError("transition_times must be finite, ordered, and in range")

    lower = transition_times[coordinate - 1] if coordinate else 0.0
    upper = (
        transition_times[coordinate + 1]
        if coordinate + 1 < len(transition_times)
        else max_time
    )
    first_grid_index = math.floor(lower / grid_step) + 1
    last_grid_index = math.ceil(upper / grid_step) - 1
    candidates = {
        grid_index * grid_step
        for grid_index in range(first_grid_index, last_grid_index + 1)
        if lower < grid_index * grid_step < upper
    }
    candidates.add(transition_times[coordinate])
    return sorted(candidates)


def _result_likelihood(result) -> float:
    value = result.log_likelihood
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    likelihood = float(value)
    if not math.isfinite(likelihood):
        raise FloatingPointError("Parameter optimization returned a non-finite likelihood")
    return likelihood


def refine_transition_schedule(
    trees,
    transition_times: list[float],
    initial_result,
    num_states: int,
    free_edges: list[tuple[int, int]],
    device: torch.device,
    fixed_sampling_probability: float,
    max_time: float,
    grid_step: float,
    max_rounds: int,
    optimizer_name: str = "adam",
    learning_rate: float = 0.01,
    max_steps: int = 50000,
    patience: int = 500,
    normalize_root_likelihood: bool = True,
    verbose: bool = False,
):
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    if not transition_times:
        raise ValueError("transition_times must not be empty")

    times = list(transition_times)
    result = initial_result

    def likelihood(candidate_times: list[float]) -> float:
        return _fixed_parameter_likelihood(
            trees=trees,
            transition_times=candidate_times,
            num_states=num_states,
            free_edges=free_edges,
            result=result,
            device=device,
            normalize_root_likelihood=normalize_root_likelihood,
            fixed_sampling_probability=fixed_sampling_probability,
            verbose=verbose,
        )

    current_likelihood = likelihood(times)
    if not math.isfinite(current_likelihood):
        raise FloatingPointError(
            f"Initial parameters have a non-finite likelihood for times {times}"
        )
    rounds_completed = 0

    for round_index in range(1, max_rounds + 1):
        round_start_likelihood = current_likelihood
        print(
            f"REFINEMENT_ROUND_START count={len(times)} round={round_index} "
            f"LL={current_likelihood:.15f}",
            flush=True,
        )
        for coordinate in range(len(times)):
            candidates = _fine_grid_candidates(
                times,
                coordinate,
                max_time,
                grid_step,
            )
            best_times = list(times)
            best_likelihood = likelihood(times)
            for candidate in candidates:
                if candidate == times[coordinate]:
                    continue
                proposal = list(times)
                proposal[coordinate] = candidate
                candidate_likelihood = likelihood(proposal)
                if (
                    math.isfinite(candidate_likelihood)
                    and candidate_likelihood > best_likelihood
                ):
                    best_times = proposal
                    best_likelihood = candidate_likelihood

            old_time = times[coordinate]
            times = best_times
            result = _fit_parameters(
                trees=trees,
                transition_times=times,
                num_states=num_states,
                free_edges=free_edges,
                device=device,
                fixed_sampling_probability=fixed_sampling_probability,
                result=result,
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
                max_steps=max_steps,
                patience=patience,
                normalize_root_likelihood=normalize_root_likelihood,
                logfile=None,
                verbose=verbose,
            )
            current_likelihood = _result_likelihood(result)
            if result.transition_times is None:
                raise ValueError("Fitted result has no transition-time schedule")
            fitted_times = result.transition_times.detach().cpu().tolist()
            if any(
                not math.isclose(fitted, selected, rel_tol=0.0, abs_tol=1e-12)
                for fitted, selected in zip(fitted_times, times)
            ):
                raise ValueError("Fitted result changed the fixed transition times")
            print(
                f"REFINEMENT_COORDINATE count={len(times)} round={round_index} "
                f"coordinate={coordinate + 1} old_time={old_time:.7g} "
                f"new_time={times[coordinate]:.7g} "
                f"sweep_LL={best_likelihood:.15f} "
                f"refit_LL={current_likelihood:.15f} "
                f"grid_candidates={len(candidates)}",
                flush=True,
            )

        rounds_completed = round_index
        improvement = current_likelihood - round_start_likelihood
        print(
            f"REFINEMENT_ROUND_END count={len(times)} round={round_index} "
            f"LL={current_likelihood:.15f} improvement={improvement:.15f}",
            flush=True,
        )
        if improvement <= 1e-8:
            print(
                f"REFINEMENT_STOP count={len(times)} "
                "reason=no_improvement_after_full_round",
                flush=True,
            )
            break

    return result, rounds_completed


def _source_directory_for_count(
    count: int,
    small_source_directory: Path,
    nine_source_directory: Path,
    large_source_directory: Path,
) -> Path:
    if 5 <= count <= 8:
        return small_source_directory
    if count == 9:
        return nine_source_directory
    if 10 <= count <= 20:
        return large_source_directory
    raise ValueError(f"Unsupported transition count: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Refine saved transition schedules by fine coordinate sweeps, "
            "refitting model parameters after every coordinate."
        )
    )
    parser.add_argument("--tree-directory", type=Path, required=True)
    parser.add_argument("--num-states", type=int, default=5)
    parser.add_argument("--num-trees", type=int, default=1)
    parser.add_argument("--transition-edges-file", type=Path, required=True)
    parser.add_argument("--small-source-directory", type=Path, required=True)
    parser.add_argument("--nine-source-directory", type=Path, required=True)
    parser.add_argument("--large-source-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--fixed-sampling-probability", type=float, default=0.05)
    parser.add_argument("--max-time", type=float, default=1.0)
    parser.add_argument("--grid-step", type=float, default=0.0001)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--optimizer", choices=("adam", "lbfgs"), default="adam")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--normalize-root-likelihood", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not math.isfinite(args.fixed_sampling_probability) or not (
        0.0 <= args.fixed_sampling_probability <= 1.0
    ):
        raise ValueError("fixed_sampling_probability must be finite and in [0, 1]")
    if not math.isfinite(args.grid_step) or args.grid_step <= 0:
        raise ValueError("grid_step must be positive and finite")
    if args.max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")

    start = time.perf_counter()
    trees = _load_trees(str(args.tree_directory), args.num_trees)
    free_edges = read_free_edges(str(args.transition_edges_file))
    args.output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    history_path = args.output_directory / "refinement_summary.csv"

    with history_path.open("w", newline="") as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=(
                "transition_count",
                "initial_likelihood",
                "refined_likelihood",
                "rounds_completed",
                "transition_times",
            ),
        )
        writer.writeheader()
        for count in range(5, 21):
            source_directory = _source_directory_for_count(
                count,
                args.small_source_directory,
                args.nine_source_directory,
                args.large_source_directory,
            )
            times_path = source_directory / f"transition_count_{count}_times.txt"
            result_path = source_directory / f"transition_count_{count}.out"
            transition_times = read_transition_times(str(times_path))
            initial_result = _load_initial_result(
                result_path,
                args.num_states,
                free_edges,
                transition_times,
            )
            initial_likelihood = _fixed_parameter_likelihood(
                trees=trees,
                transition_times=transition_times,
                num_states=args.num_states,
                free_edges=free_edges,
                result=initial_result,
                device=device,
                normalize_root_likelihood=args.normalize_root_likelihood,
                fixed_sampling_probability=args.fixed_sampling_probability,
                verbose=args.verbose,
            )
            if not math.isfinite(initial_likelihood):
                raise FloatingPointError(
                    f"Saved parameters for count {count} have a non-finite "
                    "likelihood; refusing to refine"
                )
            print(
                f"REFINEMENT_COUNT_START count={count} "
                f"source={result_path} LL={initial_likelihood:.15f}",
                flush=True,
            )
            result, rounds_completed = refine_transition_schedule(
                trees=trees,
                transition_times=transition_times,
                initial_result=initial_result,
                num_states=args.num_states,
                free_edges=free_edges,
                device=device,
                fixed_sampling_probability=args.fixed_sampling_probability,
                max_time=args.max_time,
                grid_step=args.grid_step,
                max_rounds=args.max_rounds,
                optimizer_name=args.optimizer,
                learning_rate=args.learning_rate,
                max_steps=args.max_steps,
                patience=args.patience,
                normalize_root_likelihood=args.normalize_root_likelihood,
                verbose=args.verbose,
            )
            _write_count_checkpoint(
                args.output_directory,
                count,
                result,
                args.num_states,
            )
            writer.writerow(
                {
                    "transition_count": count,
                    "initial_likelihood": f"{initial_likelihood:.15g}",
                    "refined_likelihood": f"{_result_likelihood(result):.15g}",
                    "rounds_completed": rounds_completed,
                    "transition_times": " ".join(
                        f"{value:.17g}" for value in result.transition_times.detach().cpu().tolist()
                    ),
                }
            )
            history_file.flush()
            print(
                f"REFINEMENT_COUNT_DONE count={count} "
                f"LL={_result_likelihood(result):.15f} rounds={rounds_completed}",
                flush=True,
            )

    print(f"Refined checkpoints written to {args.output_directory}")
    print(f"Summary written to {history_path}")
    print(f"Total execution time: {time.perf_counter() - start:.4f} sec")


if __name__ == "__main__":
    main()
