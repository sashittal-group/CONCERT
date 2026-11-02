from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import torch

from src.dtos import OptimizationResult
from src.main2 import (
    read_free_edges,
    read_initial_parameters,
    read_transition_times,
    write_result,
)
from src.main2_joint_transition_times import (
    _build_datastructures,
    _find_finite_initial_schedule,
    _fixed_parameter_likelihood,
    _load_trees,
)
from src.mle_closed_form2 import optimize_parameters


def _grid_times(max_time: float, grid_size: int) -> list[float]:
    if not math.isfinite(max_time) or max_time <= 0:
        raise ValueError("max_time must be positive and finite")
    if not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("grid_size must be an integer of at least 2")
    return [max_time * index / grid_size for index in range(1, grid_size)]


def _select_initial_transition_times(
    transition_times: list[float],
    count: int,
) -> list[float]:
    if not 1 <= count <= len(transition_times):
        raise ValueError(
            "initial_transition_count must be between 1 and the number of "
            "times in the seed file"
        )
    if count == 1:
        return [transition_times[len(transition_times) // 2]]
    indices = [
        round(index * (len(transition_times) - 1) / (count - 1))
        for index in range(count)
    ]
    return [transition_times[index] for index in indices]


def _load_initial_result(
    result_path: Path,
    num_states: int,
    free_edges: list[tuple[int, int]],
    transition_times: list[float],
) -> OptimizationResult:
    if not result_path.is_file():
        raise FileNotFoundError(f"Initial result file not found: {result_path}")

    lines = result_path.read_text().splitlines()
    likelihood_lines = [line for line in lines if line.startswith("LL=")]
    if len(likelihood_lines) != 1:
        raise ValueError(f"Expected exactly one log likelihood in {result_path}")
    try:
        log_likelihood = float(likelihood_lines[0].split("=", 1)[1])
    except ValueError as error:
        raise ValueError(f"Invalid log likelihood in {result_path}") from error
    if not math.isfinite(log_likelihood):
        raise ValueError(f"Log likelihood in {result_path} must be finite")

    markers = [
        index for index, line in enumerate(lines)
        if line.strip() == "transition_times:"
    ]
    if len(markers) != 1:
        raise ValueError(f"Expected one transition_times section in {result_path}")
    saved_transition_times: list[float] = []
    for line in lines[markers[0] + 1 :]:
        if not line.strip():
            break
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"Invalid transition time in {result_path}: {line!r}")
        try:
            index = int(fields[0])
            transition_time = float(fields[1])
        except ValueError as error:
            raise ValueError(
                f"Invalid transition time in {result_path}: {line!r}"
            ) from error
        if index != len(saved_transition_times) or not math.isfinite(transition_time):
            raise ValueError(f"Invalid transition time in {result_path}: {line!r}")
        saved_transition_times.append(transition_time)
    if saved_transition_times != transition_times:
        raise ValueError(
            f"Transition times in {result_path} do not match the supplied "
            "initial schedule"
        )

    birth_rates, death_rates, sampling_probabilities, transition_probabilities = (
        read_initial_parameters(result_path, num_states, free_edges)
    )
    return OptimizationResult(
        log_likelihood=log_likelihood,
        birth_rates=torch.as_tensor(birth_rates, dtype=torch.float64),
        death_rates=torch.as_tensor(death_rates, dtype=torch.float64),
        sampling_probabilities=torch.as_tensor(
            sampling_probabilities,
            dtype=torch.float64,
        ),
        transition_probabilities=torch.as_tensor(
            transition_probabilities,
            dtype=torch.float64,
        ),
        transition_times=torch.as_tensor(transition_times, dtype=torch.float64),
    )


def _write_count_checkpoint(
    checkpoint_directory: Path,
    transition_count: int,
    result: OptimizationResult,
    num_states: int,
) -> None:
    if result.transition_times is None:
        raise ValueError("Cannot checkpoint a result without transition times")
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    stem = f"transition_count_{transition_count}"
    write_result(
        str(checkpoint_directory / f"{stem}.out"),
        result,
        num_states,
    )
    times = result.transition_times.detach().cpu().tolist()
    (checkpoint_directory / f"{stem}_times.txt").write_text(
        "".join(f"{transition_time:.17g}\n" for transition_time in times)
    )


def _maximize_coordinate(
    transition_times: list[float],
    coordinate: int,
    objective,
    max_time: float,
    grid_size: int,
) -> tuple[list[float], float]:
    if not 0 <= coordinate < len(transition_times):
        raise IndexError("coordinate is outside the transition-time schedule")

    best_times: list[float] | None = None
    best_likelihood = -math.inf
    for candidate in _grid_times(max_time, grid_size):
        lower = transition_times[coordinate - 1] if coordinate else 0.0
        upper = (
            transition_times[coordinate + 1]
            if coordinate + 1 < len(transition_times)
            else max_time
        )
        if not lower < candidate < upper:
            continue

        proposal = list(transition_times)
        proposal[coordinate] = candidate
        likelihood = objective(proposal)
        if math.isfinite(likelihood) and likelihood > best_likelihood:
            best_times = proposal
            best_likelihood = likelihood

    if best_times is None:
        current_likelihood = objective(transition_times)
        if math.isfinite(current_likelihood):
            return list(transition_times), current_likelihood
        raise FloatingPointError(
            f"No finite likelihood found while sweeping transition {coordinate + 1}"
        )
    return best_times, best_likelihood


def _add_transition_at_maximum(
    transition_times: list[float],
    objective,
    max_time: float,
    grid_size: int,
) -> tuple[list[float], float]:
    best_times: list[float] | None = None
    best_likelihood = -math.inf
    for candidate in _grid_times(max_time, grid_size):
        if candidate in transition_times:
            continue
        proposal = sorted([*transition_times, candidate])
        likelihood = objective(proposal)
        if math.isfinite(likelihood) and likelihood > best_likelihood:
            best_times = proposal
            best_likelihood = likelihood

    if best_times is None:
        raise FloatingPointError("No finite likelihood found when adding a transition")
    return best_times, best_likelihood


def _fit_parameters(
    trees,
    transition_times: list[float],
    num_states: int,
    free_edges: list[tuple[int, int]],
    device: torch.device,
    fixed_sampling_probability: float | None,
    result,
    optimizer_name: str,
    learning_rate: float,
    max_steps: int,
    patience: int,
    normalize_root_likelihood: bool,
    logfile: str | None,
    verbose: bool,
):
    datastructures = _build_datastructures(
        trees, transition_times, num_states, verbose=verbose
    )
    initial_parameters = {}
    if result is not None:
        initial_parameters = {
            "initial_birth_rates": result.birth_rates.detach().cpu().tolist(),
            "initial_death_rates": result.death_rates.detach().cpu().tolist(),
            "initial_sampling_probabilities": (
                result.sampling_probabilities.detach().cpu().tolist()
            ),
            "initial_transition_probabilities": (
                result.transition_probabilities.detach().cpu().tolist()
            ),
        }
        if fixed_sampling_probability is not None:
            initial_parameters.pop("initial_sampling_probabilities")

    result = optimize_parameters(
        trees=trees,
        datastructure=datastructures,
        num_states=num_states,
        given_transition_times=transition_times,
        free_edges=free_edges,
        given_sampling_probabilities=(
            [fixed_sampling_probability] * num_states
            if fixed_sampling_probability is not None
            else None
        ),
        logfile=logfile,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        max_steps=max_steps,
        patience=patience,
        device=device,
        normalize_root_likelihood=normalize_root_likelihood,
        **initial_parameters,
    )
    likelihood = float(result.log_likelihood.detach().item())
    if not math.isfinite(likelihood):
        raise FloatingPointError(
            f"Parameter optimization returned a non-finite likelihood "
            f"for transition times {transition_times}"
        )
    return result


def optimize_progressive_joint_transition_times(
    trees,
    num_states: int,
    free_edges: list[tuple[int, int]],
    num_transition_times: int,
    max_time: float,
    device: torch.device,
    fixed_sampling_probability: float | None = None,
    initial_transition_times: list[float] | None = None,
    initial_result: OptimizationResult | None = None,
    optimizer_name: str = "adam",
    learning_rate: float = 0.01,
    max_steps: int = 50000,
    patience: int = 500,
    normalize_root_likelihood: bool = True,
    grid_size: int = 1000,
    random_seed: int = 20261025,
    initialization_attempts: int = 100,
    logfile: str | None = None,
    checkpoint_directory: Path | None = None,
    verbose: bool = False,
):
    if num_transition_times < 1:
        raise ValueError("num_transition_times must be positive")
    if num_transition_times >= grid_size:
        raise ValueError("num_transition_times must be smaller than grid_size")
    if initialization_attempts < 1:
        raise ValueError("initialization_attempts must be at least 1")
    if fixed_sampling_probability is not None and (
        not math.isfinite(fixed_sampling_probability)
        or not 0.0 <= fixed_sampling_probability <= 1.0
    ):
        raise ValueError("fixed_sampling_probability must be finite and in [0, 1]")
    if initial_result is not None and initial_transition_times is None:
        raise ValueError("initial_result requires initial_transition_times")
    if checkpoint_directory is not None:
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
    _grid_times(max_time, grid_size)

    if initial_transition_times is None:
        transition_times = _find_finite_initial_schedule(
            trees=trees,
            num_states=num_states,
            free_edges=free_edges,
            num_transition_times=1,
            max_time=max_time,
            device=device,
            normalize_root_likelihood=normalize_root_likelihood,
            random_seed=random_seed,
            max_attempts=initialization_attempts,
            fixed_sampling_probability=fixed_sampling_probability,
            verbose=verbose,
        )
    else:
        transition_times = sorted(float(time) for time in initial_transition_times)
        if (
            not transition_times
            or len(transition_times) > num_transition_times
            or any(
                not math.isfinite(time) or not 0.0 < time < max_time
                for time in transition_times
            )
            or any(
                current <= previous
                for previous, current in zip(
                    transition_times, transition_times[1:]
                )
            )
        ):
            raise ValueError(
                "initial_transition_times must be unique finite times inside "
                "(0, max_time), with no more entries than num_transition_times"
            )
        initial_likelihood = _fixed_parameter_likelihood(
            trees=trees,
            transition_times=transition_times,
            num_states=num_states,
            free_edges=free_edges,
            result=initial_result,
            device=device,
            normalize_root_likelihood=normalize_root_likelihood,
            fixed_sampling_probability=fixed_sampling_probability,
            verbose=verbose,
        )
        if not math.isfinite(initial_likelihood):
            raise FloatingPointError(
                "The supplied initial transition-time schedule has a "
                "non-finite likelihood."
            )
    print(f"INITIAL_TRANSITION_TIMES={transition_times}", flush=True)

    result = _fit_parameters(
        trees,
        transition_times,
        num_states,
        free_edges,
        device,
        fixed_sampling_probability,
        result=initial_result,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        max_steps=max_steps,
        patience=patience,
        normalize_root_likelihood=normalize_root_likelihood,
        logfile=None,
        verbose=verbose,
    )

    initial_count = len(transition_times)
    for coordinate in range(initial_count):
        def objective(candidate_times: list[float]) -> float:
            return _fixed_parameter_likelihood(
                trees=trees,
                transition_times=candidate_times,
                num_states=num_states,
                free_edges=free_edges,
                result=result,
                device=device,
                fixed_sampling_probability=fixed_sampling_probability,
                normalize_root_likelihood=normalize_root_likelihood,
                verbose=verbose,
            )

        transition_times, likelihood = _maximize_coordinate(
            transition_times,
            coordinate=coordinate,
            objective=objective,
            max_time=max_time,
            grid_size=grid_size,
        )
        print(
            f"TIME_SWEEP count={initial_count} coordinate={coordinate + 1} "
            f"LL={likelihood:.15f} times={transition_times}",
            flush=True,
        )
    result = _fit_parameters(
        trees,
        transition_times,
        num_states,
        free_edges,
        device,
        fixed_sampling_probability,
        result=result,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        max_steps=max_steps,
        patience=patience,
        normalize_root_likelihood=normalize_root_likelihood,
        logfile=(
            logfile
            if num_transition_times == initial_count
            else (
                str(checkpoint_directory / f"transition_count_{initial_count}.log")
                if checkpoint_directory is not None
                else None
            )
        ),
        verbose=verbose,
    )
    if checkpoint_directory is not None:
        _write_count_checkpoint(
            checkpoint_directory,
            initial_count,
            result,
            num_states,
        )

    for count in range(initial_count + 1, num_transition_times + 1):
        def insertion_objective(candidate_times: list[float]) -> float:
            return _fixed_parameter_likelihood(
                trees=trees,
                transition_times=candidate_times,
                num_states=num_states,
                free_edges=free_edges,
                result=result,
                device=device,
                fixed_sampling_probability=fixed_sampling_probability,
                normalize_root_likelihood=normalize_root_likelihood,
                verbose=verbose,
            )

        transition_times, likelihood = _add_transition_at_maximum(
            transition_times,
            objective=insertion_objective,
            max_time=max_time,
            grid_size=grid_size,
        )
        print(
            f"TIME_ADDED count={count} LL={likelihood:.15f} "
            f"times={transition_times}",
            flush=True,
        )
        result = _fit_parameters(
            trees,
            transition_times,
            num_states,
            free_edges,
            device,
            fixed_sampling_probability,
            result=result,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            max_steps=max_steps,
            patience=patience,
            normalize_root_likelihood=normalize_root_likelihood,
            logfile=None,
            verbose=verbose,
        )

        for coordinate in range(count):
            def coordinate_objective(candidate_times: list[float]) -> float:
                return _fixed_parameter_likelihood(
                    trees=trees,
                    transition_times=candidate_times,
                    num_states=num_states,
                    free_edges=free_edges,
                    result=result,
                    device=device,
                    fixed_sampling_probability=fixed_sampling_probability,
                    normalize_root_likelihood=normalize_root_likelihood,
                    verbose=verbose,
                )

            transition_times, likelihood = _maximize_coordinate(
                transition_times,
                coordinate=coordinate,
                objective=coordinate_objective,
                max_time=max_time,
                grid_size=grid_size,
            )
            print(
                f"TIME_SWEEP count={count} coordinate={coordinate + 1} "
                f"LL={likelihood:.15f} times={transition_times}",
                flush=True,
            )

        result = _fit_parameters(
            trees,
            transition_times,
            num_states,
            free_edges,
            device,
            fixed_sampling_probability,
            result=result,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            max_steps=max_steps,
            patience=patience,
            normalize_root_likelihood=normalize_root_likelihood,
            logfile=(
                logfile
                if count == num_transition_times
                else (
                    str(checkpoint_directory / f"transition_count_{count}.log")
                    if checkpoint_directory is not None
                    else None
                )
            ),
            verbose=verbose,
        )
        if checkpoint_directory is not None:
            _write_count_checkpoint(
                checkpoint_directory,
                count,
                result,
                num_states,
            )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate transition times progressively, optimizing model "
            "parameters between grid sweeps."
        )
    )
    parser.add_argument("--tree-directory", type=Path, required=True)
    parser.add_argument("--num-states", type=int, required=True)
    parser.add_argument("--num-trees", type=int, required=True)
    parser.add_argument("--transition-edges-file", type=Path, required=True)
    parser.add_argument("--num-transition-times", type=int, required=True)
    parser.add_argument("--max-time", type=float, default=1.0)
    parser.add_argument("--fixed-sampling-probability", type=float, default=None)
    parser.add_argument("--initial-transition-times-file", type=Path)
    parser.add_argument("--initial-transition-count", type=int)
    parser.add_argument("--initial-result-file", type=Path)
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--grid-size", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=20261025)
    parser.add_argument("--initialization-attempts", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transition-times-output", type=Path)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--optimizer", choices=("adam", "lbfgs"), default="adam")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--normalize-root-likelihood", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    trees = _load_trees(str(args.tree_directory), args.num_trees)
    free_edges = read_free_edges(str(args.transition_edges_file))
    initial_transition_times = (
        read_transition_times(str(args.initial_transition_times_file))
        if args.initial_transition_times_file is not None
        else None
    )
    if args.initial_transition_count is not None:
        if initial_transition_times is None:
            raise ValueError(
                "--initial-transition-count requires "
                "--initial-transition-times-file"
            )
        initial_transition_times = _select_initial_transition_times(
            initial_transition_times,
            args.initial_transition_count,
        )
    if args.initial_result_file is not None and initial_transition_times is None:
        raise ValueError(
            "--initial-result-file requires --initial-transition-times-file"
        )
    initial_result = (
        _load_initial_result(
            args.initial_result_file,
            args.num_states,
            free_edges,
            initial_transition_times,
        )
        if args.initial_result_file is not None
        else None
    )
    args.log.parent.mkdir(parents=True, exist_ok=True)
    result = optimize_progressive_joint_transition_times(
        trees=trees,
        num_states=args.num_states,
        free_edges=free_edges,
        num_transition_times=args.num_transition_times,
        max_time=args.max_time,
        device=torch.device(args.device),
        fixed_sampling_probability=args.fixed_sampling_probability,
        initial_transition_times=initial_transition_times,
        initial_result=initial_result,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        patience=args.patience,
        normalize_root_likelihood=args.normalize_root_likelihood,
        grid_size=args.grid_size,
        random_seed=args.random_seed,
        initialization_attempts=args.initialization_attempts,
        logfile=str(args.log),
        checkpoint_directory=args.checkpoint_directory,
        verbose=args.verbose,
    )
    write_result(str(args.output), result, args.num_states)
    if args.transition_times_output is not None:
        args.transition_times_output.parent.mkdir(parents=True, exist_ok=True)
        transition_times = result.transition_times.detach().cpu().tolist()
        args.transition_times_output.write_text(
            "".join(f"{transition_time:.17g}\n" for transition_time in transition_times)
        )
    print(f"Results written to {args.output}")
    print(f"Total execution time: {time.perf_counter() - start:.4f} sec")


if __name__ == "__main__":
    main()
