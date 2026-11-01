from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import torch

from src.main2 import read_free_edges, write_result
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

    result = optimize_parameters(
        trees=trees,
        datastructure=datastructures,
        num_states=num_states,
        given_transition_times=transition_times,
        free_edges=free_edges,
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
    optimizer_name: str = "adam",
    learning_rate: float = 0.01,
    max_steps: int = 50000,
    patience: int = 500,
    normalize_root_likelihood: bool = True,
    grid_size: int = 1000,
    random_seed: int = 20261025,
    initialization_attempts: int = 100,
    logfile: str | None = None,
    verbose: bool = False,
):
    if num_transition_times < 1:
        raise ValueError("num_transition_times must be positive")
    if num_transition_times >= grid_size:
        raise ValueError("num_transition_times must be smaller than grid_size")
    if initialization_attempts < 1:
        raise ValueError("initialization_attempts must be at least 1")
    _grid_times(max_time, grid_size)

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
        verbose=verbose,
    )
    print(f"INITIAL_TRANSITION_TIME={transition_times[0]}", flush=True)

    result = _fit_parameters(
        trees,
        transition_times,
        num_states,
        free_edges,
        device,
        result=None,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        max_steps=max_steps,
        patience=patience,
        normalize_root_likelihood=normalize_root_likelihood,
        logfile=None,
        verbose=verbose,
    )

    def objective(candidate_times: list[float]) -> float:
        return _fixed_parameter_likelihood(
            trees=trees,
            transition_times=candidate_times,
            num_states=num_states,
            free_edges=free_edges,
            result=result,
            device=device,
            normalize_root_likelihood=normalize_root_likelihood,
            verbose=verbose,
        )

    transition_times, likelihood = _maximize_coordinate(
        transition_times,
        coordinate=0,
        objective=objective,
        max_time=max_time,
        grid_size=grid_size,
    )
    print(
        f"TIME_SWEEP count=1 LL={likelihood:.15f} times={transition_times}",
        flush=True,
    )
    result = _fit_parameters(
        trees,
        transition_times,
        num_states,
        free_edges,
        device,
        result=result,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        max_steps=max_steps,
        patience=patience,
        normalize_root_likelihood=normalize_root_likelihood,
        logfile=logfile if num_transition_times == 1 else None,
        verbose=verbose,
    )

    for count in range(2, num_transition_times + 1):
        def insertion_objective(candidate_times: list[float]) -> float:
            return _fixed_parameter_likelihood(
                trees=trees,
                transition_times=candidate_times,
                num_states=num_states,
                free_edges=free_edges,
                result=result,
                device=device,
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
            result=result,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            max_steps=max_steps,
            patience=patience,
            normalize_root_likelihood=normalize_root_likelihood,
            logfile=logfile if count == num_transition_times else None,
            verbose=verbose,
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
    parser.add_argument("--grid-size", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=20261025)
    parser.add_argument("--initialization-attempts", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
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
    args.log.parent.mkdir(parents=True, exist_ok=True)
    result = optimize_progressive_joint_transition_times(
        trees=trees,
        num_states=args.num_states,
        free_edges=free_edges,
        num_transition_times=args.num_transition_times,
        max_time=args.max_time,
        device=torch.device(args.device),
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        patience=args.patience,
        normalize_root_likelihood=args.normalize_root_likelihood,
        grid_size=args.grid_size,
        random_seed=args.random_seed,
        initialization_attempts=args.initialization_attempts,
        logfile=str(args.log),
        verbose=args.verbose,
    )
    write_result(str(args.output), result, args.num_states)
    print(f"Results written to {args.output}")
    print(f"Total execution time: {time.perf_counter() - start:.4f} sec")


if __name__ == "__main__":
    main()
