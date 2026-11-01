from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
import time
from typing import Callable

import torch

from src.dataprocessor import load_edges, load_nodes
from src.main2 import find_tree_file_pairs, read_free_edges, write_result
from src.mle_closed_form2 import (
    PhylodynamicParameters,
    calculate_total_log_likelihood,
    optimize_parameters,
)
from src.tree_utils import (
    cut_tree,
    get_datastructure_for_likelihood_calculation,
    prepare_nx_tree,
)


def _build_datastructures(
    trees,
    transition_times: list[float],
    num_states: int,
    verbose: bool = False,
):
    datastructures = []
    for tree in trees:
        root_time = float(tree.nodes[tree.graph["root"]]["T"])
        local_times = [time for time in transition_times if time < root_time]
        levels, hierarchy = cut_tree(tree, local_times)
        datastructures.append(
            get_datastructure_for_likelihood_calculation(
                levels,
                hierarchy,
                local_times,
                num_states,
                verbose=verbose,
            )
        )
    return datastructures


def _project_transition_times_to_grid(
    values: list[float],
    max_time: float,
    min_gap: float,
    grid_size: int = 1000,
) -> list[float]:
    if not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("grid_size must be an integer of at least 2")
    last_grid_index = grid_size - 1
    count = len(values)
    if not math.isfinite(max_time) or max_time <= 0:
        raise ValueError("max_time must be positive and finite")
    if not math.isfinite(min_gap) or min_gap <= 0:
        raise ValueError("min_gap must be positive and finite")
    if count == 0 or count > last_grid_index:
        raise ValueError(
            f"Schedule size must be between 1 and {last_grid_index}"
        )
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("Transition times must be finite")

    min_gap_steps = max(1, math.ceil(min_gap * grid_size / max_time))
    if 1 + (count - 1) * min_gap_steps > last_grid_index:
        raise ValueError("Schedule size and minimum gap do not fit the grid")

    indices = sorted(
        min(
            max(round(float(value) * grid_size / max_time), 1),
            last_grid_index,
        )
        for value in values
    )
    for index in range(count):
        lower = 1 + index * min_gap_steps
        upper = last_grid_index - (count - index - 1) * min_gap_steps
        indices[index] = min(max(indices[index], lower), upper)
    for index in range(1, count):
        indices[index] = max(indices[index], indices[index - 1] + min_gap_steps)
    for index in range(count - 2, -1, -1):
        indices[index] = min(indices[index], indices[index + 1] - min_gap_steps)

    return [max_time * index / grid_size for index in indices]


def coordinate_search_transition_times(
    initial_times: list[float],
    max_time: float,
    step_size: float,
    objective: Callable[[list[float]], float],
) -> tuple[list[float], float]:
    """Improve an ordered schedule with likelihood-tested coordinate steps."""
    if not math.isfinite(max_time) or max_time <= 0:
        raise ValueError("max_time must be positive and finite")
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be positive and finite")
    if not initial_times or any(
        not math.isfinite(value) or value <= 0 or value >= max_time
        for value in initial_times
    ):
        raise ValueError("transition times must be inside (0, max_time)")
    if any(right <= left for left, right in zip(initial_times, initial_times[1:])):
        raise ValueError("transition times must be strictly increasing")

    times = list(initial_times)
    best_value = objective(times)
    min_gap = max_time * 1e-8

    for index in range(len(times)):
        lower = min_gap if index == 0 else times[index - 1] + min_gap
        upper = max_time - min_gap if index == len(times) - 1 else times[index + 1] - min_gap
        candidates = {
            times[index],
            max(lower, times[index] - step_size),
            min(upper, times[index] + step_size),
        }
        coordinate_best_time = times[index]
        coordinate_best_value = best_value
        for candidate in sorted(candidates):
            proposal = list(times)
            proposal[index] = candidate
            value = objective(proposal)
            if math.isfinite(value) and value > coordinate_best_value:
                coordinate_best_time = candidate
                coordinate_best_value = value
        times[index] = coordinate_best_time
        best_value = coordinate_best_value

    return times, best_value


def _fixed_parameter_likelihood(
    trees,
    transition_times: list[float],
    num_states: int,
    free_edges: list[tuple[int, int]],
    result,
    device: torch.device,
    normalize_root_likelihood: bool,
    verbose: bool = False,
) -> float:
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
    params = PhylodynamicParameters(
        num_states=num_states,
        free_edges=free_edges,
        given_transition_times=transition_times,
        device=device,
        **initial_parameters,
    ).to(device)
    datastructures = _build_datastructures(
        trees, transition_times, num_states, verbose=verbose
    )
    with torch.no_grad():
        likelihood = calculate_total_log_likelihood(
            params=params,
            trees=trees,
            datastructures=datastructures,
            static_transition_times=transition_times,
            normalize_root_likelihood=normalize_root_likelihood,
        )
    return float(likelihood.item())


def _find_finite_initial_schedule(
    trees,
    num_states: int,
    free_edges: list[tuple[int, int]],
    num_transition_times: int,
    max_time: float,
    device: torch.device,
    normalize_root_likelihood: bool,
    random_seed: int,
    max_attempts: int,
    verbose: bool = False,
) -> list[float]:
    generator = random.Random(random_seed)
    if num_transition_times > 999:
        raise ValueError("At most 999 transition times fit on the grid")
    min_gap = max_time * 1e-8
    initial = _project_transition_times_to_grid(
        [
            max_time * index / (num_transition_times + 1)
            for index in range(1, num_transition_times + 1)
        ],
        max_time,
        min_gap,
    )
    schedules = [initial]
    for _ in range(max_attempts):
        candidate = _project_transition_times_to_grid(
            [
                max_time * index / 1000
                for index in generator.sample(range(1, 1000), num_transition_times)
            ],
            max_time,
            min_gap,
        )
        schedules.append(candidate)

    for candidate in schedules:
        likelihood = _fixed_parameter_likelihood(
            trees=trees,
            transition_times=candidate,
            num_states=num_states,
            free_edges=free_edges,
            result=None,
            device=device,
            normalize_root_likelihood=normalize_root_likelihood,
            verbose=verbose,
        )
        if math.isfinite(likelihood):
            return candidate
    raise FloatingPointError(
        "Could not find finite initial likelihood for any candidate transition "
        f"schedule after {max_attempts} random attempts."
    )


def optimize_joint_transition_times(
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
    normalize_root_likelihood: bool = False,
    search_rounds: int = 3,
    logfile: str | None = None,
    random_seed: int = 20261025,
    initialization_attempts: int = 100,
    verbose: bool = False,
):
    if num_transition_times <= 0:
        raise ValueError("num_transition_times must be positive")
    if not math.isfinite(max_time) or max_time <= 0:
        raise ValueError("max_time must be positive and finite")
    if search_rounds < 1:
        raise ValueError("search_rounds must be at least 1")
    if initialization_attempts < 1:
        raise ValueError("initialization_attempts must be at least 1")

    transition_times = _find_finite_initial_schedule(
        trees=trees,
        num_states=num_states,
        free_edges=free_edges,
        num_transition_times=num_transition_times,
        max_time=max_time,
        device=device,
        normalize_root_likelihood=normalize_root_likelihood,
        random_seed=random_seed,
        max_attempts=initialization_attempts,
        verbose=verbose,
    )
    print(f"INITIAL_TRANSITION_TIMES={transition_times}", flush=True)
    step_size = max_time / (2 * (num_transition_times + 1))
    result = None

    for search_round in range(search_rounds):
        datastructures = _build_datastructures(
            trees, transition_times, num_states, verbose=verbose
        )
        result = optimize_parameters(
            trees=trees,
            datastructure=datastructures,
            num_states=num_states,
            given_transition_times=transition_times,
            free_edges=free_edges,
            logfile=None,
            initial_birth_rates=(
                result.birth_rates.detach().cpu().tolist() if result else None
            ),
            initial_death_rates=(
                result.death_rates.detach().cpu().tolist() if result else None
            ),
            initial_sampling_probabilities=(
                result.sampling_probabilities.detach().cpu().tolist()
                if result
                else None
            ),
            initial_transition_probabilities=(
                result.transition_probabilities.detach().cpu().tolist()
                if result
                else None
            ),
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            max_steps=max_steps,
            patience=patience,
            device=device,
            normalize_root_likelihood=normalize_root_likelihood,
        )

        def objective(candidate_times: list[float]) -> float:
            return _fixed_parameter_likelihood(
                trees,
                candidate_times,
                num_states,
                free_edges,
                result,
                device,
                normalize_root_likelihood,
                verbose,
            )

        transition_times, likelihood = coordinate_search_transition_times(
            transition_times,
            max_time,
            step_size,
            objective,
        )
        print(
            f"TIME_SEARCH round={search_round + 1} "
            f"LL={likelihood:.15f} times={transition_times}",
            flush=True,
        )
        step_size /= 2

    datastructures = _build_datastructures(
        trees, transition_times, num_states, verbose=verbose
    )
    result = optimize_parameters(
        trees=trees,
        datastructure=datastructures,
        num_states=num_states,
        given_transition_times=transition_times,
        free_edges=free_edges,
        logfile=logfile,
        initial_birth_rates=result.birth_rates.detach().cpu().tolist(),
        initial_death_rates=result.death_rates.detach().cpu().tolist(),
        initial_sampling_probabilities=(
            result.sampling_probabilities.detach().cpu().tolist()
        ),
        initial_transition_probabilities=(
            result.transition_probabilities.detach().cpu().tolist()
        ),
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        max_steps=max_steps,
        patience=patience,
        device=device,
        normalize_root_likelihood=normalize_root_likelihood,
    )
    return result


def _load_trees(directory: str, num_trees: int | None):
    trees = []
    for node_path, edge_path in find_tree_file_pairs(directory, num_trees):
        nodes = load_nodes(str(node_path))
        edges = load_edges(str(edge_path))
        if not nodes or not edges:
            raise RuntimeError(f"Empty nodes or edges file: {node_path}, {edge_path}")
        trees.append(prepare_nx_tree(nodes, edges))
    return trees


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jointly estimate rates, transition probabilities, and times."
    )
    parser.add_argument("directory", type=str)
    parser.add_argument("--num-states", type=int, required=True)
    parser.add_argument("--num-trees", type=int, required=True)
    parser.add_argument("--transition-edges-file", type=Path, required=True)
    parser.add_argument("--num-transition-times", type=int, required=True)
    parser.add_argument("--max-time", type=float, default=1.0)
    parser.add_argument("--search-rounds", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=20261025)
    parser.add_argument("--initialization-attempts", type=int, default=100)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed cut-tree likelihood data structures.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--optimizer", choices=("adam", "lbfgs"), default="adam")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--normalize-root-likelihood", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    trees = _load_trees(args.directory, args.num_trees)
    free_edges = read_free_edges(str(args.transition_edges_file))
    result = optimize_joint_transition_times(
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
        search_rounds=args.search_rounds,
        logfile=str(args.log),
        random_seed=args.random_seed,
        initialization_attempts=args.initialization_attempts,
        verbose=args.verbose,
    )
    write_result(str(args.output), result, args.num_states)
    print(f"Results written to {args.output}")
    print(f"Total execution time: {time.perf_counter() - start:.4f} sec")


if __name__ == "__main__":
    main()
