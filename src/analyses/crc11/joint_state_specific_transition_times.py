from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch

from src.concert.infer_parameters import read_free_edges
from src.concert.infer_joint_transition_times import (
    _build_datastructures,
    _load_trees,
)
from src.concert.optimize_state_specific_parameters import (
    StateSpecificTransitionParameters,
    calculate_total_log_likelihood,
    optimize_parameters,
)


INITIAL_WINDOW_FRACTION = 0.1
INITIAL_WINDOW_SHRINK_FACTOR = 0.5
MIN_TRANSITION_GAP = 0.01


def _grid_times(max_time: float, grid_size: int) -> list[float]:
    if not math.isfinite(max_time) or max_time <= 0.0:
        raise ValueError("max_time must be positive and finite")
    if not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("grid_size must be an integer of at least 2")
    return [max_time * index / grid_size for index in range(1, grid_size)]


def _initial_probabilities(
    result,
    epoch_edges: tuple[tuple[int, int], ...],
) -> tuple[float, ...] | None:
    if result is None:
        return None

    previous_matrices = result.transition_probabilities.detach().cpu()
    probabilities_by_source = {}
    for epoch_index, (source, target) in enumerate(epoch_edges):
        if epoch_index >= previous_matrices.shape[0]:
            break
        probabilities_by_source.setdefault(
            source,
            float(previous_matrices[epoch_index, source, target]),
        )
    return tuple(
        probabilities_by_source.get(source, 0.25)
        for source in dict.fromkeys(source for source, _ in epoch_edges)
    )


def _parameter_initial_values(result) -> dict:
    if result is None:
        return {}
    return {
        "initial_birth_rates": result.birth_rates.detach().cpu().tolist(),
        "initial_death_rates": result.death_rates.detach().cpu().tolist(),
        "initial_sampling_probabilities": (
            result.sampling_probabilities.detach().cpu().tolist()
        ),
    }


def _fixed_parameter_likelihood(
    trees,
    transition_times: list[float],
    epoch_edges: tuple[tuple[int, int], ...],
    num_states: int,
    device: torch.device,
    result,
    normalize_root_likelihood: bool,
) -> float:
    params = StateSpecificTransitionParameters(
        transition_times,
        device,
        epoch_edges=epoch_edges,
        num_states=num_states,
        initial_transition_probabilities=_initial_probabilities(
            result,
            epoch_edges,
        ),
        **_parameter_initial_values(result),
    ).to(device)
    datastructures = _build_datastructures(
        trees,
        transition_times,
        num_states,
    )
    with torch.no_grad():
        likelihood = calculate_total_log_likelihood(
            params,
            trees,
            datastructures,
            normalize_root_likelihood=normalize_root_likelihood,
        )
    return float(likelihood.item())


def _fit_parameters(
    trees,
    transition_times: list[float],
    epoch_edges: tuple[tuple[int, int], ...],
    num_states: int,
    device: torch.device,
    result,
    lbfgs_max_iter: int,
    normalize_root_likelihood: bool,
    logfile: str | None = None,
):
    datastructures = _build_datastructures(
        trees,
        transition_times,
        num_states,
    )
    result = optimize_parameters(
        trees,
        datastructures,
        transition_times,
        epoch_edges=epoch_edges,
        num_states=num_states,
        logfile=logfile,
        device=device,
        initial_transition_probabilities=_initial_probabilities(
            result,
            epoch_edges,
        ),
        lbfgs_max_iter=lbfgs_max_iter,
        normalize_root_likelihood=normalize_root_likelihood,
        **_parameter_initial_values(result),
    )
    likelihood = float(result.log_likelihood.detach().item())
    if not math.isfinite(likelihood):
        raise FloatingPointError(
            "Parameter optimization returned a non-finite likelihood "
            f"for transition times {transition_times}"
        )
    return result


def _find_finite_initial_schedule(
    trees,
    num_states: int,
    epoch_edges: tuple[tuple[int, int], ...],
    max_time: float,
    grid_size: int,
    device: torch.device,
    random_seed: int,
    max_attempts: int,
    normalize_root_likelihood: bool,
) -> list[float]:
    if max_attempts < 1:
        raise ValueError("initialization_attempts must be at least one.")
    _grid_times(max_time, grid_size)

    generator = random.Random(random_seed)
    window = max_time * INITIAL_WINDOW_FRACTION
    gap_steps = math.ceil(MIN_TRANSITION_GAP * grid_size / max_time)
    for attempt in range(1, max_attempts + 1):
        last_grid_index = min(
            grid_size - 1,
            math.floor(window * grid_size / max_time),
        )
        if last_grid_index < 1 + (len(epoch_edges) - 1) * gap_steps:
            break

        max_compressed_index = (
            last_grid_index - (len(epoch_edges) - 1) * gap_steps
        )
        compressed_indices = sorted(
            generator.choices(
                range(1, max_compressed_index + 1),
                k=len(epoch_edges),
            )
        )
        indices = [
            compressed_index + index * gap_steps
            for index, compressed_index in enumerate(compressed_indices)
        ]
        transition_times = [
            max_time * index / grid_size for index in indices
        ]
        likelihood = _fixed_parameter_likelihood(
            trees,
            transition_times,
            epoch_edges,
            num_states,
            device,
            result=None,
            normalize_root_likelihood=normalize_root_likelihood,
        )
        if math.isfinite(likelihood):
            return transition_times

        next_window = window * INITIAL_WINDOW_SHRINK_FACTOR
        print(
            f"INITIALIZATION_RETRY attempt={attempt} LL={likelihood} "
            f"times={transition_times} window={window:.15g} "
            f"next_window={next_window:.15g}",
            flush=True,
        )
        window = next_window

    raise FloatingPointError(
        "No finite likelihood found after shrinking the initial transition "
        "times toward the leaves; the near-tip grid no longer fits all "
        f"{len(epoch_edges)} ordered transitions."
    )


def _coordinate_candidates(
    transition_times: list[float],
    coordinate: int,
    max_time: float,
    grid_size: int,
) -> list[float]:
    if not 0 <= coordinate < len(transition_times):
        raise ValueError("coordinate is outside the transition schedule")
    lower = (
        0.0
        if coordinate == 0
        else transition_times[coordinate - 1] + MIN_TRANSITION_GAP
    )
    upper = (
        max_time
        if coordinate + 1 == len(transition_times)
        else transition_times[coordinate + 1] - MIN_TRANSITION_GAP
    )
    tolerance = max_time * 1e-12
    return [
        candidate
        for candidate in _grid_times(max_time, grid_size)
        if lower - tolerance <= candidate <= upper + tolerance
    ]


def _maximize_coordinate(
    trees,
    transition_times: list[float],
    coordinate: int,
    epoch_edges: tuple[tuple[int, int], ...],
    num_states: int,
    max_time: float,
    grid_size: int,
    device: torch.device,
    result,
    normalize_root_likelihood: bool,
) -> tuple[list[float], float]:
    candidates = _coordinate_candidates(
        transition_times,
        coordinate,
        max_time,
        grid_size,
    )
    best_times = list(transition_times)
    best_likelihood = _fixed_parameter_likelihood(
        trees,
        best_times,
        epoch_edges,
        num_states,
        device,
        result,
        normalize_root_likelihood,
    )
    for candidate in candidates:
        if candidate == transition_times[coordinate]:
            continue
        proposal = list(transition_times)
        proposal[coordinate] = candidate
        likelihood = _fixed_parameter_likelihood(
            trees,
            proposal,
            epoch_edges,
            num_states,
            device,
            result,
            normalize_root_likelihood,
        )
        if math.isfinite(likelihood) and likelihood > best_likelihood:
            best_times = proposal
            best_likelihood = likelihood
    if not math.isfinite(best_likelihood):
        raise FloatingPointError(
            f"No finite likelihood found while sweeping transition "
            f"{coordinate + 1} across its valid range."
        )
    return best_times, best_likelihood


def optimize_joint_state_specific_transition_times(
    trees,
    num_states: int,
    epoch_edges: tuple[tuple[int, int], ...],
    max_time: float,
    device: torch.device,
    *,
    grid_size: int = 1000,
    random_seed: int = 20261031,
    initialization_attempts: int = 100,
    search_rounds: int = 6,
    lbfgs_max_iter: int = 20,
    normalize_root_likelihood: bool = True,
    logfile: str | None = None,
):
    if not trees:
        raise ValueError("At least one tree is required.")
    if not epoch_edges:
        raise ValueError("At least one transition edge is required.")
    if any(
        source < 0
        or source >= num_states
        or target < 0
        or target >= num_states
        or source == target
        for source, target in epoch_edges
    ):
        raise ValueError("Transition edges must connect valid distinct states.")
    if search_rounds < 1:
        raise ValueError("search_rounds must be at least one.")
    _grid_times(max_time, grid_size)

    transition_times = _find_finite_initial_schedule(
        trees,
        num_states,
        epoch_edges,
        max_time,
        grid_size,
        device,
        random_seed,
        initialization_attempts,
        normalize_root_likelihood,
    )
    print(f"INITIAL_TRANSITION_TIMES={transition_times}", flush=True)

    result = _fit_parameters(
        trees,
        transition_times,
        epoch_edges,
        num_states,
        device,
        result=None,
        lbfgs_max_iter=lbfgs_max_iter,
        normalize_root_likelihood=normalize_root_likelihood,
    )
    for search_round in range(search_rounds):
        likelihood = -math.inf
        for coordinate in range(len(epoch_edges)):
            transition_times, likelihood = _maximize_coordinate(
                trees,
                transition_times,
                coordinate,
                epoch_edges,
                num_states,
                max_time,
                grid_size,
                device,
                result,
                normalize_root_likelihood,
            )
            print(
                f"TIME_SWEEP round={search_round + 1} "
                f"coordinate={coordinate + 1} "
                f"LL={likelihood:.15f} times={transition_times}",
                flush=True,
            )
        print(
            f"TIME_SEARCH round={search_round + 1} "
            f"LL={likelihood:.15f} times={transition_times}",
            flush=True,
        )
        result = _fit_parameters(
            trees,
            transition_times,
            epoch_edges,
            num_states,
            device,
            result,
            lbfgs_max_iter,
            normalize_root_likelihood,
            logfile=logfile if search_round + 1 == search_rounds else None,
        )
    return result


def _write_result(
    output_path: Path,
    result,
    epoch_edges: tuple[tuple[int, int], ...],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output:
        output.write(f"LL={result.log_likelihood.item():.15g}\n")
        for label, values in (
            ("birth", result.birth_rates),
            ("death", result.death_rates),
            ("sampling", result.sampling_probabilities),
        ):
            output.write(
                f"{label}: "
                + " ".join(
                    f"{value:.15g}"
                    for value in values.detach().cpu().tolist()
                )
                + "\n"
            )
        matrices = result.transition_probabilities.detach().cpu()
        output.write("transition_probabilities_by_epoch:\n")
        for epoch_index, (source, target) in enumerate(epoch_edges):
            output.write(f"epoch_{epoch_index + 1} {source}->{target}\n")
            for state_index, row in enumerate(matrices[epoch_index].tolist()):
                for target_index, probability in enumerate(row):
                    if probability > 0.0:
                        output.write(
                            f"{state_index}->{target_index} "
                            f"{probability:.15g}\n"
                        )
        output.write("transition_times_by_edge:\n")
        for (source, target), transition_time in zip(
            epoch_edges,
            result.transition_times.detach().cpu().tolist(),
        ):
            output.write(f"{source}->{target} {transition_time:.15g}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Jointly estimate model parameters and state-specific transition "
            "times using coordinate search."
        )
    )
    parser.add_argument("--tree-directory", type=Path, required=True)
    parser.add_argument("--num-states", type=int, required=True)
    parser.add_argument("--num-trees", type=int, required=True)
    parser.add_argument("--transition-edges-file", type=Path, required=True)
    parser.add_argument("--max-time", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=20261031)
    parser.add_argument("--initialization-attempts", type=int, default=100)
    parser.add_argument("--search-rounds", type=int, default=6)
    parser.add_argument("--lbfgs-max-iter", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-normalize-root-likelihood", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    trees = _load_trees(str(args.tree_directory), args.num_trees)
    epoch_edges = tuple(read_free_edges(str(args.transition_edges_file)))
    args.log.parent.mkdir(parents=True, exist_ok=True)
    result = optimize_joint_state_specific_transition_times(
        trees=trees,
        num_states=args.num_states,
        epoch_edges=epoch_edges,
        max_time=args.max_time,
        device=torch.device(args.device),
        grid_size=args.grid_size,
        random_seed=args.random_seed,
        initialization_attempts=args.initialization_attempts,
        search_rounds=args.search_rounds,
        lbfgs_max_iter=args.lbfgs_max_iter,
        normalize_root_likelihood=not args.no_normalize_root_likelihood,
        logfile=str(args.log),
    )
    _write_result(args.output, result, epoch_edges)
    print(f"Results written to {args.output}")
    print(f"Total execution time: {time.perf_counter() - start:.4f} sec")


if __name__ == "__main__":
    main()
