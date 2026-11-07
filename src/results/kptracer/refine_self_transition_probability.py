from __future__ import annotations

from pathlib import Path
import time

import torch

from src.main2 import read_free_edges, write_result
from src.main2_joint_transition_times import _load_trees
from src.progressive_joint_transition_times import (
    _load_initial_result,
    optimize_progressive_joint_transition_times,
)
from src.results.tls.data import load_inference_parameters

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
STATE_NAMES = ("T1", "L1", "L2", "L3", "S1")
TREE_DIRECTORY = REPOSITORY_ROOT / "data/KPTracer-Data/processed_data"
TRANSITION_EDGES_PATH = REPOSITORY_ROOT / "src/transition_edges_kptracer.txt"
INITIAL_RESULT_PATH = (
    TREE_DIRECTORY
    / "inference_progressive_fixed_sampling_0.05_5_20_regenerated_grid_100"
    / "transition_count_11.out"
)
RUN_DIRECTORY = (
    TREE_DIRECTORY / "inference_self_transition_probability_min_0.90_n11"
)
MINIMUM_SELF_TRANSITION_PROBABILITY = 0.9
FIXED_SAMPLING_PROBABILITY = 0.05
JOINT_REFINEMENT_ROUNDS = 2
MAX_STEPS = 50000
PATIENCE = 500
GRID_SIZE = 100


def main() -> None:
    RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    free_edges = read_free_edges(str(TRANSITION_EDGES_PATH))
    seed_parameters = load_inference_parameters(
        INITIAL_RESULT_PATH,
        STATE_NAMES,
    )
    transition_times = [
        float(value) for value in seed_parameters.transition_times
    ]
    result = _load_initial_result(
        INITIAL_RESULT_PATH,
        num_states=len(STATE_NAMES),
        free_edges=free_edges,
        transition_times=transition_times,
    )
    initial_likelihood = float(result.log_likelihood)
    trees = _load_trees(str(TREE_DIRECTORY), num_trees=1)
    device = torch.device("cpu")

    start_time = time.perf_counter()
    for round_index in range(1, JOINT_REFINEMENT_ROUNDS + 1):
        result = optimize_progressive_joint_transition_times(
            trees=trees,
            num_states=len(STATE_NAMES),
            free_edges=free_edges,
            num_transition_times=len(transition_times),
            max_time=1.0,
            device=device,
            fixed_sampling_probability=FIXED_SAMPLING_PROBABILITY,
            initial_transition_times=transition_times,
            initial_result=result,
            max_steps=MAX_STEPS,
            patience=PATIENCE,
            normalize_root_likelihood=True,
            grid_size=GRID_SIZE,
            logfile=str(RUN_DIRECTORY / f"round_{round_index}.log"),
            minimum_self_transition_probability=(
                MINIMUM_SELF_TRANSITION_PROBABILITY
            ),
        )
        if result.transition_times is None:
            raise RuntimeError(
                "Joint refinement returned no transition-time schedule"
            )
        transition_times = result.transition_times.detach().cpu().tolist()
        round_path = RUN_DIRECTORY / f"round_{round_index}.out"
        write_result(str(round_path), result, len(STATE_NAMES))
        diagonal = result.transition_probabilities.detach().cpu().diagonal()
        if not torch.all(
            diagonal > MINIMUM_SELF_TRANSITION_PROBABILITY
        ):
            raise RuntimeError(
                "Fitted self-transition probabilities do not satisfy the "
                "strict 0.90 lower bound"
            )
        likelihood = float(result.log_likelihood.detach().cpu())
        print(
            f"ROUND={round_index} LL={likelihood:.12f} "
            f"self_transition_probabilities={diagonal.tolist()} "
            f"transition_times={transition_times}",
            flush=True,
        )

    final_path = RUN_DIRECTORY / "transition_count_11.out"
    write_result(str(final_path), result, len(STATE_NAMES))
    print(f"Seed log likelihood: {initial_likelihood:.12f}")
    print(f"Final log likelihood: {float(result.log_likelihood):.12f}")
    print(f"Result written to {final_path}")
    print(f"Total execution time: {time.perf_counter() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
