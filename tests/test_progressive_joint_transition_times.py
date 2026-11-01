from __future__ import annotations

import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from src.progressive_joint_transition_times import (
    _add_transition_at_maximum,
    _maximize_coordinate,
    optimize_progressive_joint_transition_times,
)


class ProgressiveJointTransitionTimesTests(unittest.TestCase):
    def test_coordinate_sweep_finds_global_grid_maximum(self) -> None:
        times, likelihood = _maximize_coordinate(
            [0.2, 0.8],
            coordinate=0,
            objective=lambda proposal: -(proposal[0] - 0.37) ** 2,
            max_time=1.0,
            grid_size=1000,
        )

        self.assertEqual(times, [0.37, 0.8])
        self.assertEqual(likelihood, 0.0)

    def test_new_transition_is_added_at_global_maximum(self) -> None:
        times, likelihood = _add_transition_at_maximum(
            [0.6],
            objective=lambda proposal: -(proposal[0] - 0.25) ** 2,
            max_time=1.0,
            grid_size=1000,
        )

        self.assertEqual(times, [0.25, 0.6])
        self.assertEqual(likelihood, 0.0)

    def test_nonfinite_grid_candidates_are_skipped(self) -> None:
        def objective(proposal: list[float]) -> float:
            return float("-inf") if proposal[0] < 0.3 else -abs(proposal[0] - 0.4)

        times, likelihood = _maximize_coordinate(
            [0.5],
            coordinate=0,
            objective=objective,
            max_time=1.0,
            grid_size=1000,
        )

        self.assertEqual(times, [0.4])
        self.assertEqual(likelihood, 0.0)

    def test_progressive_optimizer_fits_parameters_at_each_transition_count(self) -> None:
        fitted_times: list[tuple[float, ...]] = []
        added_schedules = iter(([0.2, 0.6], [0.2, 0.4, 0.6]))
        result = SimpleNamespace(
            log_likelihood=torch.tensor(1.0),
            birth_rates=torch.tensor([1.0]),
            death_rates=torch.tensor([0.1]),
            sampling_probabilities=torch.tensor([0.5]),
            transition_probabilities=torch.tensor([[1.0]]),
        )

        def fake_fit(_trees, times, *_args, **_kwargs):
            fitted_times.append(tuple(times))
            return result

        with (
            patch(
                "src.progressive_joint_transition_times._find_finite_initial_schedule",
                return_value=[0.6],
            ),
            patch(
                "src.progressive_joint_transition_times._fit_parameters",
                side_effect=fake_fit,
            ),
            patch(
                "src.progressive_joint_transition_times._maximize_coordinate",
                side_effect=lambda times, **_kwargs: (list(times), 1.0),
            ),
            patch(
                "src.progressive_joint_transition_times._add_transition_at_maximum",
                side_effect=lambda *_args, **_kwargs: (next(added_schedules), 1.0),
            ),
        ):
            optimize_progressive_joint_transition_times(
                trees=[],
                num_states=1,
                free_edges=[],
                num_transition_times=3,
                max_time=1.0,
                device=torch.device("cpu"),
                grid_size=10,
            )

        self.assertEqual(
            fitted_times,
            [
                (0.6,),
                (0.6,),
                (0.2, 0.6),
                (0.2, 0.6),
                (0.2, 0.4, 0.6),
                (0.2, 0.4, 0.6),
            ],
        )


if __name__ == "__main__":
    unittest.main()
