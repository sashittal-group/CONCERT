from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import torch

from src.joint_state_specific_transition_times import (
    _coordinate_candidates,
    _find_finite_initial_schedule,
    _maximize_coordinate,
    optimize_joint_state_specific_transition_times,
)


class JointStateSpecificTransitionTimesTests(unittest.TestCase):
    def test_initial_schedule_is_ordered_and_near_the_leaves(self) -> None:
        edges = ((2, 3), (1, 2), (0, 1))
        with patch(
            "src.joint_state_specific_transition_times._fixed_parameter_likelihood",
            return_value=-12.0,
        ) as likelihood:
            times = _find_finite_initial_schedule(
                trees=[object()],
                num_states=4,
                epoch_edges=edges,
                max_time=1.0,
                grid_size=1000,
                device=torch.device("cpu"),
                random_seed=42,
                max_attempts=1,
                normalize_root_likelihood=True,
            )

        self.assertEqual(times, sorted(times))
        self.assertEqual(len(set(times)), 3)
        self.assertGreater(times[0], 0.0)
        self.assertLessEqual(times[-1], 0.1)
        self.assertTrue(
            all(
                right - left >= 0.01 - 1e-12
                for left, right in zip(times, times[1:])
            )
        )
        self.assertEqual(likelihood.call_args.args[2], edges)

    def test_negative_infinity_retries_with_times_closer_to_the_leaves(self) -> None:
        edges = ((2, 3), (1, 2), (0, 1))
        with patch(
            "src.joint_state_specific_transition_times._fixed_parameter_likelihood",
            side_effect=[-math.inf, -math.inf, -4.0],
        ) as likelihood:
            times = _find_finite_initial_schedule(
                trees=[object()],
                num_states=4,
                epoch_edges=edges,
                max_time=1.0,
                grid_size=1000,
                device=torch.device("cpu"),
                random_seed=42,
                max_attempts=100,
                normalize_root_likelihood=True,
            )

        attempted_times = [
            call.args[1] for call in likelihood.call_args_list
        ]
        self.assertEqual(len(attempted_times), 3)
        self.assertLessEqual(max(attempted_times[1]), 0.05)
        self.assertLessEqual(max(attempted_times[2]), 0.025)
        self.assertLessEqual(max(times), 0.025)

    def test_retries_stop_when_the_near_tip_grid_cannot_fit_all_edges(self) -> None:
        with patch(
            "src.joint_state_specific_transition_times._fixed_parameter_likelihood",
            return_value=-math.inf,
        ) as likelihood:
            with self.assertRaisesRegex(
                FloatingPointError,
                "shrinking the initial transition times toward the leaves",
            ):
                _find_finite_initial_schedule(
                    trees=[object()],
                    num_states=4,
                    epoch_edges=((2, 3), (1, 2), (0, 1)),
                    max_time=1.0,
                    grid_size=1000,
                    device=torch.device("cpu"),
                    random_seed=42,
                    max_attempts=100,
                    normalize_root_likelihood=True,
                )

        self.assertEqual(likelihood.call_count, 3)

    def test_coordinate_candidates_cover_the_full_valid_grid_range(self) -> None:
        times = [0.02, 0.04, 0.06]

        first_candidates = _coordinate_candidates(times, 0, 1.0, 1000)
        middle_candidates = _coordinate_candidates(times, 1, 1.0, 1000)
        last_candidates = _coordinate_candidates(times, 2, 1.0, 1000)

        self.assertEqual((first_candidates[0], first_candidates[-1]), (0.001, 0.03))
        self.assertEqual((middle_candidates[0], middle_candidates[-1]), (0.03, 0.05))
        self.assertEqual((last_candidates[0], last_candidates[-1]), (0.05, 0.999))
        self.assertEqual(len(last_candidates), 950)

    def test_coordinate_sweep_can_select_a_distant_likelihood_maximum(self) -> None:
        with patch(
            "src.joint_state_specific_transition_times._fixed_parameter_likelihood",
            side_effect=lambda trees, times, *args, **kwargs: (
                -abs(times[2] - 0.777)
            ),
        ) as likelihood:
            times, best_likelihood = _maximize_coordinate(
                trees=[object()],
                transition_times=[0.02, 0.04, 0.06],
                coordinate=2,
                epoch_edges=((2, 3), (1, 2), (0, 1)),
                num_states=4,
                max_time=1.0,
                grid_size=1000,
                device=torch.device("cpu"),
                result=None,
                normalize_root_likelihood=True,
            )

        self.assertAlmostEqual(times[2], 0.777)
        self.assertEqual(best_likelihood, 0.0)
        self.assertGreater(likelihood.call_count, 900)

    def test_coordinate_search_uses_six_full_sweeps(self) -> None:
        edges = ((2, 3), (1, 2), (0, 1))
        with (
            patch(
                "src.joint_state_specific_transition_times."
                "_find_finite_initial_schedule",
                return_value=[0.02, 0.04, 0.06],
            ),
            patch(
                "src.joint_state_specific_transition_times._fit_parameters",
                return_value=object(),
            ),
            patch(
                "src.joint_state_specific_transition_times._maximize_coordinate",
                side_effect=lambda trees, times, coordinate, *args: (
                    times,
                    0.0,
                ),
            ) as coordinate_sweep,
        ):
            optimize_joint_state_specific_transition_times(
                trees=[object()],
                num_states=4,
                epoch_edges=edges,
                max_time=1.0,
                device=torch.device("cpu"),
            )

        self.assertEqual(
            [call.args[2] for call in coordinate_sweep.call_args_list],
            [0, 1, 2] * 6,
        )


if __name__ == "__main__":
    unittest.main()
