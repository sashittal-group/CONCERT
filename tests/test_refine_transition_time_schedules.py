from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.refine_transition_time_schedules import (
    _fine_grid_candidates,
    refine_transition_schedule,
)


class RefineTransitionTimeSchedulesTests(unittest.TestCase):
    def test_fine_grid_candidates_stay_between_fixed_neighbors(self) -> None:
        candidates = _fine_grid_candidates(
            [0.2, 0.8],
            coordinate=0,
            max_time=1.0,
            grid_step=0.1,
        )

        self.assertEqual(len(candidates), 7)
        self.assertAlmostEqual(candidates[0], 0.1)
        self.assertAlmostEqual(candidates[-1], 0.7)
        self.assertTrue(all(0.0 < candidate < 0.8 for candidate in candidates))

    def test_refinement_refits_parameters_after_each_coordinate(self) -> None:
        result = SimpleNamespace(
            log_likelihood=torch.tensor(-0.02),
            birth_rates=torch.tensor([1.0]),
            death_rates=torch.tensor([0.1]),
            sampling_probabilities=torch.tensor([0.05]),
            transition_probabilities=torch.tensor([[1.0]]),
            transition_times=torch.tensor([0.2, 0.8]),
        )
        fitted_schedules: list[list[float]] = []

        def fake_likelihood(*, transition_times, **_kwargs):
            return -(
                (transition_times[0] - 0.3) ** 2
                + (transition_times[1] - 0.7) ** 2
            )

        def fake_fit(*, transition_times, **_kwargs):
            fitted_schedules.append(list(transition_times))
            result.transition_times = torch.tensor(transition_times)
            result.log_likelihood = torch.tensor(fake_likelihood(
                transition_times=transition_times
            ))
            return result

        with (
            patch(
                "src.refine_transition_time_schedules._fixed_parameter_likelihood",
                side_effect=fake_likelihood,
            ),
            patch(
                "src.refine_transition_time_schedules._fit_parameters",
                side_effect=fake_fit,
            ),
        ):
            refined, rounds_completed = refine_transition_schedule(
                trees=[],
                transition_times=[0.2, 0.8],
                initial_result=result,
                num_states=1,
                free_edges=[],
                device=torch.device("cpu"),
                fixed_sampling_probability=0.05,
                max_time=1.0,
                grid_step=0.1,
                max_rounds=1,
            )

        self.assertEqual(len(fitted_schedules), 2)
        self.assertAlmostEqual(fitted_schedules[0][0], 0.3)
        self.assertAlmostEqual(fitted_schedules[0][1], 0.8)
        self.assertAlmostEqual(fitted_schedules[1][0], 0.3)
        self.assertAlmostEqual(fitted_schedules[1][1], 0.7)
        self.assertAlmostEqual(refined.transition_times[0].item(), 0.3)
        self.assertAlmostEqual(refined.transition_times[1].item(), 0.7)
        self.assertEqual(rounds_completed, 1)


if __name__ == "__main__":
    unittest.main()
