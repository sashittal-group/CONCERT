from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd
import torch

from src.results.kptracer.preprocess import (
    _candidate_parameters,
    build_parameter_profiles,
)
from src.results.tls.config import load_config
from src.results.tls.data import DatasetContext
from src.results.tls.plot import _transition_probability_matrix


class KpTracerProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(
            Path("src/results/kptracer/config.yaml"),
            section="kptracer_profiles",
        )
        self.context = DatasetContext(
            birth_rates=torch.tensor(
                [45.65, 4_938_111_306.98, 32.14, 7.84, 20.22],
                dtype=torch.float64,
            ),
            death_rates=torch.tensor(
                [39.23, 4_938_111_306.15, 21.72, 0.23, 10.21],
                dtype=torch.float64,
            ),
            sampling_probabilities=torch.full(
                (5,),
                0.016943,
                dtype=torch.float64,
            ),
            transition_probabilities=torch.tensor(
                [
                    [0.987505, 0.009901, 0.001199, 0.000784, 0.000611],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.026804, 0.0, 0.973196, 0.0, 0.0],
                    [0.008341, 0.0, 0.0, 0.991659, 0.0],
                    [0.018052, 0.0, 0.000280, 0.0, 0.981668],
                ],
                dtype=torch.float64,
            ),
            transition_times=[],
            sensitivity_inputs=[],
        )

    def test_profiles_cover_all_states_and_inferred_transitions(self) -> None:
        profiles = build_parameter_profiles(self.context, self.config)
        counts = {
            parameter_type: sum(
                profile.parameter_type == parameter_type for profile in profiles
            )
            for parameter_type in {
                profile.parameter_type for profile in profiles
            }
        }

        self.assertEqual(len(self.config.state_names), 5)
        self.assertEqual(counts["birth"], 5)
        self.assertEqual(counts["death"], 5)
        self.assertEqual(counts["growth"], 5)
        self.assertEqual(counts["sampling"], 1)
        self.assertEqual(counts["scaled_sampling"], 5)
        self.assertEqual(counts["transition"], 8)
        self.assertEqual(
            next(
                profile
                for profile in profiles
                if profile.parameter_type == "sampling"
            ).name,
            "sampling[shared]",
        )

    def test_rate_domains_are_local_to_fitted_growth(self) -> None:
        profiles = build_parameter_profiles(self.context, self.config)
        state_one_birth = next(
            profile
            for profile in profiles
            if profile.name == "birth[L1]"
        )
        state_one_death = next(
            profile
            for profile in profiles
            if profile.name == "death[L1]"
        )

        self.assertLess(
            float(state_one_birth.values[-1] - state_one_birth.values[0]),
            10.0,
        )
        self.assertLess(
            float(state_one_death.values[-1] - state_one_death.values[0]),
            10.0,
        )
        self.assertIn(state_one_birth.original_value, state_one_birth.values)
        self.assertIn(state_one_death.original_value, state_one_death.values)
        self.assertGreater(
            float(
                state_one_birth.values[0]
                - self.context.death_rates[1]
            ),
            0.0,
        )
        self.assertGreater(
            float(
                self.context.birth_rates[1]
                - state_one_death.values[-1]
            ),
            0.0,
        )

    def test_shared_sampling_and_transition_candidates_preserve_constraints(self) -> None:
        profiles = build_parameter_profiles(self.context, self.config)
        sampling_profile = next(
            profile
            for profile in profiles
            if profile.parameter_type == "sampling"
        )
        _, _, sampling, _ = _candidate_parameters(
            self.context,
            sampling_profile,
            0.03,
            self.config,
        )
        self.assertTrue(torch.allclose(sampling, torch.full((5,), 0.03)))

        transition_profile = next(
            profile
            for profile in profiles
            if profile.name == "transition[T1->L1]"
        )
        _, _, _, transitions = _candidate_parameters(
            self.context,
            transition_profile,
            transition_profile.original_value / 2.0,
            self.config,
        )
        self.assertTrue(
            torch.allclose(
                transitions.sum(dim=1),
                torch.ones(5, dtype=torch.float64),
                atol=1e-10,
            )
        )

    def test_transition_heatmap_matrix_includes_diagonal_probabilities(self) -> None:
        metadata = pd.DataFrame(
            [
                {
                    "parameter": f"transition[{self.config.state_names[source]}->"
                    f"{self.config.state_names[target]}]",
                    "parameter_type": "transition",
                    "parameter_index": source * len(self.config.state_names) + target,
                    "original_value": float(
                        self.context.transition_probabilities[source, target]
                    ),
                }
                for source in range(len(self.config.state_names))
                for target in range(len(self.config.state_names))
                if source != target
                and self.context.transition_probabilities[source, target] > 0
            ]
        )

        matrix = _transition_probability_matrix(metadata, self.config)

        self.assertTrue(
            torch.allclose(
                torch.as_tensor(matrix).sum(dim=1),
                torch.ones(5, dtype=torch.float64),
                atol=1e-8,
            )
        )
        self.assertAlmostEqual(
            matrix[0, 0],
            1.0 - (0.009901 + 0.001199 + 0.000784 + 0.000611),
        )
        self.assertAlmostEqual(matrix[1, 1], 1.0)
        self.assertEqual(matrix[1, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
