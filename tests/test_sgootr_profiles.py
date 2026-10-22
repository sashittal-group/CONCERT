from __future__ import annotations

from pathlib import Path
import unittest

import torch

from src.results.sgootr.preprocess import (
    _candidate_parameters,
    build_parameter_profiles,
)
from src.results.tls.config import load_config
from src.results.tls.data import DatasetContext
from src.results.tls.plot import _dataset_offsets


class SgootrProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(
            Path("src/results/sgootr/config.yaml"),
            section="sgootr_profiles",
        )
        self.context = DatasetContext(
            birth_rates=torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64),
            death_rates=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
            sampling_probabilities=torch.tensor(
                [0.4, 0.4, 0.4],
                dtype=torch.float64,
            ),
            transition_probabilities=torch.tensor(
                [[0.7, 0.3, 0.0], [0.0, 0.8, 0.2], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
            transition_times=[0.25, 0.75],
            sensitivity_inputs=[],
        )

    def test_profiles_include_observed_sgootr_transition_edges(self) -> None:
        profiles = build_parameter_profiles(self.context, self.config)
        transition_names = [
            profile.name
            for profile in profiles
            if profile.parameter_type == "transition"
        ]
        self.assertEqual(
            transition_names,
            ["transition[NC->PT]", "transition[PT->LN]"],
        )
        self.assertEqual(
            sum(profile.parameter_type == "transition_ratio" for profile in profiles),
            0,
        )

    def test_transition_profiles_adjust_each_edge_independently(self) -> None:
        profiles = [
            profile
            for profile in build_parameter_profiles(self.context, self.config)
            if profile.parameter_type == "transition"
        ]
        for profile in profiles:
            candidate = profile.original_value * 0.5
            _, _, _, transitions = _candidate_parameters(
                self.context,
                profile,
                candidate,
                self.config,
            )
            source, target = divmod(profile.index, len(self.config.state_names))
            self.assertAlmostEqual(float(transitions[source, target]), candidate)
            self.assertTrue(
                torch.allclose(
                    transitions.sum(dim=1),
                    torch.ones(3, dtype=torch.float64),
                )
            )

    def test_dataset_offsets_center_any_number_of_datasets(self) -> None:
        self.assertEqual(_dataset_offsets(("only",), 0.23), {"only": 0.0})
        self.assertEqual(
            _dataset_offsets(("first", "second"), 0.23),
            {"first": -0.23, "second": 0.23},
        )
        self.assertEqual(
            _dataset_offsets(("first", "middle", "last"), 0.23),
            {"first": -0.23, "middle": 0.0, "last": 0.23},
        )


if __name__ == "__main__":
    unittest.main()
