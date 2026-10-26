from __future__ import annotations

import unittest

import torch

from src.closed_form_given_transitions import calculate_Us
from src.results.tls.process_state_coloring import (
    _component_state_posteriors,
)


class TlsStateColoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.birth = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        self.death = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
        self.sampling = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        self.transitions = torch.tensor(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ],
            dtype=torch.float64,
        )
        self.times = [0.5]
        self.u_result = calculate_Us(
            self.birth,
            self.death,
            self.sampling,
            self.transitions,
            torch.tensor(self.times, dtype=torch.float64),
            root_times=[1.0],
        )

    def test_leaf_label_determines_single_component_state(self) -> None:
        datastructure = [
            {
                "num_births": torch.tensor([0]),
                "num_sampling_by_state": torch.tensor([[1, 0, 0]]),
                "start_times": torch.tensor([1.0]),
                "end_times": torch.tensor([0.0]),
                "edge_matrix": torch.tensor([[1.0]]),
            }
        ]

        posteriors, log_likelihood = _component_state_posteriors(
            datastructure,
            self.birth,
            self.death,
            self.sampling,
            self.transitions,
            self.u_result.Cs,
            self.times,
        )

        self.assertEqual(posteriors[0].shape, (1, 3))
        self.assertTrue(
            torch.allclose(
                posteriors[0],
                torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
            )
        )
        self.assertTrue(torch.isfinite(torch.tensor(log_likelihood)))

    def test_child_component_posterior_follows_transition_likelihood(self) -> None:
        empty_component = {
            "num_births": torch.tensor([0]),
            "num_sampling_by_state": torch.tensor([[0, 0, 0]]),
            "start_times": torch.empty(0, dtype=torch.float64),
            "end_times": torch.empty(0, dtype=torch.float64),
            "edge_matrix": torch.empty((1, 0), dtype=torch.float64),
        }
        datastructure = [
            {
                **empty_component,
                "transition_matrix": torch.tensor([[1.0]], dtype=torch.float64),
            },
            empty_component,
        ]

        posteriors, _ = _component_state_posteriors(
            datastructure,
            self.birth,
            self.death,
            self.sampling,
            self.transitions,
            self.u_result.Cs,
            self.times,
        )

        self.assertTrue(
            torch.allclose(
                posteriors[0],
                torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
            )
        )
        self.assertTrue(
            torch.allclose(
                posteriors[1],
                self.transitions[:1],
            )
        )


if __name__ == "__main__":
    unittest.main()
