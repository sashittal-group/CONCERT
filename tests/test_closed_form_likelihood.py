from __future__ import annotations

import unittest

import torch

from src.closed_form_coord2 import calculate_log_likelihood


class ClosedFormLikelihoodTests(unittest.TestCase):
    def test_disconnected_impossible_children_do_not_produce_nan(self) -> None:
        dtype = torch.float64
        empty_edges = torch.empty(0, dtype=dtype)
        child_level = {
            "num_births": torch.zeros(2, dtype=torch.int64),
            "num_sampling_by_state": torch.tensor([[1, 1], [1, 0]]),
            "start_times": empty_edges,
            "end_times": empty_edges,
            "edge_matrix": torch.empty((2, 0), dtype=dtype),
        }
        parent_level = {
            "num_births": torch.zeros(1, dtype=torch.int64),
            "num_sampling_by_state": torch.zeros((1, 2), dtype=torch.int64),
            "start_times": empty_edges,
            "end_times": empty_edges,
            "edge_matrix": torch.empty((1, 0), dtype=dtype),
            "transition_matrix": torch.tensor([[0.0, 1.0]], dtype=dtype),
        }

        likelihood = calculate_log_likelihood(
            datastructure=[parent_level, child_level],
            birth_rates=torch.tensor([1.0, 1.0], dtype=dtype),
            death_rates=torch.tensor([0.1, 0.1], dtype=dtype),
            transition_probabilities=torch.full((2, 2), 0.5, dtype=dtype),
            sampling_probabilities=torch.tensor([0.1, 0.1], dtype=dtype),
            Cs=torch.zeros((2, 2), dtype=dtype),
            transition_times=[0.5],
        )

        self.assertTrue(torch.isfinite(likelihood).all())


if __name__ == "__main__":
    unittest.main()
