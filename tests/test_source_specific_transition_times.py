from __future__ import annotations

import unittest

import torch

from src.closed_form_coord2 import calculate_Us, calculate_log_likelihood
from src.dtos import Edge, Node
from src.results.kptracer.infer_source_specific_transition_times import (
    SOURCE_STATES,
    _initial_time_starts,
)
from src.tree_utils import (
    cut_tree,
    get_datastructure_for_likelihood_calculation,
    prepare_nx_tree,
)


class SourceSpecificTransitionTimeTests(unittest.TestCase):
    def test_initial_starts_rotate_seed_times_across_sources(self) -> None:
        starts = _initial_time_starts(
            [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            num_starts=4,
        )
        self.assertEqual(len(starts), len(SOURCE_STATES))
        self.assertEqual(
            set(starts[0].values()),
            {0.01, 0.2, 0.5, 0.7},
        )
        self.assertEqual(
            starts[1][SOURCE_STATES[0]],
            starts[0][SOURCE_STATES[1]],
        )

    def test_only_active_source_row_transitions_at_an_epoch(self) -> None:
        birth = torch.tensor([2.0, 2.0], dtype=torch.float64)
        death = torch.tensor([1.0, 1.0], dtype=torch.float64)
        sampling = torch.tensor([0.1, 0.1], dtype=torch.float64)
        transitions = torch.tensor(
            [[0.8, 0.2], [0.4, 0.6]],
            dtype=torch.float64,
        )
        times = torch.tensor([0.5], dtype=torch.float64)
        result = calculate_Us(
            birth,
            death,
            sampling,
            transitions,
            times,
            root_times=[1.0],
            transition_sources_by_time=[[0]],
        )

        torch.testing.assert_close(
            result.U[1, 1],
            result.flow_U[1, 1],
        )
        torch.testing.assert_close(
            result.U[1, 0],
            transitions[0] @ result.flow_U[1],
        )

    def test_source_specific_transition_likelihood_is_finite(self) -> None:
        tree = prepare_nx_tree(
            [Node(0, 1.0, False, -1), Node(1, 0.0, True, 1)],
            [Edge(0, 1, 1.0)],
        )
        levels, hierarchy = cut_tree(tree, [0.5])
        datastructure = get_datastructure_for_likelihood_calculation(
            levels,
            hierarchy,
            [0.5],
            2,
        )
        birth = torch.tensor([2.0, 2.0], dtype=torch.float64)
        death = torch.tensor([1.0, 1.0], dtype=torch.float64)
        sampling = torch.tensor([0.1, 0.1], dtype=torch.float64)
        transitions = torch.tensor(
            [[0.8, 0.2], [0.4, 0.6]],
            dtype=torch.float64,
        )
        times = torch.tensor([0.5], dtype=torch.float64)
        u_result = calculate_Us(
            birth,
            death,
            sampling,
            transitions,
            times,
            root_times=[1.0],
            transition_sources_by_time=[[0]],
        )

        likelihood = calculate_log_likelihood(
            datastructure,
            birth,
            death,
            transitions,
            sampling,
            u_result.Cs,
            [0.5],
            transition_sources_by_time=[[0]],
        )
        self.assertTrue(torch.isfinite(likelihood[0, 0]))

        alternate_transitions = torch.tensor(
            [[0.8, 0.2], [0.99, 0.01]],
            dtype=torch.float64,
        )
        alternate_u_result = calculate_Us(
            birth,
            death,
            sampling,
            alternate_transitions,
            times,
            root_times=[1.0],
            transition_sources_by_time=[[0]],
        )
        alternate_likelihood = calculate_log_likelihood(
            datastructure,
            birth,
            death,
            alternate_transitions,
            sampling,
            alternate_u_result.Cs,
            [0.5],
            transition_sources_by_time=[[0]],
        )
        torch.testing.assert_close(likelihood, alternate_likelihood)


if __name__ == "__main__":
    unittest.main()
