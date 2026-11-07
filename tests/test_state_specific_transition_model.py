from __future__ import annotations

import unittest

import torch

from src.closed_form_state_specific_transitions import calculate_Us
from src.main2_state_specific_transitions import _read_events
from src.mle_state_specific_transitions import StateSpecificTransitionParameters


class StateSpecificTransitionModelTests(unittest.TestCase):
    def test_linear_four_state_epochs_use_only_the_scheduled_edge(self) -> None:
        parameters = StateSpecificTransitionParameters(
            [0.1, 0.2, 0.3],
            torch.device("cpu"),
            epoch_edges=((0, 1), (1, 2), (2, 3)),
            num_states=4,
            initial_transition_probabilities=(0.2, 0.3, 0.4),
        )

        matrices = parameters.transition_probabilities
        self.assertEqual(tuple(matrices.shape), (3, 4, 4))
        for epoch, (source, target) in enumerate(((0, 1), (1, 2), (2, 3))):
            self.assertAlmostEqual(
                matrices[epoch, source, target].detach().item(),
                (0.2, 0.3, 0.4)[epoch],
                places=7,
            )
            for other_source in range(4):
                if other_source != source:
                    self.assertEqual(
                        matrices[epoch, other_source, other_source].item(),
                        1.0,
                    )
                    self.assertTrue(
                        torch.count_nonzero(matrices[epoch, other_source]).item()
                        == 1
                    )

    def test_epoch_matrices_allow_only_the_assigned_transition(self) -> None:
        parameters = StateSpecificTransitionParameters(
            [0.1, 0.2],
            torch.device("cpu"),
            initial_transition_probabilities=(0.3, 0.4),
        )

        matrices = parameters.transition_probabilities
        expected = torch.tensor(
            [
                [[0.7, 0.3, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 0.6, 0.4], [0.0, 0.0, 1.0]],
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(matrices, expected)
        torch.testing.assert_close(
            matrices.sum(dim=2),
            torch.ones((2, 3), dtype=torch.float64),
        )

    def test_u_recursion_uses_the_matrix_for_each_epoch(self) -> None:
        parameters = StateSpecificTransitionParameters(
            [0.1, 0.2],
            torch.device("cpu"),
            initial_transition_probabilities=(0.2, 0.7),
        )
        birth_rates = torch.tensor([4.0, 3.5, 3.0], dtype=torch.float64)
        death_rates = torch.tensor([0.5, 0.4, 0.3], dtype=torch.float64)
        sampling = torch.tensor([0.2, 0.15, 0.1], dtype=torch.float64)
        times = torch.tensor([0.1, 0.2], dtype=torch.float64)

        epoch_specific = calculate_Us(
            birth_rates,
            death_rates,
            sampling,
            parameters.transition_probabilities,
            times,
            [0.3],
        )
        repeated_first_epoch = calculate_Us(
            birth_rates,
            death_rates,
            sampling,
            parameters.transition_probabilities[:1].expand(2, -1, -1),
            times,
            [0.3],
        )

        self.assertFalse(torch.allclose(epoch_specific.U, repeated_first_epoch.U))
        self.assertEqual(tuple(epoch_specific.Cs.shape), (3, 3))

    def test_epoch_edges_remain_attached_when_times_are_sorted(self) -> None:
        parameters = StateSpecificTransitionParameters(
            [0.1, 0.2],
            torch.device("cpu"),
            epoch_edges=((1, 2), (0, 1)),
        )

        matrices = parameters.transition_probabilities
        self.assertAlmostEqual(matrices[0, 1, 2].detach().item(), 0.25, places=7)
        self.assertEqual(matrices[0, 0, 1].detach().item(), 0.0)
        self.assertAlmostEqual(matrices[1, 0, 1].detach().item(), 0.25, places=7)
        self.assertEqual(matrices[1, 1, 2].detach().item(), 0.0)

    def test_transition_times_must_be_distinct(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            StateSpecificTransitionParameters(
                [0.1, 0.1],
                torch.device("cpu"),
            )

    def test_repeated_pt_to_ln_epochs_share_the_source_probability(self) -> None:
        parameters = StateSpecificTransitionParameters(
            [0.1, 0.2, 0.3],
            torch.device("cpu"),
            epoch_edges=((1, 2), (1, 2), (0, 1)),
            initial_transition_probabilities=(0.2, 0.4),
        )

        matrices = parameters.transition_probabilities
        self.assertAlmostEqual(matrices[0, 1, 2].item(), 0.2, places=7)
        self.assertAlmostEqual(matrices[1, 1, 2].item(), 0.2, places=7)
        self.assertAlmostEqual(matrices[2, 0, 1].item(), 0.4, places=7)
        torch.testing.assert_close(matrices[0], matrices[1])
        self.assertEqual(matrices[0, 0, 1].item(), 0.0)
        self.assertEqual(matrices[1, 0, 1].item(), 0.0)
        self.assertEqual(matrices[2, 1, 2].item(), 0.0)

    def test_three_event_times_sort_with_their_transition_labels(self) -> None:
        times, edges = _read_events(
            nc_to_pt_time=0.3,
            pt_to_ln_time=0.1,
            second_pt_to_ln_time=0.2,
        )

        self.assertEqual(times, [0.1, 0.2, 0.3])
        self.assertEqual(edges, ((1, 2), (1, 2), (0, 1)))


if __name__ == "__main__":
    unittest.main()
