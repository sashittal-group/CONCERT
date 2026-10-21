import unittest

import torch

from src.sgootr_single_state import (
    Edge,
    EdgeSegment,
    Node,
    Transition,
    build_partition,
    segment_log_likelihood,
    single_state_u,
    state_log_likelihoods,
)


class SingleStatePartitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.nodes = {
            10: Node(10, 2.0, False, -1),
            0: Node(0, 1.0, False, -1),
            1: Node(1, 0.0, True, 0),
            2: Node(2, 0.0, True, 1),
        }
        self.edges = [
            Edge(10, 0, 1.0),
            Edge(0, 1, 1.0),
            Edge(0, 2, 1.0),
        ]
        self.transitions = [Transition(0, 2, 0, 1, 0.5)]

    def test_known_transition_splits_edge_and_propagates_states(self) -> None:
        partition = build_partition(
            self.nodes,
            self.edges,
            self.transitions,
            num_states=2,
        )
        self.assertEqual(partition.root, 10)
        self.assertEqual(partition.node_states, {10: 0, 0: 0, 1: 0, 2: 1})
        self.assertEqual(partition.births, (1, 0))
        self.assertEqual(partition.leaves, (1, 1))
        self.assertAlmostEqual(partition.exposure[0], 2.5)
        self.assertAlmostEqual(partition.exposure[1], 0.5)

    def test_conflicting_leaf_state_is_rejected(self) -> None:
        conflicting_nodes = dict(self.nodes)
        conflicting_nodes[1] = Node(1, 0.0, True, 1)
        with self.assertRaisesRegex(ValueError, "Conflicting states"):
            build_partition(
                conflicting_nodes,
                self.edges,
                self.transitions,
                num_states=2,
            )

    def test_single_state_likelihood_and_root_u_are_finite(self) -> None:
        partition = build_partition(
            self.nodes,
            self.edges,
            self.transitions,
            num_states=2,
        )
        births = torch.tensor([4.0, 3.0], requires_grad=True)
        deaths = torch.tensor([1.0, 0.5], requires_grad=True)
        values = state_log_likelihoods(partition, births, deaths, 0.05)
        root_u = single_state_u(2.0, births[0], deaths[0], 0.05)
        self.assertEqual(values.shape, (2,))
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(torch.isfinite(root_u))
        (values.sum() - torch.log1p(-root_u)).backward()
        self.assertTrue(torch.isfinite(births.grad).all())
        self.assertTrue(torch.isfinite(deaths.grad).all())

    def test_stable_form_matches_single_state_closed_form(self) -> None:
        birth = torch.tensor(4.0)
        death = torch.tensor(1.0)
        sampling = 0.05
        net_rate = birth - death
        c0 = ((1.0 - sampling) - death / birth) / (-sampling)
        time = 0.6
        expected_u = 1.0 - net_rate / (
            birth * (1.0 - c0 * torch.exp(-net_rate * time))
        )
        self.assertAlmostEqual(
            float(single_state_u(time, birth, death, sampling)),
            float(expected_u),
            places=12,
        )

        segment = EdgeSegment(0, 1, 0, 1.2, 0.4)
        expected_flow = -net_rate * (segment.start - segment.end) + 2.0 * (
            torch.log(1.0 - c0 * torch.exp(-net_rate * segment.end))
            - torch.log(1.0 - c0 * torch.exp(-net_rate * segment.start))
        )
        self.assertAlmostEqual(
            float(segment_log_likelihood(segment, birth, death, sampling)),
            float(expected_flow),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
