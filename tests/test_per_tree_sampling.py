from __future__ import annotations

import io
import tempfile
from contextlib import redirect_stdout
from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.dtos import Edge, Node
from src.main2_given_transitions import (
    parse_args,
    read_per_tree_transition_edges,
    write_result,
)
from src.mle2_given_transitions import (
    PhylodynamicParameters,
    calculate_total_log_likelihood,
    inverse_sigmoid,
)
from src.tree_utils import (
    cut_tree,
    get_datastructure_for_likelihood_calculation,
    prepare_nx_tree,
)
from src.results.sgootr.plot_combined_inference import load_parameter_estimates
from src.results.tls.data import load_inference_parameters


class PerTreeSamplingTests(unittest.TestCase):
    def test_runner_flag_is_opt_in(self) -> None:
        required = [
            "trees",
            "--output",
            "result.txt",
            "--log",
            "optimizer.log",
        ]
        with patch("sys.argv", ["main2_given_transitions", *required]):
            self.assertFalse(parse_args().per_tree_sampling_rates)

        with patch(
            "sys.argv",
            [
                "main2_given_transitions",
                *required,
                "--per-tree-sampling-rates",
            ],
        ):
            self.assertTrue(parse_args().per_tree_sampling_rates)

        with patch(
            "sys.argv",
            [
                "main2_given_transitions",
                *required,
                "--per-tree-state-sampling-rates",
            ],
        ):
            self.assertTrue(parse_args().per_tree_state_sampling_rates)

        with patch(
            "sys.argv",
            [
                "main2_given_transitions",
                *required,
                "--per-tree-transition-rates",
            ],
        ):
            self.assertTrue(parse_args().per_tree_transition_rates)

    def _parameters(
        self,
        sampling: list[float] | list[list[float]],
        *,
        per_tree: bool,
        per_tree_state: bool = False,
        per_tree_transition: bool = False,
    ) -> PhylodynamicParameters:
        parameters = PhylodynamicParameters(
            num_states=3,
            free_edges=(
                [(0, 1), (0, 2), (1, 2), (2, 1)]
                if per_tree_transition
                else []
            ),
            given_transition_times=[0.25],
            device=torch.device("cpu"),
            initial_birth_rates=[2.0, 2.0, 2.0],
            initial_death_rates=[0.5, 0.5, 0.5],
            initial_sampling_probabilities=sampling,
            initial_transition_probabilities=(
                [
                    [0.5, 0.25, 0.25],
                    [0.0, 0.5, 0.5],
                    [0.0, 0.5, 0.5],
                ]
                if per_tree_transition
                else torch.eye(3).tolist()
            ),
            num_trees=(
                2
                if per_tree or per_tree_state or per_tree_transition
                else None
            ),
            per_tree_sampling_rates=per_tree,
            per_tree_state_sampling_rates=per_tree_state,
            per_tree_transition_rates=per_tree_transition,
            tree_transition_times=[[0.2], [0.3]] if per_tree_transition else None,
        )
        return parameters

    def _trees_and_datastructures(self):
        nodes = [
            Node(0, 0.5, False, -1),
            Node(1, 0.0, True, 0),
            Node(2, 0.0, True, 1),
        ]
        edges = [Edge(0, 1, 0.5), Edge(0, 2, 0.5)]
        tree = prepare_nx_tree(nodes, edges)
        levels, hierarchy = cut_tree(tree, [0.25])
        data = get_datastructure_for_likelihood_calculation(
            levels,
            hierarchy,
            [0.25],
            3,
        )
        return [tree, tree.copy()], [data, data]

    def test_datastructure_matrices_print_only_when_verbose(self) -> None:
        tree = prepare_nx_tree(
            [
                Node(0, 0.5, False, -1),
                Node(1, 0.0, True, 0),
                Node(2, 0.0, True, 1),
            ],
            [Edge(0, 1, 0.5), Edge(0, 2, 0.5)],
        )
        levels, hierarchy = cut_tree(tree, [0.25])

        captured = io.StringIO()
        with redirect_stdout(captured):
            get_datastructure_for_likelihood_calculation(
                levels,
                hierarchy,
                [0.25],
                3,
            )
        self.assertEqual(captured.getvalue(), "")

        captured = io.StringIO()
        with redirect_stdout(captured):
            get_datastructure_for_likelihood_calculation(
                levels,
                hierarchy,
                [0.25],
                3,
                verbose=True,
            )
        self.assertIn("edge_matrix", captured.getvalue())

    def test_sampling_is_state_shared_across_trees_without_flag(self) -> None:
        parameters = self._parameters([0.3, 0.4, 0.5], per_tree=False)

        self.assertEqual(tuple(parameters.sampling_probabilities.shape), (3,))
        self.assertIsNone(parameters.tree_sampling_probabilities)

    def test_sampling_is_one_rate_per_tree_shared_across_states_with_flag(self) -> None:
        sampling = [0.3, 0.6]
        parameters = self._parameters(sampling, per_tree=True)

        self.assertEqual(tuple(parameters.sampling_probabilities.shape), (2, 3))
        self.assertTrue(
            torch.allclose(
                parameters.tree_sampling_probabilities,
                torch.tensor(
                    sampling,
                    dtype=parameters.sampling_probabilities.dtype,
                ),
            )
        )
        self.assertTrue(
            torch.allclose(
                parameters.sampling_probabilities,
                torch.tensor(
                    [[0.3, 0.3, 0.3], [0.6, 0.6, 0.6]],
                    dtype=parameters.sampling_probabilities.dtype,
                ),
            )
        )

    def test_transition_matrices_are_independent_per_tree(self) -> None:
        parameters = self._parameters(
            [0.4, 0.4, 0.4],
            per_tree=False,
            per_tree_transition=True,
        )
        matrices = parameters.transition_probabilities

        self.assertEqual(tuple(matrices.shape), (2, 3, 3))
        self.assertTrue(
            torch.allclose(
                matrices.sum(dim=2),
                torch.ones((2, 3), dtype=matrices.dtype),
            )
        )
        with torch.no_grad():
            parameters.transition_raw[0, 3] += 1.0
        updated = parameters.transition_probabilities
        self.assertFalse(torch.allclose(updated[0], updated[1]))

        trees, _ = self._trees_and_datastructures()
        datastructures = []
        for tree, transition_time in zip(trees, (0.2, 0.3)):
            levels, hierarchy = cut_tree(tree, [transition_time])
            datastructures.append(
                get_datastructure_for_likelihood_calculation(
                    levels,
                    hierarchy,
                    [transition_time],
                    3,
                )
            )
        likelihood = calculate_total_log_likelihood(
            parameters,
            trees,
            datastructures,
            no_stem=True,
        )
        self.assertTrue(torch.isfinite(likelihood))

    def test_per_tree_transition_graphs_mask_disallowed_edges(self) -> None:
        parameters = PhylodynamicParameters(
            num_states=3,
            free_edges=[(0, 1), (0, 2), (1, 2), (2, 1)],
            given_transition_times=[0.25],
            device=torch.device("cpu"),
            initial_birth_rates=[2.0, 2.0, 2.0],
            initial_death_rates=[0.5, 0.5, 0.5],
            initial_sampling_probabilities=[0.4, 0.5, 0.6],
            initial_transition_probabilities=None,
            num_trees=2,
            per_tree_transition_rates=True,
            tree_transition_times=[[0.2], [0.3]],
            tree_free_edges=[
                [(0, 1), (1, 2)],
                [(0, 2), (1, 2), (2, 1)],
            ],
        )
        matrices = parameters.transition_probabilities

        self.assertTrue(torch.allclose(matrices.sum(dim=2), torch.ones((2, 3))))
        self.assertEqual(matrices[0, 0, 2].item(), 0.0)
        self.assertEqual(matrices[0, 2, 1].item(), 0.0)
        self.assertEqual(matrices[1, 0, 1].item(), 0.0)
        self.assertGreater(matrices[1, 0, 2].item(), 0.0)

    def test_per_tree_transition_graph_file_is_grouped_by_tree_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            graph_path = Path(temporary_directory) / "graphs.txt"
            graph_path.write_text(
                "CRC04:\n0 1\n1 2\n\nCRC10:\n0 1\n2 1\n"
            )
            graphs = read_per_tree_transition_edges(graph_path)

        self.assertEqual(graphs["CRC04"], [(0, 1), (1, 2)])
        self.assertEqual(graphs["CRC10"], [(0, 1), (2, 1)])

    def test_sampling_can_vary_by_tree_and_state_with_state_flag(self) -> None:
        sampling = [[0.3, 0.4, 0.5], [0.6, 0.7, 0.8]]
        parameters = self._parameters(
            sampling,
            per_tree=False,
            per_tree_state=True,
        )

        self.assertEqual(tuple(parameters.sampling_probabilities.shape), (2, 3))
        self.assertTrue(
            torch.allclose(
                parameters.tree_sampling_probabilities,
                torch.tensor(sampling, dtype=torch.float64),
            )
        )
        self.assertTrue(
            torch.allclose(
                parameters.sampling_probabilities,
                torch.tensor(sampling, dtype=torch.float64),
            )
        )

    def test_per_tree_likelihood_reduces_to_shared_likelihood_for_equal_rows(self) -> None:
        trees, datastructures = self._trees_and_datastructures()
        shared = self._parameters([0.4, 0.4, 0.4], per_tree=False)
        per_tree = self._parameters([0.4, 0.4], per_tree=True)
        with torch.no_grad():
            per_tree.sampling_raw.copy_(
                inverse_sigmoid(torch.tensor([0.4, 0.4], dtype=torch.float64))
            )

        shared_ll = calculate_total_log_likelihood(
            shared,
            trees,
            datastructures,
            no_stem=True,
        )
        per_tree_ll = calculate_total_log_likelihood(
            per_tree,
            trees,
            datastructures,
            no_stem=True,
        )

        self.assertTrue(torch.allclose(shared_ll, per_tree_ll, atol=1e-10, rtol=1e-10))

        with torch.no_grad():
            per_tree.sampling_raw[1] = inverse_sigmoid(torch.tensor(0.6))
        distinct_ll = calculate_total_log_likelihood(
            per_tree,
            trees,
            datastructures,
            no_stem=True,
        )
        self.assertNotAlmostEqual(
            float(distinct_ll.detach()),
            float(shared_ll.detach()),
            places=8,
        )

    def test_plot_loader_reads_one_sampling_rate_per_tree(self) -> None:
        content = """LL=1
birth: 2 2 2
death: 1 1 1
sampling: 0.45 0.45 0.45
sampling_scope: per_tree_shared_across_states
sampling_by_tree:
tree_0: 0.3
tree_1: 0.6
transitions:
0->0 1
1->1 1
2->2 1
transition_times:
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "result.txt"
            result_path.write_text(content)

            _, state_names, _, _, tree_sampling, _ = load_parameter_estimates(
                result_path
            )

        self.assertEqual(state_names, ("NC", "PT", "LN"))
        self.assertIsNotNone(tree_sampling)
        self.assertTrue(
            np.allclose(
                tree_sampling,
                [[0.3], [0.6]],
            )
        )

    def test_plot_loader_reads_tree_by_state_sampling_matrix(self) -> None:
        content = """LL=1
birth: 2 2 2
death: 1 1 1
sampling: 0.45 0.55 0.65
sampling_scope: per_tree_by_state
sampling_by_tree:
tree_0: 0.3 0.4 0.5
tree_1: 0.6 0.7 0.8
transitions:
0->0 1
1->1 1
2->2 1
transition_times:
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "result.txt"
            result_path.write_text(content)

            _, _, _, _, tree_sampling, _ = load_parameter_estimates(result_path)

        self.assertIsNotNone(tree_sampling)
        self.assertTrue(
            np.allclose(
                tree_sampling,
                [[0.3, 0.4, 0.5], [0.6, 0.7, 0.8]],
            )
        )

    def test_result_writer_serializes_one_rate_per_tree(self) -> None:
        result = SimpleNamespace(
            log_likelihood=torch.tensor(-1.0),
            birth_rates=torch.tensor([2.0, 2.0, 2.0]),
            death_rates=torch.tensor([1.0, 1.0, 1.0]),
            sampling_probabilities=torch.tensor([0.45, 0.45, 0.45]),
            tree_sampling_probabilities=torch.tensor([0.3, 0.6]),
            transition_probabilities=torch.eye(3),
            transition_times=torch.tensor([0.25]),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "result.txt"
            write_result(str(result_path), result, num_states=3)
            result_text = result_path.read_text()

        self.assertIn("sampling_scope: per_tree_shared_across_states\n", result_text)
        self.assertIn("sampling_by_tree:\ntree_0: 0.3\ntree_1: 0.6\n", result_text)

    def test_result_writer_and_profile_parser_preserve_tree_state_rates(self) -> None:
        result = SimpleNamespace(
            log_likelihood=torch.tensor(-1.0),
            birth_rates=torch.tensor([2.0, 2.0, 2.0]),
            death_rates=torch.tensor([1.0, 1.0, 1.0]),
            sampling_probabilities=torch.tensor([0.45, 0.55, 0.65]),
            tree_sampling_probabilities=torch.tensor(
                [[0.3, 0.4, 0.5], [0.6, 0.7, 0.8]]
            ),
            transition_probabilities=torch.eye(3),
            transition_times=torch.tensor([0.25]),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "result.txt"
            write_result(str(result_path), result, num_states=3)
            parameters = load_inference_parameters(
                result_path,
                ("NC", "PT", "LN"),
            )

        self.assertEqual(parameters.tree_sampling_scope, "per_tree_by_state")
        self.assertTrue(
            torch.allclose(
                parameters.tree_sampling_probabilities,
                torch.tensor([[0.3, 0.4, 0.5], [0.6, 0.7, 0.8]]),
            )
        )

    def test_result_writer_and_parser_preserve_per_tree_transition_matrices_times(
        self,
    ) -> None:
        matrices = torch.tensor(
            [
                [[0.8, 0.2, 0.0], [0.0, 0.9, 0.1], [0.0, 0.1, 0.9]],
                [[0.6, 0.4, 0.0], [0.0, 0.7, 0.3], [0.0, 0.2, 0.8]],
            ],
            dtype=torch.float64,
        )
        result = SimpleNamespace(
            log_likelihood=torch.tensor(-1.0),
            birth_rates=torch.tensor([2.0, 2.0, 2.0]),
            death_rates=torch.tensor([1.0, 1.0, 1.0]),
            sampling_probabilities=torch.tensor([0.45, 0.55, 0.65]),
            tree_sampling_probabilities=None,
            transition_probabilities=matrices.mean(dim=0),
            tree_transition_probabilities=matrices,
            transition_times=torch.tensor([0.2, 0.3]),
            tree_transition_times=[[0.2], [0.3]],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "result.txt"
            write_result(str(result_path), result, num_states=3)
            parameters = load_inference_parameters(
                result_path,
                ("NC", "PT", "LN"),
            )

        self.assertTrue(
            torch.allclose(parameters.tree_transition_probabilities, matrices)
        )
        self.assertEqual(parameters.tree_transition_times, [[0.2], [0.3]])


if __name__ == "__main__":
    unittest.main()
