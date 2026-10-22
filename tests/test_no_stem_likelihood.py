from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import torch

from src.main2_given_transitions import (
    _validate_no_stem_tree,
    load_per_edge_transition_data,
)
from src.mle2_given_transitions import crown_conditioned_root_log_likelihood


class NoStemLikelihoodTests(unittest.TestCase):
    def test_crown_likelihood_conditions_both_lineages_in_known_root_state(self) -> None:
        root_state_log_likelihoods = torch.log(torch.tensor([0.2, 0.8]))
        root_state_extinction_probabilities = torch.tensor([0.5, 0.25])

        result = crown_conditioned_root_log_likelihood(
            root_state_log_likelihoods,
            root_state_extinction_probabilities,
        )

        self.assertAlmostEqual(float(result), math.log(0.2 / (1.0 - 0.5) ** 2))

    def test_no_stem_requires_a_binary_crown_root(self) -> None:
        tree = nx.DiGraph()
        tree.graph["root"] = 0
        tree.add_edges_from([(0, 1), (0, 2)])
        _validate_no_stem_tree(tree)

        tree.add_edge(0, 3)
        with self.assertRaisesRegex(ValueError, "exactly two daughter edges"):
            _validate_no_stem_tree(tree)

    def test_pipeline_transition_csv_loads_for_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "transition_times.csv"
            path.write_text(
                "parent,child,state_0,state_1,interval_start,interval_end,"
                "transition_time\n"
                "0,1,NC,PT,0.8,1.0,0.9\n"
            )

            transitions = load_per_edge_transition_data(path)

        self.assertEqual(transitions, {(0, 1): [(0, 1, 0.9)]})


if __name__ == "__main__":
    unittest.main()
