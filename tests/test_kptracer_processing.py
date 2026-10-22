from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import networkx as nx
import pandas as pd

from src.process_data.kptracer.transitions import process_transition_events
from src.process_data.kptracer.tree import (
    apply_location_states,
    build_tree_graph,
    export_processed_tree,
    normalize_tree,
    write_initial_tree_exports,
)


class KpTracerProcessingTests(unittest.TestCase):
    def test_tree_preprocessing_exports_states_and_enforces_edge_floor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            newick_path = root / "tree.nwk"
            metadata_path = root / "metadata.csv"
            newick_path.write_text("(A:0.2,B:0.3);")
            pd.DataFrame(
                {"SubTumor": ["1234_NT_T1", "1234_NT_L1"]},
                index=["A", "B"],
            ).to_csv(metadata_path)

            graph = build_tree_graph(newick_path, metadata_path)
            self.assertEqual(list(graph.nodes), [1, 2, 3])
            self.assertEqual(list(graph.edges), [(1, 2), (1, 3)])
            self.assertEqual(graph.graph["root"], 1)
            self.assertEqual(graph.nodes[2]["state"], 0)
            self.assertEqual(graph.nodes[3]["state"], 1)

            initial_exports = write_initial_tree_exports(graph, root)
            self.assertEqual(
                initial_exports["tree_edges"].read_text().splitlines(),
                ["1\t2", "1\t3"],
            )
            self.assertEqual(
                len(initial_exports["leaf_labels"].read_text().splitlines()),
                2,
            )

            graph = normalize_tree(graph, minimum_edge_length=0.001)
            labels_path = root / "locations.tsv"
            labels_path.write_text("0\t0\n1\t0\n2\t0\n3\t1\n")
            apply_location_states(graph, labels_path)
            nodes_path = root / "nodes.csv"
            edges_path = root / "edges.csv"
            export_processed_tree(graph, nodes_path, edges_path)

            nodes = pd.read_csv(nodes_path)
            edges = pd.read_csv(edges_path)
            self.assertEqual(set(nodes["node_id"]), {1, 2, 3})
            self.assertTrue((edges["branch_length"] >= 0.001 - 1e-12).all())
            self.assertEqual(set(nodes["state_id"]), {0, 1})

    def test_transition_processing_outputs_only_state_change_edges(self) -> None:
        graph = nx.DiGraph()
        graph.graph["root"] = 1
        graph.add_node(1, T=0.7, state=0)
        graph.add_node(2, T=0.2, state=1)
        graph.add_node(3, T=0.0, state=0)
        graph.add_edges_from([(1, 2), (1, 3)])

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            refined_tree_path = root / "refined.tree"
            refined_tree_path.write_text("0\t1\t1\n1\t2\t1\n1\t3\t1\n")

            outputs = process_transition_events(
                graph,
                refined_tree_path,
                root,
                minimum_edge_length=0.001,
            )
            raw = pd.read_csv(outputs["transitions"])
            adjusted = pd.read_csv(outputs["adjusted_transitions"])

            self.assertEqual(len(raw), 2)
            self.assertEqual(
                adjusted[["parent", "child", "state_0", "state_1"]].values.tolist(),
                [[1, 2, 0, 1]],
            )
            self.assertAlmostEqual(float(adjusted["t"].iloc[0]), 0.2)
            self.assertEqual(
                outputs["transition_times"].read_text(),
                "0.2\n",
            )

    def test_transition_times_are_unique_but_per_edge_rows_are_preserved(self) -> None:
        graph = nx.DiGraph()
        graph.graph["root"] = 1
        graph.add_node(1, T=0.7, state=0)
        graph.add_node(2, T=0.2, state=1)
        graph.add_node(3, T=0.2, state=2)
        graph.add_edges_from([(1, 2), (1, 3)])

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            refined_tree_path = root / "refined.tree"
            refined_tree_path.write_text("0\t1\t1\n1\t2\t1\n1\t3\t1\n")

            outputs = process_transition_events(
                graph,
                refined_tree_path,
                root,
                minimum_edge_length=0.001,
            )
            adjusted = pd.read_csv(outputs["adjusted_transitions"])

            self.assertEqual(len(adjusted), 2)
            self.assertEqual(
                outputs["transition_times"].read_text(),
                "0.2\n",
            )


if __name__ == "__main__":
    unittest.main()
