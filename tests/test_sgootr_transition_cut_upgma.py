from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from Bio import Phylo

from src.process_data.sgootr.transition_cut_upgma import (
    run_transition_cut_upgma,
)
from src.process_data.sgootr.tree import build_tree_representation, load_tree


class SgootrTransitionCutUpgmaTests(unittest.TestCase):
    def test_rebuilds_descendant_subtrees_and_preserves_cutoff(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            tree_path = input_dir / "tree.nwk"
            tree_path.write_text(
                "(((gray_a:0.15,blue_a:0.15):0.7,orange_a:0.85):0.15,"
                "gray_b:1.0);"
            )
            source = build_tree_representation(load_tree(tree_path))
            parent = next(
                node
                for node, data in source.graph.nodes(data=True)
                if data["label"] == "" and source.depths[node] == 0.15
            )
            child = next(
                descendant
                for descendant in source.graph.successors(parent)
                if len(
                    source.clade_by_node_id[descendant].get_terminals()
                )
                == 2
            )
            with (input_dir / "transition_times.csv").open(
                "w",
                newline="",
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    (
                        "parent",
                        "child",
                        "state_0",
                        "state_1",
                        "interval_start",
                        "interval_end",
                        "transition_time",
                    )
                )
                writer.writerow((parent, child, "NC", "PT", 0.15, 0.85, 0.25))
            (input_dir / "transition_times.txt").write_text("0.25\n")

            outputs = run_transition_cut_upgma(input_dir, output_dir)

            transformed = build_tree_representation(
                load_tree(outputs["tree"])
            )
            tip_depths = [
                transformed.depths[node]
                for node, data in transformed.graph.nodes(data=True)
                if data["terminal"]
            ]
            self.assertAlmostEqual(transformed.root_age, 1.0)
            self.assertAlmostEqual(min(tip_depths), max(tip_depths))
            self.assertAlmostEqual(min(tip_depths), 1.0)

            with outputs["subtree_distances"].open(newline="") as handle:
                distance_row = next(csv.DictReader(handle))
            self.assertEqual(distance_row["distance"], "0.3")
            self.assertEqual(distance_row["adjusted_distance"], "0")

            with outputs["subtree_summary"].open(newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            rebuilt = next(
                row
                for row in summary_rows
                if row["source_parent"] == str(parent)
                and row["source_child"] == str(child)
            )
            self.assertEqual(rebuilt["tip_count"], "2")
            self.assertEqual(
                rebuilt["adjustment"],
                "extended_zero_height_upgma",
            )

            with outputs["transition_times"].open(newline="") as handle:
                transition = next(csv.DictReader(handle))
            parent_id = int(transition["parent"])
            child_id = int(transition["child"])
            self.assertTrue(transformed.graph.has_edge(parent_id, child_id))
            self.assertAlmostEqual(
                transformed.graph.nodes[child_id]["T"],
                0.25,
            )
            self.assertAlmostEqual(
                float(transition["transition_time"]),
                0.25,
            )


if __name__ == "__main__":
    unittest.main()
