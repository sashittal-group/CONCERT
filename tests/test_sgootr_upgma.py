from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from Bio import Phylo

from src.process_data.sgootr.tree import (
    TransitionEvent,
    build_tree_representation,
    export_stage,
)
from src.process_data.sgootr.upgma import (
    build_upgma_tree,
    calculate_pairwise_distances,
    scale_tree_to_height,
    subtract_minimum_pairwise_distance,
)
from src.process_data.sgootr.shift_transition_times import (
    shift_transition_time_files,
)


class SgootrUpgmaTests(unittest.TestCase):
    def test_pairwise_distances_and_upgma_tree(self) -> None:
        tree = Phylo.read(
            StringIO("((gray_a:0.5,blue_a:0.5):0.5,orange_a:1.0);"),
            "newick",
        )

        names, matrix = calculate_pairwise_distances(tree)

        self.assertEqual(names, ("gray_a", "blue_a", "orange_a"))
        self.assertEqual(
            matrix,
            (
                (0.0, 1.0, 2.0),
                (1.0, 0.0, 2.0),
                (2.0, 2.0, 0.0),
            ),
        )
        upgma_tree = build_upgma_tree(names, matrix)
        stage = build_tree_representation(upgma_tree)
        tip_depths = [
            stage.depths[node]
            for node, data in stage.graph.nodes(data=True)
            if data["terminal"]
        ]

        self.assertAlmostEqual(max(tip_depths), min(tip_depths))
        self.assertEqual(
            {tip.name for tip in upgma_tree.get_terminals()},
            set(names),
        )

    def test_upgma_rejects_a_malformed_distance_matrix(self) -> None:
        with self.assertRaisesRegex(ValueError, "square"):
            build_upgma_tree(("gray_a", "blue_a"), ((0.0,),))

    def test_scale_tree_to_unit_height_scales_all_branches(self) -> None:
        tree = Phylo.read(
            StringIO("((gray_a:1,blue_a:1):1,orange_a:2);"),
            "newick",
        )

        scale_factor = scale_tree_to_height(tree)
        stage = build_tree_representation(tree)
        tip_heights = [
            stage.depths[node]
            for node, data in stage.graph.nodes(data=True)
            if data["terminal"]
        ]

        self.assertEqual(scale_factor, 0.5)
        self.assertAlmostEqual(max(tip_heights), 1.0)
        self.assertAlmostEqual(min(tip_heights), 1.0)
        self.assertAlmostEqual(tree.distance("gray_a", "blue_a"), 1.0)

    def test_scale_tree_rejects_zero_height(self) -> None:
        tree = Phylo.read(StringIO("(gray_a:0,blue_a:0);"), "newick")

        with self.assertRaisesRegex(ValueError, "zero height"):
            scale_tree_to_height(tree)

    def test_scaled_transition_times_include_scaled_offsets(self) -> None:
        tree = Phylo.read(
            StringIO("((gray_a:1,blue_a:1):1,orange_a:2);"),
            "newick",
        )
        scale_factor = scale_tree_to_height(tree)
        stage = build_tree_representation(tree)
        transition = TransitionEvent(
            parent=stage.root,
            child=next(iter(stage.graph.successors(stage.root))),
            state_0="NC",
            state_1="PT",
            interval_start=0.0,
            interval_end=0.5,
            transition_time=0.25,
        )

        with TemporaryDirectory() as temporary_directory:
            outputs = export_stage(
                stage,
                Path(temporary_directory),
                [transition],
                transition_time_scale=scale_factor,
            )

            with outputs["transition_times"].open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertAlmostEqual(float(row["interval_start"]), 0.5000005)
            self.assertAlmostEqual(float(row["interval_end"]), 1.0000005)
            self.assertAlmostEqual(float(row["transition_time"]), 0.7499505)
            self.assertAlmostEqual(
                float(outputs["transition_times_text"].read_text()),
                0.7499505,
            )

    def test_minimum_pairwise_distance_is_subtracted_before_upgma(self) -> None:
        names = ("gray_a", "blue_a", "orange_a")
        matrix = (
            (0.0, 1.0, 2.0),
            (1.0, 0.0, 2.0),
            (2.0, 2.0, 0.0),
        )

        adjusted_matrix, minimum_distance = subtract_minimum_pairwise_distance(
            names,
            matrix,
        )
        tree = build_upgma_tree(names, adjusted_matrix)
        stage = build_tree_representation(tree)
        tip_depths = [
            stage.depths[node]
            for node, data in stage.graph.nodes(data=True)
            if data["terminal"]
        ]

        self.assertEqual(minimum_distance, 1.0)
        self.assertEqual(
            adjusted_matrix,
            (
                (0.0, 0.0, 1.0),
                (0.0, 0.0, 1.0),
                (1.0, 1.0, 0.0),
            ),
        )
        self.assertAlmostEqual(max(tip_depths), min(tip_depths))

    def test_transition_times_are_shifted_in_both_input_formats(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_csv = directory / "transition_times.csv"
            input_text = directory / "transition_times.txt"
            with input_csv.open("w", newline="") as handle:
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
                writer.writerow((1, 2, "NC", "PT", 0.2, 0.4, 0.3))
            input_text.write_text("0.3\n0.5\n")

            csv_output, text_output = shift_transition_time_files(
                input_csv,
                input_text,
                directory / "shifted",
                epsilon=1e-4,
            )

            with csv_output.open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["transition_time"], "0.2999")
            self.assertEqual(row["interval_start"], "0.2")
            self.assertEqual(row["interval_end"], "0.4")
            self.assertEqual(text_output.read_text(), "0.2999\n0.4999\n")


if __name__ == "__main__":
    unittest.main()
