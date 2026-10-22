from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from Bio import Phylo

from src.process_data.sgootr.birth_shift import shift_birth_events
from src.process_data.sgootr.transitions import project_transition_events
from src.process_data.sgootr.tree import (
    TransitionEvent,
    build_tree_representation,
    export_stage,
)


class SgootrDataPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = Phylo.read(
            StringIO("((gray_a:0.5,blue_a:0.5):0.5,orange_a:1.0);"),
            "newick",
        )

    def test_tree_stages_export_without_stem_edges(self) -> None:
        stage = build_tree_representation(self.tree)
        self.assertEqual(stage.graph.number_of_nodes(), 5)
        self.assertEqual(stage.graph.number_of_edges(), 4)
        self.assertEqual(stage.graph.in_degree(stage.root), 0)

        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "input"
            outputs = export_stage(stage, output_dir)
            self.assertTrue(all(path.is_file() for path in outputs.values()))
            self.assertGreater(outputs["plot"].stat().st_size, 0)
            with outputs["nodes"].open(newline="") as handle:
                nodes = list(csv.DictReader(handle))
            with outputs["edges"].open(newline="") as handle:
                edges = list(csv.DictReader(handle))
            self.assertEqual(len(nodes), 5)
            self.assertEqual(len(edges), 4)
            self.assertEqual(outputs["transition_times_text"].read_text(), "")
            exported_tree = Phylo.read(outputs["tree"], "newick")
            self.assertIsNone(exported_tree.root.branch_length)
            self.assertNotIn("stem", {clade.name for clade in exported_tree.find_clades()})

    def test_birth_shift_enforces_floor_without_changing_tip_dates(self) -> None:
        shifted_tree = shift_birth_events(self.tree, 0.025)
        shifted_stage = build_tree_representation(shifted_tree)
        branch_lengths = [
            data["length"]
            for _, _, data in shifted_stage.graph.edges(data=True)
        ]
        self.assertGreaterEqual(min(branch_lengths), 0.025 - 1e-10)
        self.assertAlmostEqual(shifted_stage.root_age, 1.0)
        self.assertEqual(
            shifted_stage.graph.number_of_edges(),
            shifted_stage.graph.number_of_nodes() - 1,
        )

    def test_transition_positions_project_to_shifted_branches(self) -> None:
        source = build_tree_representation(self.tree)
        target = build_tree_representation(shift_birth_events(self.tree, 0.025))
        event = TransitionEvent(
            parent=1,
            child=2,
            state_0="NC",
            state_1="PT",
            interval_start=0.5,
            interval_end=1.0,
            transition_time=0.75,
        )
        projected = project_transition_events(source, target, [event])[0]
        self.assertAlmostEqual(projected.transition_time, 0.9875)
        self.assertAlmostEqual(projected.interval_start, 0.975)
        self.assertAlmostEqual(projected.interval_end, 1.0)

    def test_exported_transition_times_are_inverted_with_epsilon(self) -> None:
        stage = build_tree_representation(self.tree)
        events = [
            TransitionEvent(
                parent=0,
                child=1,
                state_0="NC",
                state_1="PT",
                interval_start=0.0,
                interval_end=0.5,
                transition_time=0.0,
            ),
            TransitionEvent(
                parent=1,
                child=2,
                state_0="PT",
                state_1="LN",
                interval_start=0.5,
                interval_end=1.0,
                transition_time=0.75,
            ),
        ]

        with TemporaryDirectory() as temporary_directory:
            outputs = export_stage(
                stage,
                Path(temporary_directory) / "transitions",
                events,
            )

            self.assertEqual(
                outputs["transition_times_text"].read_text(),
                "0.249901\n0.999901\n",
            )
            with outputs["transition_times"].open(newline="") as handle:
                transitions = list(csv.DictReader(handle))
            self.assertAlmostEqual(
                float(transitions[0]["interval_start"]), 0.500001
            )
            self.assertAlmostEqual(
                float(transitions[0]["interval_end"]), 1.000001
            )
            self.assertAlmostEqual(
                float(transitions[0]["transition_time"]), 0.999901
            )
            self.assertAlmostEqual(
                float(transitions[1]["interval_start"]), 0.000001
            )
            self.assertAlmostEqual(
                float(transitions[1]["interval_end"]), 0.500001
            )
            self.assertAlmostEqual(
                float(transitions[1]["transition_time"]), 0.249901
            )


if __name__ == "__main__":
    unittest.main()
