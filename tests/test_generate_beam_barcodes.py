from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.generate_beam_barcodes import (
    _sample_descendant_states,
    simulate_barcode_matrix,
    write_barcode_matrix,
)


class GenerateBeamBarcodesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.nodes_path = self.root / "nodes.csv"
        self.edges_path = self.root / "edges.csv"
        self.nodes_path.write_text(
            "node_id,T,is_leaf,state_id\n"
            "root,1,0,-1\n"
            "left,0,1,0\n"
            "right,0,1,1\n"
        )
        self.edges_path.write_text(
            "parent,child,branch_length\n"
            "root,left,1\n"
            "root,right,1\n"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_simulation_is_reproducible_and_returns_only_tips(self) -> None:
        first = simulate_barcode_matrix(
            self.nodes_path,
            self.edges_path,
            num_sites=20,
            mutation_rate=1.0,
            num_edit_states=3,
            seed=12,
        )
        second = simulate_barcode_matrix(
            self.nodes_path,
            self.edges_path,
            num_sites=20,
            mutation_rate=1.0,
            num_edit_states=3,
            seed=12,
        )

        self.assertEqual(first, second)
        tip_ids, matrix = first
        self.assertEqual(tip_ids, ("left", "right"))
        self.assertEqual(len(matrix), 2)
        self.assertTrue(all(len(row) == 20 for row in matrix))
        self.assertTrue(all(state in {0, 1, 2, 3} for row in matrix for state in row))

    def test_stem_time_is_simulated_from_an_unedited_origin(self) -> None:
        tip_ids, matrix = simulate_barcode_matrix(
            self.nodes_path,
            self.edges_path,
            num_sites=100,
            mutation_rate=1.0,
            num_edit_states=2,
            origin_time=101.0,
            seed=3,
        )

        self.assertEqual(tip_ids, ("left", "right"))
        self.assertTrue(any(state > 0 for row in matrix for state in row))

    def test_matrix_writer_creates_beam_tsv(self) -> None:
        output_path = self.root / "out" / "barcodes.tsv"
        write_barcode_matrix(output_path, ("tip_1", "tip_2"), [[0, 2], [-1, 1]])

        with output_path.open(newline="") as input_file:
            rows = list(csv.reader(input_file, delimiter="\t"))
        self.assertEqual(
            rows,
            [
                ["", "barcode_1", "barcode_2"],
                ["tip_1", "0", "2"],
                ["tip_2", "-1", "1"],
            ],
        )

    def test_transition_draws_follow_beam_probabilities(self) -> None:
        draws = _sample_descendant_states(
            np.zeros(10_000, dtype=np.int64),
            branch_length=1.0,
            mutation_rate=1.0,
            num_edit_states=2,
            silencing_rate=0.0,
            rng=np.random.default_rng(5),
        )
        edit_fraction = sum(state > 0 for state in draws) / len(draws)
        self.assertAlmostEqual(edit_fraction, 1.0 - 2.718281828459045**-1, delta=0.02)

    def test_invalid_origin_time_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least the root age"):
            simulate_barcode_matrix(
                self.nodes_path,
                self.edges_path,
                num_sites=1,
                mutation_rate=0.1,
                num_edit_states=2,
                origin_time=0.5,
            )


if __name__ == "__main__":
    unittest.main()
