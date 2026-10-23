from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.main2 import read_initial_parameters


class Main2InitializationTests(unittest.TestCase):
    def test_loads_parameters_and_derives_transition_diagonals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "inference.tmp.out"
            path.write_text(
                "ITER 120 LL=-10\n"
                "birth: 5 6 7\n"
                "death: 1 2 3\n"
                "sampling: 0.1 0.1 0.1\n"
                "transitions:\n"
                "0->1 0.3\n"
                "0->2 0.2\n"
                "transition_times:\n"
                "0 0.0001\n"
            )

            birth, death, sampling, transitions = read_initial_parameters(
                path,
                num_states=3,
                free_edges=[(0, 1), (0, 2)],
            )

        self.assertEqual(birth, [5.0, 6.0, 7.0])
        self.assertEqual(death, [1.0, 2.0, 3.0])
        self.assertEqual(sampling, [0.1, 0.1, 0.1])
        self.assertEqual(
            transitions,
            [
                [0.5, 0.3, 0.2],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        )

    def test_rejects_initial_transition_edges_that_do_not_match_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "inference.tmp.out"
            path.write_text(
                "birth: 5 6 7\n"
                "death: 1 2 3\n"
                "sampling: 0.1 0.1 0.1\n"
                "transitions:\n"
                "0->1 0.3\n"
                "transition_times:\n"
                "0 0.0001\n"
            )

            with self.assertRaisesRegex(ValueError, "do not match"):
                read_initial_parameters(
                    path,
                    num_states=3,
                    free_edges=[(0, 1), (0, 2)],
                )


if __name__ == "__main__":
    unittest.main()
