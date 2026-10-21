import csv
import tempfile
import unittest
from pathlib import Path

from src.prepare_sgootr_transitions import convert_transitions


class SgootrTransitionPreparationTests(unittest.TestCase):
    def test_mach2_root_sentinel_maps_to_inference_tree_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            nodes_path = directory / "nodes.csv"
            edges_path = directory / "edges.csv"
            transitions_path = directory / "transitions.csv"

            with nodes_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["node_id", "T", "is_leaf", "state_id"])
                writer.writerows(
                    [
                        [10, 1.1, 0, -1],
                        [0, 1.0, 0, -1],
                        [1, 0.0, 1, 2],
                    ]
                )

            with edges_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["parent", "child", "branch_length"])
                writer.writerows([[10, 0, 0.1], [0, 1, 1.0]])

            with transitions_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "parent",
                        "child",
                        "state_0",
                        "state_1",
                        "interval_start",
                        "interval_end",
                        "transition_time",
                    ]
                )
                writer.writerow([0, 1, "NC", "LN", 0.0, 0.1, 0.05])

            transitions, epochs = convert_transitions(
                transitions_path,
                nodes_path,
                edges_path,
            )

        self.assertEqual(
            transitions,
            [{"parent": 10, "child": 0, "s0": 0, "s1": 2, "t": 1.0499}],
        )
        self.assertEqual(epochs, [1.0499])


if __name__ == "__main__":
    unittest.main()
