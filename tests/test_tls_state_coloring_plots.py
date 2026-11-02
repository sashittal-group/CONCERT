from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import pandas as pd

from src.results.cell_state_colors import CELL_STATE_COLORS
from src.results.tls import plot_state_coloring


class TlsStateColoringPlotTests(unittest.TestCase):
    def test_plot_dataset_colors_and_indexes_subtrees(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            subtree_dir = input_dir / "tree_001" / "subtree_001"
            subtree_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {"node_id": 0, "T": 0.1, "is_leaf": 0, "state_id": -1},
                    {"node_id": 1, "T": 0.0, "is_leaf": 1, "state_id": 0},
                    {"node_id": 2, "T": 0.0, "is_leaf": 1, "state_id": 1},
                ]
            ).to_csv(subtree_dir / "nodes.csv", index=False)
            segment_data = root / "segments.csv"
            rows = []
            for segment_id, child, map_state in (
                ("first", "1", "NMPs"),
                ("second", "2", "NeuralTube"),
            ):
                for state in ("NMPs", "Somite", "NeuralTube"):
                    probability = 0.8 if state == map_state else 0.1
                    rows.append(
                        {
                            "subtree": "tree_001/subtree_001",
                            "segment_id": segment_id,
                            "segment_parent_node": "0",
                            "segment_child_node": child,
                            "segment_start_time": 0.1,
                            "segment_end_time": 0.0,
                            "level": 0,
                            "state": state,
                            "state_probability": probability,
                            "most_likely_state": map_state,
                            "most_likely_state_probability": 0.8,
                        }
                    )
            pd.DataFrame(rows).to_csv(segment_data, index=False)
            output_dir = root / "plots"

            created = {}
            original_subplots = plt.subplots

            def capture_subplots(*args, **kwargs):
                result = original_subplots(*args, **kwargs)
                created["axis"] = result[1]
                return result

            with patch.object(
                plot_state_coloring.plt,
                "subplots",
                side_effect=capture_subplots,
            ):
                index_path = plot_state_coloring.plot_dataset(
                    "TLS",
                    segment_data,
                    input_dir,
                    output_dir,
                    ("NMPs", "Somite", "NeuralTube"),
                    dpi=40,
                )

            self.assertTrue(index_path.is_file())
            index = pd.read_csv(index_path)
            self.assertEqual(len(index), 1)
            self.assertEqual(index.iloc[0]["segment_count"], 2)
            self.assertTrue((output_dir / "tree_001/subtree_001.png").is_file())
            axis = created["axis"]
            self.assertEqual(len(axis.lines), 4)
            self.assertEqual(len(axis.collections), 2)
            self.assertEqual(axis.lines[0].get_color(), CELL_STATE_COLORS["NMPs"])
            self.assertEqual(
                axis.lines[2].get_color(), CELL_STATE_COLORS["NeuralTube"]
            )
            leaf_markers = axis.collections[1]
            self.assertEqual(len(leaf_markers.get_offsets()), 2)
            self.assertTrue(
                (
                    leaf_markers.get_sizes()
                    > axis.collections[0].get_sizes()[0]
                ).all()
            )
            self.assertEqual(
                [tuple(color) for color in leaf_markers.get_facecolors()],
                [
                    to_rgba(CELL_STATE_COLORS["NMPs"]),
                    to_rgba(CELL_STATE_COLORS["NeuralTube"]),
                ],
            )
            self.assertEqual(list(axis.lines[0].get_ydata()), [0.0, 0.0])
            self.assertEqual(list(axis.lines[1].get_ydata()), [0.0, -0.1])
            self.assertEqual(list(axis.lines[2].get_ydata()), [0.0, 0.0])
            self.assertEqual(list(axis.lines[3].get_ydata()), [0.0, -0.1])
            self.assertEqual(len(axis.get_yticks()), 21)
            self.assertEqual(
                len(axis.get_yticklabels()),
                len({label.get_text() for label in axis.get_yticklabels()}),
            )


if __name__ == "__main__":
    unittest.main()
