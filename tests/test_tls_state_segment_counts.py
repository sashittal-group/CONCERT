from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.results.cell_state_colors import CELL_STATE_ORDER
from src.results.tls.plot_state_segment_counts import (
    _plot_counts,
    _plot_proportions,
    _unique_segments,
    count_segments_by_time,
    make_transition_bin_edges,
    normalize_state_counts,
)


class TlsStateSegmentCountTests(unittest.TestCase):
    def test_leaf_endpoint_roundoff_is_clamped_without_changing_root_time(self) -> None:
        segment = pd.DataFrame(
            [
                {
                    "segment_id": "leaf_edge",
                    "segment_start_time": 0.8,
                    "segment_end_time": 6.661338147750939e-16,
                    "most_likely_state": "NMPs",
                }
            ]
        )

        unique = _unique_segments(segment)

        self.assertEqual(unique.loc[0, "segment_end_time"], 0.0)
        self.assertEqual(unique.loc[0, "segment_start_time"], 0.8)

    def test_state_proportions_sum_to_one_and_plot_as_full_bars(self) -> None:
        counts = pd.DataFrame(
            [
                {
                    "bin_index": 0,
                    "time_start": 0.0,
                    "time_end": 0.01,
                    "NMPs": 1,
                    "Somite": 1,
                    "NeuralTube": 2,
                    "total_segments": 4,
                },
                {
                    "bin_index": 1,
                    "time_start": 0.01,
                    "time_end": 0.02,
                    "NMPs": 3,
                    "Somite": 1,
                    "NeuralTube": 0,
                    "total_segments": 4,
                },
            ]
        )

        proportions = normalize_state_counts(counts)

        self.assertTrue(
            np.allclose(proportions[list(CELL_STATE_ORDER)].sum(axis=1), 1.0)
        )
        self.assertEqual(proportions.loc[0, "NeuralTube"], 0.5)
        self.assertEqual(proportions.loc[1, "NMPs"], 0.75)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "proportions.png"
            _plot_proportions(
                "TLS",
                proportions,
                CELL_STATE_ORDER,
                output_path,
                dpi=30,
            )
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)

        empty_counts = counts.copy()
        empty_counts.loc[0, list(CELL_STATE_ORDER)] = 0
        empty_counts.loc[0, "total_segments"] = 0
        with self.assertRaises(ValueError):
            normalize_state_counts(empty_counts)

    def test_transition_times_define_101_intervals_without_boundary_leaks(self) -> None:
        transition_times = [0.0001, *[index / 100 for index in range(1, 100)]]
        bin_edges = make_transition_bin_edges(
            transition_times,
            0.0,
            1.0000000000000002,
        )
        self.assertEqual(len(bin_edges) - 1, 101)
        self.assertEqual(bin_edges[0], 0.0)
        self.assertEqual(bin_edges[1], 0.0001)
        self.assertEqual(bin_edges[2], 0.01)
        self.assertEqual(bin_edges[-1], 1.0)

        segment_rows = [
            {
                "segment_id": "at_transition_boundary",
                "segment_start_time": 0.02,
                "segment_end_time": 0.01,
                "most_likely_state": "Somite",
            }
            for _ in CELL_STATE_ORDER
        ]
        segments = pd.DataFrame(segment_rows)
        counts = count_segments_by_time(
            segments,
            bin_edges,
        )
        self.assertEqual(counts.loc[1, "Somite"], 0)
        self.assertEqual(counts.loc[2, "Somite"], 1)
        self.assertEqual(counts.loc[3, "Somite"], 0)

        imprecise_equal_edges = np.linspace(0.0, 1.0000000000000002, 101)
        imprecise_counts = count_segments_by_time(segments, imprecise_equal_edges)
        self.assertEqual(imprecise_counts.loc[0, "Somite"], 0)
        self.assertEqual(imprecise_counts.loc[1, "Somite"], 1)
        self.assertEqual(imprecise_counts.loc[2, "Somite"], 0)

    def test_counts_long_format_segments_in_each_overlapping_bin(self) -> None:
        segment_specs = (
            ("spanning", 0.8, 0.3, "NMPs"),
            ("lower_half", 0.5, 0.0, "Somite"),
            ("upper_half", 1.0, 0.5, "NeuralTube"),
        )
        rows = []
        for segment_id, start, end, map_state in segment_specs:
            for state in CELL_STATE_ORDER:
                rows.append(
                    {
                        "segment_id": segment_id,
                        "segment_start_time": start,
                        "segment_end_time": end,
                        "most_likely_state": map_state,
                        "state": state,
                    }
                )
        bin_edges = np.linspace(0.0, 1.0, 101)

        counts = count_segments_by_time(pd.DataFrame(rows), bin_edges)

        self.assertEqual(len(counts), 100)
        self.assertEqual(int(counts["NMPs"].sum()), 50)
        self.assertEqual(int(counts["Somite"].sum()), 50)
        self.assertEqual(int(counts["NeuralTube"].sum()), 50)
        self.assertEqual(counts.loc[49, "Somite"], 1)
        self.assertEqual(counts.loc[50, "Somite"], 0)
        self.assertEqual(counts.loc[49, "NeuralTube"], 0)
        self.assertEqual(counts.loc[50, "NeuralTube"], 1)
        self.assertEqual(counts.loc[49, "NMPs"], 1)
        self.assertEqual(counts.loc[50, "NMPs"], 1)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "counts.png"
            _plot_counts("TLS", counts, CELL_STATE_ORDER, output_path, dpi=30)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
