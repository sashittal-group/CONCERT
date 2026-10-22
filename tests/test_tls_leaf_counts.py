from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.results.tls.config import load_config
from src.results.tls.leaf_counts import build_leaf_count_rows
from src.results.tls.plot import (
    _render_population_size_profile,
    _render_sampled_population_size_plot,
)


class FakeLeaf:
    def __init__(self, state: str | None) -> None:
        self.state = state


class FakeTree:
    def __init__(self, states: list[str | None]) -> None:
        self.leaves = [FakeLeaf(state) for state in states]

    def iter_leaves(self):
        return iter(self.leaves)


class TlsLeafCountTests(unittest.TestCase):
    def test_build_leaf_count_rows_includes_state_and_labeled_totals(self) -> None:
        rows, states = build_leaf_count_rows(
            {
                "TLS": [FakeTree(["NMPs", "Somite", None])],
                "TLSC": [FakeTree(["NeuralTube", "Other"])],
            },
            ("TLS", "TLSC"),
            ("NMPs", "Somite", "NeuralTube"),
        )

        self.assertEqual(states, ["NMPs", "Somite", "NeuralTube", "Other"])
        self.assertEqual(
            rows[0],
            {
                "dataset": "TLS",
                "tree_name": "tree_001",
                "NMPs": 1,
                "Somite": 1,
                "NeuralTube": 0,
                "Other": 0,
                "unlabeled_leaf_count": 1,
                "total_labeled_leaves": 2,
            },
        )
        tls_total = next(
            row for row in rows if row["dataset"] == "TLS" and row["tree_name"] == "TOTAL"
        )
        self.assertEqual(tls_total["total_labeled_leaves"], 2)
        self.assertEqual(tls_total["unlabeled_leaf_count"], 1)

    def test_population_plots_write_configured_formats(self) -> None:
        base_config = load_config(Path("src/results/tls/config.yaml"))
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = replace(
                base_config,
                figure_root=Path(temporary_directory),
                plotting={
                    **base_config.plotting,
                    "population_size_formats": ["png"],
                },
            )
            ranges = {}
            metadata = {}
            for dataset, rate in (("TLS", 0.1), ("TLSC", 0.2)):
                range_rows = []
                metadata_rows = []
                for state_index, state in enumerate(config.state_names):
                    parameter = f"sampling[{state}]"
                    metadata_rows.append(
                        {
                            "parameter": parameter,
                            "parameter_type": "sampling",
                            "parameter_index": state_index,
                            "original_value": rate,
                        }
                    )
                    range_rows.extend(
                        [
                            {
                                "parameter": parameter,
                                "likelihood_fraction": 0.5,
                                "lower_parameter_value": rate * 0.9,
                                "higher_parameter_value": rate * 1.1,
                            },
                            {
                                "parameter": parameter,
                                "likelihood_fraction": 1e-5,
                                "lower_parameter_value": rate * 0.8,
                                "higher_parameter_value": rate * 1.2,
                            },
                        ]
                    )
                ranges[dataset] = pd.DataFrame(range_rows)
                metadata[dataset] = pd.DataFrame(metadata_rows)
            sampled_counts = {
                "TLS": {"NMPs": 378, "Somite": 869, "NeuralTube": 1216},
                "TLSC": {"NMPs": 1342, "Somite": 1381, "NeuralTube": 411},
            }

            profile_outputs = _render_population_size_profile(
                config,
                ranges,
                metadata,
                sampled_counts,
            )
            dot_outputs = _render_sampled_population_size_plot(
                config,
                sampled_counts,
            )

            self.assertEqual(
                {path.suffix for path in profile_outputs},
                {".png"},
            )
            self.assertEqual({path.suffix for path in dot_outputs}, {".png"})
            self.assertTrue(all(path.is_file() for path in profile_outputs + dot_outputs))


if __name__ == "__main__":
    unittest.main()
