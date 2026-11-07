from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.plot_linear_transition_inference_accuracy import (
    _read_diversitree_estimates,
    _read_ground_truth,
    _read_joint_estimates,
    load_concert_transition_probability_errors,
    load_concert_transition_time_errors,
)


class PlotLinearTransitionInferenceAccuracyTests(unittest.TestCase):
    def test_ground_truth_uses_backward_age_for_each_edge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "generated_settings.csv"
            fieldnames = [
                "setting",
                "sampling_rate",
                *(f"birth_{state}" for state in "ABCD"),
                *(f"death_{state}" for state in "ABCD"),
                "p_A_to_B",
                "p_B_to_C",
                "p_C_to_D",
                "transition_time_A",
                "transition_time_B",
                "transition_time_C",
            ]
            row = {
                "setting": "setting_1",
                "sampling_rate": 0.5,
                **{f"birth_{state}": 4 for state in "ABCD"},
                **{f"death_{state}": 1 for state in "ABCD"},
                "p_A_to_B": 0.7,
                "p_B_to_C": 0.6,
                "p_C_to_D": 0.5,
                "transition_time_A": 0.1,
                "transition_time_B": 0.2,
                "transition_time_C": 0.3,
            }
            with path.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)

            truth = _read_ground_truth(path)["setting_1"]

        self.assertAlmostEqual(truth["transition_time_A_to_B"], 0.9)
        self.assertAlmostEqual(truth["transition_time_B_to_C"], 0.8)
        self.assertAlmostEqual(truth["transition_time_C_to_D"], 0.7)
        self.assertAlmostEqual(truth["net_growth_A"], 3.0)
        self.assertAlmostEqual(truth["scaled_birth_A"], 2.0)

    def test_joint_output_reads_edge_specific_probabilities_and_times(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "inference.out"
            path.write_text(
                "LL=-12\n"
                "birth: 4 5 6 7\n"
                "death: 1 1 2 2\n"
                "sampling: 0.5 0.4 0.3 0.2\n"
                "transition_probabilities_by_epoch:\n"
                "epoch_1 2->3\n"
                "2->2 0.4\n"
                "2->3 0.6\n"
                "epoch_2 1->2\n"
                "1->1 0.3\n"
                "1->2 0.7\n"
                "epoch_3 0->1\n"
                "0->0 0.2\n"
                "0->1 0.8\n"
                "transition_times_by_edge:\n"
                "2->3 0.11\n"
                "1->2 0.22\n"
                "0->1 0.33\n"
            )

            estimates = _read_joint_estimates(path)

        self.assertAlmostEqual(estimates["p_A_to_B"], 0.8)
        self.assertAlmostEqual(estimates["p_B_to_C"], 0.7)
        self.assertAlmostEqual(estimates["p_C_to_D"], 0.6)
        self.assertAlmostEqual(estimates["transition_time_A_to_B"], 0.33)
        self.assertAlmostEqual(estimates["transition_time_B_to_C"], 0.22)
        self.assertAlmostEqual(estimates["transition_time_C_to_D"], 0.11)

    def test_loads_absolute_transition_time_errors_for_concert(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            settings_path = data_dir / "generated_settings.csv"
            fieldnames = [
                "setting",
                "sampling_rate",
                *(f"birth_{state}" for state in "ABCD"),
                *(f"death_{state}" for state in "ABCD"),
                "p_A_to_B",
                "p_B_to_C",
                "p_C_to_D",
                "transition_time_A",
                "transition_time_B",
                "transition_time_C",
            ]
            row = {
                "setting": "setting_1",
                "sampling_rate": 0.5,
                **{f"birth_{state}": 4 for state in "ABCD"},
                **{f"death_{state}": 1 for state in "ABCD"},
                "p_A_to_B": 0.7,
                "p_B_to_C": 0.6,
                "p_C_to_D": 0.5,
                "transition_time_A": 0.1,
                "transition_time_B": 0.2,
                "transition_time_C": 0.3,
            }
            with settings_path.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)

            batch_dir = data_dir / "setting_1" / "batch_0"
            batch_dir.mkdir(parents=True)
            (
                batch_dir / "inference_joint_coordinate_first_8_trees.out"
            ).write_text(
                "LL=-1\n"
                "birth: 4 5 6 7\n"
                "death: 1 1 2 2\n"
                "sampling: 0.5 0.4 0.3 0.2\n"
                "transition_probabilities_by_epoch:\n"
                "epoch_1 2->3\n"
                "2->2 0.4\n"
                "2->3 0.6\n"
                "epoch_2 1->2\n"
                "1->1 0.3\n"
                "1->2 0.7\n"
                "epoch_3 0->1\n"
                "0->0 0.2\n"
                "0->1 0.8\n"
                "transition_times_by_edge:\n"
                "2->3 0.6\n"
                "1->2 0.7\n"
                "0->1 0.8\n"
            )

            errors = load_concert_transition_time_errors(data_dir)
            probability_errors = load_concert_transition_probability_errors(data_dir)

        self.assertEqual(
            [error.edge for error in errors],
            ["A→B", "B→C", "C→D"],
        )
        for error in errors:
            self.assertAlmostEqual(error.value, 0.1)
        self.assertEqual(
            [error.edge for error in probability_errors],
            ["A→B", "B→C", "C→D"],
        )
        for error in probability_errors:
            self.assertAlmostEqual(error.value, 0.1)

    def test_diversitree_estimates_derive_only_comparable_rate_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fit_dir = Path(temporary_directory)
            summary_path = fit_dir / "joint_musse_fit.txt"
            summary_path.write_text(
                "transition_model=linear\n"
                "convergence_code=0\n"
                "sampling_fraction=0.5,0.5,0.5,0.5\n"
            )
            estimates_path = fit_dir / "parameter_estimates.csv"
            with estimates_path.open("w", newline="") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=("parameter", "estimate"),
                )
                writer.writeheader()
                for index in range(1, 5):
                    writer.writerow(
                        {"parameter": f"lambda{index}", "estimate": 4.0}
                    )
                    writer.writerow(
                        {"parameter": f"mu{index}", "estimate": 1.0}
                    )
                for parameter in ("q12", "q23", "q34"):
                    writer.writerow({"parameter": parameter, "estimate": 0.25})

            estimates = _read_diversitree_estimates(summary_path)

        self.assertAlmostEqual(estimates["net_growth_A"], 3.0)
        self.assertAlmostEqual(estimates["scaled_birth_A"], 2.0)
        self.assertAlmostEqual(estimates["transition_rate_A_to_B"], 0.25)
        self.assertAlmostEqual(estimates["transition_rate_B_to_C"], 0.25)
        self.assertAlmostEqual(estimates["transition_rate_C_to_D"], 0.25)
        self.assertNotIn("p_A_to_B", estimates)
        self.assertNotIn("transition_time_A_to_B", estimates)


if __name__ == "__main__":
    unittest.main()
