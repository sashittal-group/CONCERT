from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.plot_star_diversitree_errors import (
    _derived_rates,
    load_matched_absolute_errors,
    plot_error_boxplots,
)


class PlotStarDiversitreeErrorsTests(unittest.TestCase):
    def test_derives_net_growth_and_scaled_birth(self) -> None:
        rates = _derived_rates(
            (8.0, 7.0, 6.0, 5.0),
            (2.0, 3.0, 1.0, 4.0),
            (0.5, 0.4, 0.3, 0.2),
        )

        self.assertEqual(rates["net_growth"], (6.0, 4.0, 5.0, 1.0))
        for actual, expected in zip(
            rates["scaled_birth"],
            (4.0, 2.8, 1.8, 1.0),
        ):
            self.assertAlmostEqual(actual, expected)

    def test_loads_matched_errors_and_omits_nonconverged_fits(self) -> None:
        header = [
            "setting",
            *(f"birth_{state}" for state in "ABCD"),
            *(f"death_{state}" for state in "ABCD"),
            *(f"p_A_to_{state}" for state in "ABCD"),
            "transition_time_1",
            "transition_time_2",
            "sampling_rate",
            "num_trees",
        ]
        truth_row = [
            "setting_1",
            *("8" for _ in "ABCD"),
            *("2" for _ in "ABCD"),
            *("0.25" for _ in "ABCD"),
            "0.1",
            "0.2",
            "0.5",
            "100",
        ]
        coordinated_output = """\
LL=1.0
birth: 9 8 7 6
death: 2 2 2 2
sampling: 0.5 0.5 0.5 0.5
transitions:
0->0 0.25
0->1 0.25
0->2 0.25
0->3 0.25
transition_times:
0 0.1
1 0.2
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            with (data_dir / "generated_settings.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerow(truth_row)
            batch_dir = data_dir / "setting_1" / "batch_0"
            batch_dir.mkdir(parents=True)
            (
                batch_dir / "inference_progressive_joint_first_8_trees.out"
            ).write_text(coordinated_output)

            fit_dir = batch_dir / "diversitree_musse"
            fit_dir.mkdir()
            (fit_dir / "joint_musse_fit.txt").write_text(
                "convergence_code=0\nsampling_fraction=0.5,0.5,0.5,0.5\n"
            )
            with (fit_dir / "parameter_estimates.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(("parameter", "estimate"))
                for index, state in enumerate("ABCD", start=1):
                    writer.writerow((f"lambda{index}", 8))
                    writer.writerow((f"mu{index}", 2))

            failed_fit = data_dir / "setting_1" / "batch_1" / "diversitree_musse"
            failed_fit.mkdir(parents=True)
            (failed_fit / "joint_musse_fit.txt").write_text(
                "convergence_code=52\n"
            )

            errors = load_matched_absolute_errors(data_dir)
            output_path = data_dir / "figures" / "errors.png"
            plot_error_boxplots(errors, output_path)
            self.assertTrue(output_path.is_file())

        self.assertEqual(len(errors), 16)
        self.assertEqual(
            {
                (error.method, error.metric, error.state)
                for error in errors
            },
            {
                (method, metric, state)
                for method in ("Coordinated", "diversitree")
                for metric in ("net_growth", "scaled_birth")
                for state in "ABCD"
            },
        )
        coordinated_net_growth_a = next(
            error.value
            for error in errors
            if error.method == "Coordinated"
            and error.metric == "net_growth"
            and error.state == "A"
        )
        self.assertEqual(coordinated_net_growth_a, 1.0)


if __name__ == "__main__":
    unittest.main()
