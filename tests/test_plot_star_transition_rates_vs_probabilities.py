from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.plot_star_transition_rates_vs_probabilities import (
    RateObservation,
    _proportionality_statistics,
    calibrate_rates_to_probabilities,
    collect_rate_observations,
    plot_rate_probability_relationships,
)


class PlotStarTransitionRatesTests(unittest.TestCase):
    def test_proportionality_statistics_identify_origin_line(self) -> None:
        slope, correlation, r_squared = _proportionality_statistics(
            [0.1, 0.2, 0.3],
            [0.2, 0.4, 0.6],
        )

        self.assertAlmostEqual(slope, 2.0)
        self.assertAlmostEqual(correlation, 1.0)
        self.assertAlmostEqual(r_squared, 1.0)

    def test_calibrates_rate_to_probability_without_training_on_held_out_setting(self) -> None:
        observations = [
            RateObservation(
                method="diversitree",
                setting=f"setting_{index}",
                transition=transition,
                probability=probability,
                rate=slope * probability,
                batch="batch_0",
            )
            for transition, slope in (("B", 2.0), ("C", 3.0), ("D", 4.0))
            for index, probability in enumerate((0.1, 0.2, 0.3), start=1)
        ]

        calibrated = calibrate_rates_to_probabilities(
            observations,
            "diversitree",
        )

        self.assertEqual(len(calibrated), 9)
        for item, expected_probability in zip(
            calibrated,
            (0.1, 0.2, 0.3) * 3,
        ):
            self.assertAlmostEqual(item.estimate, expected_probability)
            self.assertFalse(item.clipped)

    def test_loads_diversitree_and_clock_scaled_beam_rates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            with (data_dir / "generated_settings.csv").open("w", newline="") as source:
                writer = csv.writer(source)
                writer.writerow(("setting", "p_A_to_B", "p_A_to_C", "p_A_to_D"))
                writer.writerow(("setting_1", 0.2, 0.3, 0.4))

            fit_dir = data_dir / "setting_1" / "batch_0" / "diversitree_musse"
            fit_dir.mkdir(parents=True)
            (fit_dir / "joint_musse_fit.txt").write_text("convergence_code=0\n")
            with (fit_dir / "parameter_estimates.csv").open(
                "w",
                newline="",
            ) as source:
                writer = csv.writer(source)
                writer.writerow(("parameter", "estimate"))
                writer.writerows((("q12", 1.0), ("q13", 2.0), ("q14", 3.0)))

            beam_dir = (
                data_dir
                / "setting_1"
                / "batch_0"
                / "tree_0"
                / "beam_snakemake"
            )
            beam_dir.mkdir(parents=True)
            (beam_dir / "run_status.tsv").write_text(
                "tree_id\tstatus\nsetting_1/batch_0/tree_0\tcompleted\n"
            )
            (beam_dir / "parameter_summary.tsv").write_text(
                "parameter\tsamples_after_burnin\n"
                "tissueClockRate\t2\n"
            )
            with (beam_dir / "result.log").open("w") as source:
                source.write(
                    "Sample\ttissueClockRate\t"
                    "tissueSubstModelLogger.relMigRate_A_B\t"
                    "tissueSubstModelLogger.relMigRate_A_C\t"
                    "tissueSubstModelLogger.relMigRate_A_D\n"
                    "0\t100\t100\t100\t100\n"
                    "1\t2\t0.5\t0.25\t0.1\n"
                    "2\t4\t0.25\t0.5\t0.2\n"
                )

            observations = collect_rate_observations(data_dir)
            output_path = data_dir / "figures" / "rates.png"
            plot_rate_probability_relationships(observations, output_path)
            self.assertTrue(output_path.is_file())

        diversitree_rates = {
            item.transition: item.rate
            for item in observations
            if item.method == "diversitree"
        }
        beam_rates = {
            item.transition: item.rate
            for item in observations
            if item.method == "BEAM"
        }
        self.assertEqual(diversitree_rates, {"B": 1.0, "C": 2.0, "D": 3.0})
        self.assertEqual(beam_rates, {"B": 1.0, "C": 1.25, "D": 0.5})


if __name__ == "__main__":
    unittest.main()
