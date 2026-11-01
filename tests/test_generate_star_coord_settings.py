from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.generate_star_coord_settings import (
    ROOT_PROBABILITY_WEIGHTS_BY_TRANSITION_COUNT,
    _sample_weighted_probabilities,
    generate_settings,
)
from src.simulator_coord import read_simulation_settings


class GenerateStarCoordSettingsTests(unittest.TestCase):
    def test_transition_count_weights(self) -> None:
        self.assertEqual(
            ROOT_PROBABILITY_WEIGHTS_BY_TRANSITION_COUNT,
            {
                2: (2.0, 1.0, 1.0, 1.0),
                3: (3.0, 1.0, 1.0, 1.0),
            },
        )

    def test_weighted_uniform_transition_probabilities_are_normalized(self) -> None:
        class FixedUniform:
            def random(self) -> float:
                return 0.5

        rng = FixedUniform()
        for count, weights in ROOT_PROBABILITY_WEIGHTS_BY_TRANSITION_COUNT.items():
            with self.subTest(count=count):
                probabilities = _sample_weighted_probabilities(rng, weights)
                total_weight = sum(weights)
                for probability, weight in zip(probabilities, weights):
                    self.assertAlmostEqual(probability, weight / total_weight)
                self.assertAlmostEqual(sum(probabilities), 1.0)

    def test_resume_reuses_matching_settings_and_generates_missing_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            generate_settings(
                output_dir,
                num_settings=1,
                seed=2,
                num_transition_times=2,
            )
            existing_setting = (
                output_dir / "setting_1" / "setting.txt"
            ).read_text()

            generated = generate_settings(
                output_dir,
                num_settings=3,
                seed=2,
                num_transition_times=2,
                resume_existing=True,
            )

            self.assertEqual(generated, 3)
            self.assertEqual(
                (output_dir / "setting_1" / "setting.txt").read_text(),
                existing_setting,
            )
            for setting_id in range(1, 4):
                settings = read_simulation_settings(
                    output_dir / f"setting_{setting_id}" / "setting.txt"
                )
                self.assertEqual(len(settings["transition_times"]["A"]), 2)
            with (output_dir / "generated_settings.csv").open(newline="") as manifest:
                self.assertEqual(len(list(csv.DictReader(manifest))), 3)

    def test_resume_rejects_existing_settings_with_wrong_transition_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            generate_settings(
                output_dir,
                num_settings=1,
                seed=2,
                num_transition_times=2,
            )

            with self.assertRaisesRegex(ValueError, "expected 3"):
                generate_settings(
                    output_dir,
                    num_settings=1,
                    seed=2,
                    num_transition_times=3,
                    resume_existing=True,
                )

    def test_generates_settings_with_requested_transition_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            for count in (2, 3):
                with self.subTest(count=count):
                    output_dir = Path(temporary_directory) / f"star_{count}"
                    generated = generate_settings(
                        output_dir,
                        num_settings=1,
                        seed=7,
                        num_transition_times=count,
                    )

                    self.assertEqual(generated, 1)
                    settings = read_simulation_settings(
                        output_dir / "setting_1" / "setting.txt"
                    )
                    self.assertEqual(
                        len(settings["transition_times"]["A"]),
                        count,
                    )
                    for state in settings["states"]:
                        self.assertAlmostEqual(
                            settings["transition_probabilities"].loc[state].sum(),
                            1.0,
                        )

    def test_generates_valid_four_state_star_setting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)

            count = generate_settings(output_dir)

            self.assertEqual(count, 1)
            self.assertTrue((output_dir / "generated_settings.csv").is_file())
            setting_path = output_dir / "setting_1" / "setting.txt"
            self.assertTrue(setting_path.is_file())

            setting = read_simulation_settings(setting_path)
            self.assertEqual(
                setting["birth_rates"],
                {"A": 6.0, "B": 8.0, "C": 10.0, "D": 12.0},
            )
            self.assertEqual(
                setting["death_rates"],
                {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0},
            )
            self.assertEqual(
                setting["sampling_rates"],
                {"A": 0.5, "B": 0.5, "C": 0.5, "D": 0.5},
            )
            self.assertEqual(
                setting["transition_times"],
                {"A": [0.5], "B": [], "C": [], "D": []},
            )
            probabilities = setting["transition_probabilities"]
            self.assertEqual(probabilities.loc["A", "A"], 0.0)
            for state in ("B", "C", "D"):
                self.assertAlmostEqual(probabilities.loc["A", state], 1 / 3)
            for state in ("B", "C", "D"):
                self.assertEqual(probabilities.loc[state, state], 1.0)
                for other_state in ("A", "B", "C", "D"):
                    if state != other_state:
                        self.assertEqual(probabilities.loc[state, other_state], 0.0)
            self.assertEqual(setting["num_trees"], 100)

    def test_does_not_overwrite_existing_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            (output_dir / "setting_1").mkdir()

            with self.assertRaises(FileExistsError):
                generate_settings(output_dir)

            self.assertFalse((output_dir / "generated_settings.csv").exists())


if __name__ == "__main__":
    unittest.main()
