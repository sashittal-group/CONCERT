from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.generate_star_coord_settings import generate_settings
from src.simulator_coord import read_simulation_settings


class GenerateStarCoordSettingsTests(unittest.TestCase):
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
