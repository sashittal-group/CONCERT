from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.generate_coord_settings import generate_settings
from src.simulator_coord import read_simulation_settings


class GenerateCoordSettingsTests(unittest.TestCase):
    def test_generates_full_valid_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)

            count = generate_settings(output_dir)

            self.assertEqual(count, 768)
            self.assertTrue((output_dir / "generated_settings.csv").is_file())
            self.assertTrue((output_dir / "setting_1" / "setting.txt").is_file())
            self.assertTrue((output_dir / "setting_768" / "setting.txt").is_file())

            first = read_simulation_settings(
                output_dir / "setting_1" / "setting.txt"
            )
            self.assertEqual(first["birth_rates"], {"A": 6.0, "B": 6.0})
            self.assertEqual(first["death_rates"], {"A": 1.0, "B": 1.0})
            self.assertEqual(first["sampling_rates"], {"A": 0.5, "B": 0.5})
            self.assertEqual(first["transition_times"], {"A": [0.5], "B": []})
            self.assertEqual(
                first["transition_probabilities"].loc["A", "B"],
                0.25,
            )
            self.assertEqual(first["num_trees"], 100)

    def test_does_not_overwrite_existing_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            (output_dir / "setting_3").mkdir()

            with self.assertRaises(FileExistsError):
                generate_settings(output_dir)

            self.assertFalse((output_dir / "setting_1").exists())


if __name__ == "__main__":
    unittest.main()
