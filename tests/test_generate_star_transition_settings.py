from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.generate_star_transition_settings import generate_star_transition_settings
from src.simulator_coord import read_simulation_settings


class GenerateStarTransitionSettingsTests(unittest.TestCase):
    def test_generates_separate_two_and_three_time_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            generate_star_transition_settings(output_root, num_settings=1)

            for count in (2, 3):
                settings = read_simulation_settings(
                    output_root
                    / f"star_{count}"
                    / "setting_1"
                    / "setting.txt"
                )
                self.assertEqual(len(settings["transition_times"]["A"]), count)
                self.assertTrue(
                    (
                        output_root
                        / f"star_{count}"
                        / "generated_settings.csv"
                    ).is_file()
                )

    def test_overwrite_regenerates_settings_with_current_probabilities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            generate_star_transition_settings(output_root, num_settings=1)

            setting_path = output_root / "star_2" / "setting_1" / "setting.txt"
            original_setting = setting_path.read_text()
            setting_path.write_text(
                original_setting.replace(
                    "A -> A ",
                    "A -> A 0.123456789 ",
                    1,
                )
            )

            generate_star_transition_settings(
                output_root,
                num_settings=1,
                overwrite_existing=True,
            )

            self.assertNotIn("A -> A 0.123456789", setting_path.read_text())
            settings = read_simulation_settings(setting_path)
            self.assertAlmostEqual(
                settings["transition_probabilities"].loc["A"].sum(),
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
