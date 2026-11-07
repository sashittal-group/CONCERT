from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from src.generate_linear_transition_settings import (
    BIRTH_RATE_CHOICES,
    DEATH_RATE_CHOICES,
    generate_settings,
    sample_setting,
)


class GenerateLinearTransitionSettingsTests(unittest.TestCase):
    def test_birth_rates_remain_above_the_death_rate_range(self) -> None:
        self.assertEqual(BIRTH_RATE_CHOICES, (6, 8, 10, 12))
        self.assertGreater(min(BIRTH_RATE_CHOICES), max(DEATH_RATE_CHOICES))

    def test_settings_have_ordered_times_and_only_linear_transitions(self) -> None:
        setting = sample_setting(random.Random(12))
        times = [setting.transition_times[state] for state in ("A", "B", "C")]

        self.assertEqual(len(set(times)), 3)
        self.assertEqual(times, sorted(times))
        self.assertIsNone(setting.transition_times["D"])

        for source, target in (("A", "B"), ("B", "C"), ("C", "D")):
            row = setting.transition_probabilities[source]
            self.assertAlmostEqual(sum(row.values()), 1.0)
            self.assertGreater(row[source], 0.0)
            self.assertGreater(row[target], 0.0)
            self.assertTrue(
                all(
                    probability == 0.0
                    for state, probability in row.items()
                    if state not in (source, target)
                )
            )
        self.assertEqual(setting.transition_probabilities["D"]["D"], 1.0)

    def test_weighted_probabilities_reflect_configured_relative_weights(self) -> None:
        settings = [
            sample_setting(random.Random(seed))
            for seed in range(1000)
        ]
        mean_probabilities = {
            edge: sum(setting.transition_probabilities[edge[0]][edge[1]]
                      for setting in settings) / len(settings)
            for edge in (("A", "B"), ("B", "C"), ("C", "D"))
        }

        self.assertAlmostEqual(mean_probabilities[("A", "B")], 0.75, delta=0.04)
        self.assertAlmostEqual(mean_probabilities[("B", "C")], 2 / 3, delta=0.04)
        self.assertAlmostEqual(mean_probabilities[("C", "D")], 0.5, delta=0.04)

    def test_generated_settings_include_times_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            generate_settings(output_dir, num_settings=1, seed=5)

            setting_text = (
                output_dir / "setting_1" / "setting.txt"
            ).read_text()
            lines = setting_text.splitlines()
            transition_times_start = lines.index(
                "# transition times (use - if a state has none)"
            ) + 1
            transition_time_lines = lines[
                transition_times_start:transition_times_start + 4
            ]
            self.assertEqual(len(transition_time_lines), 4)
            self.assertEqual(transition_time_lines[3], "D -")
            times = [float(line.split()[1]) for line in transition_time_lines[:3]]
            self.assertEqual(times, sorted(times))
            self.assertIn("A -> B ", setting_text)
            self.assertIn("B -> C ", setting_text)
            self.assertIn("C -> D ", setting_text)
            self.assertTrue((output_dir / "generated_settings.csv").is_file())


if __name__ == "__main__":
    unittest.main()
