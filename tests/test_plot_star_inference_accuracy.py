from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.main2_joint_transition_times import coordinate_search_transition_times
from src.plot_star_inference_accuracy import (
    IDENTIFIABLE_PARAMETERS,
    SAMPLE_SIZES,
    TRANSITION_TIME_PARAMETERS,
    _add_identifiable_parameters,
    _parse_output,
    _read_ground_truth,
    _r_squared,
    load_completed_progressive_comparisons,
)


class PlotStarInferenceAccuracyTests(unittest.TestCase):
    def test_plot_sample_sizes_include_32_trees(self) -> None:
        self.assertEqual(SAMPLE_SIZES, (4, 8, 32))

    def test_identifiable_parameters_include_transition_times(self) -> None:
        self.assertEqual(
            IDENTIFIABLE_PARAMETERS[-len(TRANSITION_TIME_PARAMETERS) :],
            TRANSITION_TIME_PARAMETERS,
        )

    def test_parses_main2_parameter_output(self) -> None:
        output = """\
LL=1.0
birth: 1 2 3 4
death: 0.1 0.2 0.3 0.4
sampling: 0.5 0.6 0.7 0.8
transitions:
0->0 0.4
0->1 0.2
0->2 0.2
0->3 0.2
1->1 1
2->2 1
3->3 1
transition_times:
0 0.3
1 0.1
2 0.5
3 0.2
4 0.4
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "inference.out"
            output_path.write_text(output)

            estimates = _parse_output(output_path)

        self.assertEqual(estimates["birth_C"], 3.0)
        self.assertEqual(estimates["death_D"], 0.4)
        self.assertEqual(estimates["sampling_B"], 0.6)
        self.assertEqual(estimates["p_A_to_A"], 0.4)
        self.assertEqual(estimates["p_A_to_D"], 0.2)
        self.assertEqual(
            [
                estimates[f"transition_time_{index}"]
                for index in range(1, 6)
            ],
            [0.1, 0.2, 0.3, 0.4, 0.5],
        )

    def test_r_squared_and_constant_truth(self) -> None:
        self.assertEqual(_r_squared([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)
        self.assertIsNone(_r_squared([0.5, 0.5], [0.4, 0.6]))

    def test_derived_identifiable_parameters(self) -> None:
        values = {}
        for state in ("A", "B", "C", "D"):
            values.update(
                {
                    f"birth_{state}": 6.0,
                    f"death_{state}": 2.0,
                    f"sampling_{state}": 0.5,
                }
            )

        derived = _add_identifiable_parameters(values)

        self.assertEqual(derived["net_growth_A"], 4.0)
        self.assertEqual(derived["scaled_sampling_A"], 3.0)

    def test_ground_truth_transition_times_use_tree_time_coordinates(self) -> None:
        header = [
            "setting",
            *(f"birth_{state}" for state in ("A", "B", "C", "D")),
            *(f"death_{state}" for state in ("A", "B", "C", "D")),
            *(f"p_A_to_{state}" for state in ("A", "B", "C", "D")),
            *(f"transition_time_{index}" for index in range(1, 6)),
            "sampling_rate",
            "num_trees",
        ]
        values = [
            "setting_1",
            *("8" for _ in range(4)),
            *("2" for _ in range(4)),
            *("0.25" for _ in range(4)),
            "0.1",
            "0.2",
            "0.3",
            "0.4",
            "0.5",
            "0.5",
            "100",
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "settings.csv"
            manifest_path.write_text(",".join(header) + "\n" + ",".join(values) + "\n")

            ground_truth = _read_ground_truth(manifest_path)

        self.assertEqual(
            [ground_truth["setting_1"][f"transition_time_{index}"] for index in range(1, 6)],
            [0.5, 0.6, 0.7, 0.8, 0.9],
        )

    def test_loads_completed_two_transition_progressive_inference(self) -> None:
        header = [
            "setting",
            *(f"birth_{state}" for state in ("A", "B", "C", "D")),
            *(f"death_{state}" for state in ("A", "B", "C", "D")),
            *(f"p_A_to_{state}" for state in ("A", "B", "C", "D")),
            "transition_time_1",
            "transition_time_2",
            "sampling_rate",
            "num_trees",
        ]
        values = [
            "setting_1",
            *("8" for _ in range(4)),
            *("2" for _ in range(4)),
            *("0.25" for _ in range(4)),
            "0.1",
            "0.2",
            "0.5",
            "100",
        ]
        inference_output = """\
LL=1.0
birth: 7 8 9 10
death: 1 2 3 4
sampling: 0.5 0.5 0.5 0.5
transitions:
0->0 0.25
0->1 0.25
0->2 0.25
0->3 0.25
transition_times:
0 0.81
1 0.91
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            (data_dir / "generated_settings.csv").write_text(
                ",".join(header) + "\n" + ",".join(values) + "\n"
            )
            batch_dir = data_dir / "setting_1" / "batch_0"
            batch_dir.mkdir(parents=True)
            (
                batch_dir / "inference_progressive_joint_first_8_trees.out"
            ).write_text(inference_output)

            comparisons = load_completed_progressive_comparisons(
                data_dir,
                num_transition_times=2,
            )

        self.assertEqual(len(comparisons), 1)
        self.assertEqual(comparisons[0].sample_size, 8)
        self.assertEqual(comparisons[0].truth["transition_time_1"], 0.8)
        self.assertEqual(comparisons[0].truth["transition_time_2"], 0.9)
        self.assertEqual(comparisons[0].inferred["net_growth_A"], 6.0)
        self.assertEqual(comparisons[0].inferred["transition_time_1"], 0.81)
        self.assertEqual(comparisons[0].inferred["transition_time_2"], 0.91)

    def test_transition_time_coordinate_search_keeps_order_and_improves_likelihood(
        self,
    ) -> None:
        objective = lambda times: -sum(
            (actual - expected) ** 2
            for actual, expected in zip(times, (0.3, 0.7))
        )
        initial_times = [0.2, 0.8]
        initial_likelihood = objective(initial_times)

        result_times, result_likelihood = coordinate_search_transition_times(
            initial_times,
            max_time=1.0,
            step_size=0.1,
            objective=objective,
        )

        self.assertAlmostEqual(result_times[0], 0.3)
        self.assertAlmostEqual(result_times[1], 0.7)
        self.assertGreater(result_likelihood, initial_likelihood)


if __name__ == "__main__":
    unittest.main()
