from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.summarize_fixed_sampling_transition_runs import (
    _scaled_birth_rate,
    _transition_rate,
    create_summary_figures,
    load_runs,
)


class FixedSamplingTransitionSummaryTests(unittest.TestCase):
    def test_missing_output_is_rebuilt_from_latest_complete_log_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            log_path = data_dir / "inference_transition_times_2.log"
            log_path.write_text(
                "ITER 10 LL=-12\n"
                "birth: 3 4 5\n"
                "death: 1 2 3\n"
                "sampling: 0.1 0.1 0.1\n"
                "transitions:\n"
                "0->1 0.2\n"
                "0->2 0.1\n"
                "transition_times:\n"
                "0.0001 0.5\n"
                "\n"
                "ITER 20 LL=-10\n"
                "birth: 5 6 7\n"
                "death: 1 2 3\n"
                "sampling: 0.1 0.1 0.1\n"
                "transitions:\n"
                "0->1 0.3\n"
                "0->2 0.15\n"
                "transition_times:\n"
                "0.0001 0.5 1.0\n"
                "\n"
                "ITER 30 LL=-9\n"
                "birth: 7 8 9\n"
            )

            runs = load_runs(data_dir)

            temporary_output = data_dir / "inference_transition_times_2.tmp.out"
            self.assertTrue(temporary_output.is_file())
            self.assertIn("ITER 20 LL=-10", temporary_output.read_text())
            self.assertEqual(runs[0].birth_rates, (5.0, 6.0, 7.0))
            self.assertEqual(runs[0].death_rates, (1.0, 2.0, 3.0))
            self.assertEqual(
                runs[0].transition_probabilities,
                {(0, 1): 0.3, (0, 2): 0.15},
            )
            self.assertEqual(runs[0].iterations_completed, 20)
            self.assertFalse(runs[0].completed)
            self.assertEqual(_scaled_birth_rate(runs[0], 0), 0.5)
            self.assertAlmostEqual(_scaled_birth_rate(runs[0], 1), 0.6)
            self.assertEqual(_transition_rate(runs[0], (0, 1)), 0.6)
            self.assertEqual(runs[0].count, 2)

    def test_existing_output_is_used_and_figures_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            (data_dir / "inference_transition_times_2.out").write_text(
                "LL=-10\n"
                "birth: 5 6 7\n"
                "death: 1 2 3\n"
                "sampling: 0.1 0.1 0.1\n"
                "transitions:\n"
                "0->1 0.2\n"
                "0->2 0.1\n"
                "transition_times:\n"
                "0 0.0001\n"
                "1 0.5\n"
                "2 1.0\n"
            )
            (data_dir / "inference_transition_times_2.log").write_text(
                "ITER 10 LL=-12\n"
                "birth: 3 4\n"
                "death: 1 2\n"
                "sampling: 0.1 0.1\n"
            )
            figure_dir = data_dir / "figures"

            figures = create_summary_figures(data_dir, figure_dir)

            self.assertEqual(len(figures), 4)
            self.assertTrue(all(path.is_file() for path in figures))
            run = load_runs(data_dir)[0]
            self.assertEqual(run.iterations_completed, 10)
            self.assertTrue(run.completed)
            self.assertFalse(
                (data_dir / "inference_transition_times_2.tmp.out").exists()
            )

    def test_existing_temporary_output_is_used_when_log_has_no_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            (data_dir / "inference_transition_times_2.log").write_text("")
            temporary_output = data_dir / "inference_transition_times_2.tmp.out"
            temporary_output.write_text(
                "birth: 5 6 7\n"
                "death: 1 2 3\n"
                "sampling: 0.1 0.1 0.1\n"
                "transitions:\n"
                "0->1 0.2\n"
                "0->2 0.1\n"
                "transition_times:\n"
                "0 0.0001\n"
                "1 0.5\n"
            )

            runs = load_runs(data_dir)

            self.assertEqual(runs[0].source, temporary_output)
            self.assertEqual(runs[0].iterations_completed, 0)

    def test_max_count_skips_larger_transition_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            (data_dir / "inference_transition_times_2.out").write_text(
                "LL=-10\n"
                "birth: 5 6 7\n"
                "death: 1 2 3\n"
                "sampling: 0.1 0.1 0.1\n"
                "transitions:\n"
                "0->1 0.2\n"
                "0->2 0.1\n"
                "transition_times:\n"
                "0 0.0001\n"
                "1 0.5\n"
            )
            (data_dir / "inference_transition_times_200.log").write_text(
                "ITER 1 LL=-20\nbirth: 1 2\n"
            )

            runs = load_runs(data_dir, max_count=100)

            self.assertEqual([run.count for run in runs], [2])
            self.assertFalse(
                (data_dir / "inference_transition_times_200.tmp.out").exists()
            )


if __name__ == "__main__":
    unittest.main()
