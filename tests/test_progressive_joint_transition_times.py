from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace

import torch

from src.dtos import OptimizationResult
from src.main2 import write_result
from src.main2_joint_transition_times import _fixed_parameter_likelihood
from src.progressive_joint_transition_times import (
    _add_transition_at_maximum,
    _fit_parameters,
    _load_initial_result,
    _maximize_coordinate,
    _select_initial_transition_times,
    optimize_progressive_joint_transition_times,
)


class ProgressiveJointTransitionTimesTests(unittest.TestCase):
    def test_coordinate_sweep_finds_global_grid_maximum(self) -> None:
        times, likelihood = _maximize_coordinate(
            [0.2, 0.8],
            coordinate=0,
            objective=lambda proposal: -(proposal[0] - 0.37) ** 2,
            max_time=1.0,
            grid_size=1000,
        )

        self.assertEqual(times, [0.37, 0.8])
        self.assertEqual(likelihood, 0.0)

    def test_coordinate_sweep_keeps_finite_time_when_grid_has_no_candidate(self) -> None:
        initial = [0.0047, 0.0081]
        times, likelihood = _maximize_coordinate(
            initial,
            coordinate=0,
            objective=lambda _proposal: -1.0,
            max_time=1.0,
            grid_size=100,
        )

        self.assertEqual(times, initial)
        self.assertEqual(likelihood, -1.0)

    def test_new_transition_is_added_at_global_maximum(self) -> None:
        times, likelihood = _add_transition_at_maximum(
            [0.6],
            objective=lambda proposal: -(proposal[0] - 0.25) ** 2,
            max_time=1.0,
            grid_size=1000,
        )

        self.assertEqual(times, [0.25, 0.6])
        self.assertEqual(likelihood, 0.0)

    def test_initial_transition_times_can_be_evenly_subsampled(self) -> None:
        times = [float(index) / 20 for index in range(1, 16)]

        selected = _select_initial_transition_times(times, 10)

        self.assertEqual(
            selected,
            [times[index] for index in (0, 2, 3, 5, 6, 8, 9, 11, 12, 14)],
        )

    def test_initial_result_file_loads_fitted_parameters(self) -> None:
        first_transition_time = 2.220446049250313e-15
        result = OptimizationResult(
            log_likelihood=-5.0,
            birth_rates=torch.tensor([1.0, 1.2]),
            death_rates=torch.tensor([0.2, 0.3]),
            sampling_probabilities=torch.tensor([0.05, 0.05]),
            transition_probabilities=torch.tensor([[0.7, 0.3], [0.4, 0.6]]),
            transition_times=torch.tensor([first_transition_time, 0.8], dtype=torch.float64),
        )

        with TemporaryDirectory() as directory:
            result_path = Path(directory) / "initial.out"
            write_result(str(result_path), result, num_states=2)

            loaded = _load_initial_result(
                result_path,
                num_states=2,
                free_edges=[(0, 1), (1, 0)],
                transition_times=[first_transition_time, 0.8],
            )
            with self.assertRaisesRegex(ValueError, "do not match"):
                _load_initial_result(
                    result_path,
                    num_states=2,
                    free_edges=[(0, 1), (1, 0)],
                    transition_times=[0.2, 0.7],
                )

        self.assertEqual(loaded.log_likelihood, -5.0)
        self.assertEqual(loaded.birth_rates.tolist(), [1.0, 1.2])
        self.assertEqual(loaded.transition_probabilities.tolist(), [[0.7, 0.3], [0.4, 0.6]])
        self.assertEqual(loaded.transition_times.tolist(), [first_transition_time, 0.8])

    def test_fixed_sampling_probability_is_passed_to_parameter_fit(self) -> None:
        result = SimpleNamespace(
            log_likelihood=torch.tensor(1.0),
            birth_rates=torch.tensor([1.0, 1.0]),
            death_rates=torch.tensor([0.1, 0.1]),
            sampling_probabilities=torch.tensor([0.5, 0.5]),
            transition_probabilities=torch.eye(2),
        )

        with (
            patch(
                "src.progressive_joint_transition_times._build_datastructures",
                return_value=[],
            ),
            patch(
                "src.progressive_joint_transition_times.optimize_parameters",
                return_value=result,
            ) as optimize,
        ):
            _fit_parameters(
                trees=[],
                transition_times=[0.5],
                num_states=2,
                free_edges=[],
                device=torch.device("cpu"),
                fixed_sampling_probability=0.05,
                result=result,
                optimizer_name="adam",
                learning_rate=0.01,
                max_steps=1,
                patience=1,
                normalize_root_likelihood=True,
                logfile=None,
                verbose=False,
            )

        self.assertEqual(
            optimize.call_args.kwargs["given_sampling_probabilities"],
            [0.05, 0.05],
        )
        self.assertNotIn(
            "initial_sampling_probabilities",
            optimize.call_args.kwargs,
        )

    def test_fixed_sampling_probability_is_used_during_time_search(self) -> None:
        with (
            patch(
                "src.main2_joint_transition_times._build_datastructures",
                return_value=[],
            ),
            patch(
                "src.main2_joint_transition_times.PhylodynamicParameters"
            ) as parameters,
            patch(
                "src.main2_joint_transition_times.calculate_total_log_likelihood",
                return_value=torch.tensor(-1.0),
            ),
        ):
            _fixed_parameter_likelihood(
                trees=[],
                transition_times=[0.5],
                num_states=2,
                free_edges=[],
                result=None,
                device=torch.device("cpu"),
                normalize_root_likelihood=True,
                fixed_sampling_probability=0.05,
            )

        self.assertEqual(
            parameters.call_args.kwargs["given_sampling_probabilities"],
            [0.05, 0.05],
        )

    def test_seed_schedule_is_refined_before_progressive_additions(self) -> None:
        fitted_times: list[tuple[float, ...]] = []
        result = SimpleNamespace(
            log_likelihood=torch.tensor(1.0),
            birth_rates=torch.tensor([1.0]),
            death_rates=torch.tensor([0.1]),
            sampling_probabilities=torch.tensor([0.05]),
            transition_probabilities=torch.tensor([[1.0]]),
        )

        def fake_fit(_trees, times, *_args, **_kwargs):
            fitted_times.append(tuple(times))
            return result

        with (
            patch(
                "src.progressive_joint_transition_times._fixed_parameter_likelihood",
                return_value=1.0,
            ),
            patch(
                "src.progressive_joint_transition_times._find_finite_initial_schedule",
                side_effect=AssertionError("seed schedule should be used"),
            ),
            patch(
                "src.progressive_joint_transition_times._fit_parameters",
                side_effect=fake_fit,
            ),
            patch(
                "src.progressive_joint_transition_times._maximize_coordinate",
                side_effect=lambda times, **_kwargs: (list(times), 1.0),
            ),
            patch(
                "src.progressive_joint_transition_times._add_transition_at_maximum",
                return_value=([0.2, 0.4, 0.6], 1.0),
            ),
        ):
            optimize_progressive_joint_transition_times(
                trees=[],
                num_states=1,
                free_edges=[],
                num_transition_times=3,
                max_time=1.0,
                device=torch.device("cpu"),
                fixed_sampling_probability=0.05,
                initial_transition_times=[0.2, 0.6],
                grid_size=10,
            )

        self.assertEqual(fitted_times[0], (0.2, 0.6))
        self.assertEqual(fitted_times[-1], (0.2, 0.4, 0.6))

    def test_nonfinite_grid_candidates_are_skipped(self) -> None:
        def objective(proposal: list[float]) -> float:
            return float("-inf") if proposal[0] < 0.3 else -abs(proposal[0] - 0.4)

        times, likelihood = _maximize_coordinate(
            [0.5],
            coordinate=0,
            objective=objective,
            max_time=1.0,
            grid_size=1000,
        )

        self.assertEqual(times, [0.4])
        self.assertEqual(likelihood, 0.0)

    def test_progressive_optimizer_fits_parameters_at_each_transition_count(self) -> None:
        fitted_times: list[tuple[float, ...]] = []
        added_schedules = iter(([0.2, 0.6], [0.2, 0.4, 0.6]))
        result = SimpleNamespace(
            log_likelihood=torch.tensor(1.0),
            birth_rates=torch.tensor([1.0]),
            death_rates=torch.tensor([0.1]),
            sampling_probabilities=torch.tensor([0.5]),
            transition_probabilities=torch.tensor([[1.0]]),
        )

        def fake_fit(_trees, times, *_args, **_kwargs):
            self.assertEqual(_args[3], 0.05)
            fitted_times.append(tuple(times))
            return result

        with (
            patch(
                "src.progressive_joint_transition_times._find_finite_initial_schedule",
                return_value=[0.6],
            ),
            patch(
                "src.progressive_joint_transition_times._fit_parameters",
                side_effect=fake_fit,
            ),
            patch(
                "src.progressive_joint_transition_times._maximize_coordinate",
                side_effect=lambda times, **_kwargs: (list(times), 1.0),
            ),
            patch(
                "src.progressive_joint_transition_times._add_transition_at_maximum",
                side_effect=lambda *_args, **_kwargs: (next(added_schedules), 1.0),
            ),
        ):
            optimize_progressive_joint_transition_times(
                trees=[],
                num_states=1,
                free_edges=[],
                num_transition_times=3,
                max_time=1.0,
                device=torch.device("cpu"),
                fixed_sampling_probability=0.05,
                grid_size=10,
            )

        self.assertEqual(
            fitted_times,
            [
                (0.6,),
                (0.6,),
                (0.2, 0.6),
                (0.2, 0.6),
                (0.2, 0.4, 0.6),
                (0.2, 0.4, 0.6),
            ],
        )

    def test_previous_result_is_used_for_seed_validation_and_fit(self) -> None:
        initial_result = OptimizationResult(
            log_likelihood=-1.0,
            birth_rates=torch.tensor([1.0]),
            death_rates=torch.tensor([0.1]),
            sampling_probabilities=torch.tensor([0.05]),
            transition_probabilities=torch.tensor([[1.0]]),
        )

        with (
            patch(
                "src.progressive_joint_transition_times._fixed_parameter_likelihood",
                return_value=1.0,
            ) as likelihood,
            patch(
                "src.progressive_joint_transition_times._fit_parameters",
                return_value=initial_result,
            ) as fit,
            patch(
                "src.progressive_joint_transition_times._maximize_coordinate",
                side_effect=lambda times, **_kwargs: (list(times), 1.0),
            ),
        ):
            optimize_progressive_joint_transition_times(
                trees=[],
                num_states=1,
                free_edges=[],
                num_transition_times=1,
                max_time=1.0,
                device=torch.device("cpu"),
                initial_transition_times=[0.5],
                initial_result=initial_result,
                grid_size=10,
            )

        self.assertIs(likelihood.call_args.kwargs["result"], initial_result)
        self.assertIs(fit.call_args.kwargs["result"], initial_result)

    def test_progressive_counts_write_parameter_and_time_checkpoints(self) -> None:
        result = OptimizationResult(
            log_likelihood=1.0,
            birth_rates=torch.tensor([1.0]),
            death_rates=torch.tensor([0.1]),
            sampling_probabilities=torch.tensor([0.05]),
            transition_probabilities=torch.tensor([[1.0]]),
        )
        optimizer_logs: list[str | None] = []

        def fake_fit(_trees, times, *_args, **_kwargs):
            optimizer_logs.append(_kwargs["logfile"])
            result.transition_times = torch.tensor(times, dtype=torch.float64)
            return result

        with TemporaryDirectory() as directory:
            checkpoint_directory = Path(directory)
            with (
                patch(
                    "src.progressive_joint_transition_times._fixed_parameter_likelihood",
                    return_value=1.0,
                ),
                patch(
                    "src.progressive_joint_transition_times._fit_parameters",
                    side_effect=fake_fit,
                ),
                patch(
                    "src.progressive_joint_transition_times._maximize_coordinate",
                    side_effect=lambda times, **_kwargs: (list(times), 1.0),
                ),
                patch(
                    "src.progressive_joint_transition_times._add_transition_at_maximum",
                    return_value=([0.25, 0.5], 1.0),
                ),
            ):
                optimize_progressive_joint_transition_times(
                    trees=[],
                    num_states=1,
                    free_edges=[],
                    num_transition_times=2,
                    max_time=1.0,
                    device=torch.device("cpu"),
                    initial_transition_times=[2.220446049250313e-15],
                    grid_size=10,
                    checkpoint_directory=checkpoint_directory,
                    logfile=str(checkpoint_directory / "stage.log"),
                )

            for count, expected_times in (
                (1, [2.220446049250313e-15]),
                (2, [0.25, 0.5]),
            ):
                self.assertTrue(
                    (checkpoint_directory / f"transition_count_{count}.out").is_file()
                )
                time_path = checkpoint_directory / f"transition_count_{count}_times.txt"
                self.assertEqual(
                    [float(line) for line in time_path.read_text().splitlines()],
                    expected_times,
                )
            self.assertEqual(
                optimizer_logs,
                [
                    None,
                    str(checkpoint_directory / "transition_count_1.log"),
                    None,
                    str(checkpoint_directory / "stage.log"),
                ],
            )


if __name__ == "__main__":
    unittest.main()
