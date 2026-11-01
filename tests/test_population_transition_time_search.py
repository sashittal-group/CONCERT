from __future__ import annotations

from contextlib import redirect_stdout
import inspect
from io import StringIO
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.main2_joint_transition_times import (
    _find_finite_initial_schedule,
    _project_transition_times_to_grid,
)
from src.population_transition_time_search import (
    ScheduleCandidate,
    _crossover_schedule,
    _fit_candidate,
    _mutate_schedule,
    _parse_initial_output,
    _project_schedule,
    _random_schedule,
    _schedule_key,
    _unique_crossover_schedule,
    population_search,
)


class PopulationTransitionTimeSearchTests(unittest.TestCase):
    def test_parse_initial_output(self) -> None:
        output = """\
LL=1.0
birth: 1 2
death: 0.1 0.2
sampling: 0.5 0.6
transitions:
0->0 0.7
0->1 0.3
1->1 1
transition_times:
0 0.2
1 0.8
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "inference.out"
            path.write_text(output)

            times, parameters = _parse_initial_output(path, num_states=2)

        self.assertEqual(times, (0.2, 0.8))
        self.assertEqual(parameters["initial_birth_rates"], [1.0, 2.0])
        self.assertEqual(
            parameters["initial_transition_probabilities"],
            [[0.7, 0.3], [0.0, 1.0]],
        )

    def test_project_mutate_and_crossover_preserve_schedule_constraints(self) -> None:
        rng = random.Random(37)
        minimum_gap = 1e-4
        parent = _project_schedule([0.0, 0.2, 0.2, 0.9, 1.0], 1.0, minimum_gap)
        random_schedule = _random_schedule(rng, 5, 1.0, minimum_gap)
        mutated = _mutate_schedule(parent, rng, 1.0, minimum_gap, 0.15)
        crossed = _crossover_schedule(
            parent,
            (0.1, 0.3, 0.5, 0.7, 0.9),
            rng,
            1.0,
            minimum_gap,
        )

        for schedule in (parent, random_schedule, mutated, crossed):
            self.assertTrue(all(0.0 < time < 1.0 for time in schedule))
            self.assertTrue(
                all(
                    right - left >= minimum_gap
                    for left, right in zip(schedule, schedule[1:])
                )
            )
            for time in schedule:
                self.assertAlmostEqual(time * 1000, round(time * 1000))

    def test_schedule_generation_supports_coarse_grid_resolutions(self) -> None:
        projected = _project_transition_times_to_grid(
            [0.11, 0.22, 0.45, 0.7, 0.89],
            max_time=1.0,
            min_gap=1e-4,
            grid_size=25,
        )
        sampled = _random_schedule(
            random.Random(37),
            num_times=5,
            max_time=1.0,
            min_gap=1e-4,
            grid_resolution=25,
        )

        for schedule in (projected, sampled):
            self.assertTrue(all(0.0 < time < 1.0 for time in schedule))
            self.assertTrue(
                all(
                    right > left
                    for left, right in zip(schedule, schedule[1:])
                )
            )
            for time in schedule:
                self.assertAlmostEqual(time * 25, round(time * 25))

    def test_unique_crossover_reports_exhausted_and_uses_unseen_children(self) -> None:
        rng = random.Random(37)
        left = (0.034, 0.102, 0.148, 0.454, 0.622)
        right = (0.034, 0.147, 0.148, 0.454, 0.622)
        seen = {_schedule_key(left, 1.0), _schedule_key(right, 1.0)}

        self.assertIsNone(
            _unique_crossover_schedule(
                left,
                right,
                rng,
                max_time=1.0,
                min_gap=1e-4,
                seen_schedules=seen,
            )
        )

        other = (0.2, 0.4, 0.6, 0.8, 0.9)
        unseen = _unique_crossover_schedule(
            left,
            other,
            rng,
            max_time=1.0,
            min_gap=1e-4,
            seen_schedules=seen,
        )
        self.assertIsNotNone(unseen)
        self.assertNotIn(_schedule_key(unseen, 1.0), seen)

    def test_population_defaults_use_priority_queue_search_settings(self) -> None:
        parameters = inspect.signature(population_search).parameters

        self.assertEqual(parameters["grid_resolutions"].default, (25, 100, 250, 1000))
        self.assertEqual(parameters["stage_budgets"].default, (100, 200, 300, 400))
        self.assertEqual(parameters["parent_cooldown"].default, 5)
        self.assertEqual(parameters["crossover_probability"].default, 0.5)
        self.assertEqual(parameters["max_steps"].default, 100)
        self.assertEqual(parameters["max_total_steps"].default, 300)
        self.assertEqual(parameters["patience"].default, 30)
        self.assertEqual(parameters["plateau_tolerance"].default, 1e-4)

    def test_staged_search_keeps_unique_schedules_and_frozen_parents(self) -> None:
        fit_calls = []

        def fake_fit(
            candidate,
            _trees,
            _num_states,
            _free_edges,
            _device,
            initial_parameters,
            max_steps,
            _patience,
            *,
            max_total_steps,
            improvement_tolerance,
            verbose=False,
        ):
            fit_calls.append((candidate.schedule_id, initial_parameters, verbose))
            candidate.optimization_steps += max_steps
            candidate.fit_blocks += 1
            if candidate.schedule_id == 7 and candidate.fit_blocks == 1:
                candidate.fit_status = "invalid"
                candidate.fit_error = "Non-finite likelihood at step 1"
                return
            candidate.score = 100.0 - candidate.schedule_id
            candidate.parameters = {
                "initial_birth_rates": [candidate.schedule_id]
            }
            if candidate.schedule_id == 1 and candidate.fit_blocks == 1:
                candidate.fit_status = "frozen"
                candidate.freeze_reason = "plateau"
            elif candidate.optimization_steps >= max_total_steps:
                candidate.fit_status = "frozen"
                candidate.freeze_reason = "step_cap"
            else:
                candidate.fit_status = "active"

        with tempfile.TemporaryDirectory() as temporary_directory:
            queue_log_path = Path(temporary_directory) / "queue.jsonl"
            with patch(
                "src.population_transition_time_search._fit_candidate",
                side_effect=fake_fit,
            ), redirect_stdout(StringIO()):
                result = population_search(
                    trees=[],
                    num_states=4,
                    free_edges=[],
                    initial_times=(0.1, 0.3, 0.5, 0.7, 0.9),
                    initial_parameters={"initial_birth_rates": [1.0]},
                    grid_resolutions=(25, 100),
                    stage_budgets=(7, 8),
                    parent_cooldown=5,
                    crossover_probability=0.0,
                    max_steps=2,
                    max_total_steps=4,
                    random_seed=37,
                    queue_log_path=queue_log_path,
                )
            queue_events = [
                json.loads(line)
                for line in queue_log_path.read_text().splitlines()
            ]

        history = result["schedule_history"]
        self.assertEqual(len(history), 15)
        self.assertEqual(result["unique_schedules_fitted"], 15)
        self.assertEqual([item["schedule_id"] for item in history], list(range(1, 16)))
        self.assertEqual(len({_schedule_key(tuple(item["transition_times"]), 1.0) for item in history}), 15)
        self.assertGreater(len(fit_calls), 15)
        self.assertEqual(history[6]["fit_status"], "invalid")
        self.assertIn("Non-finite likelihood", history[6]["fit_error"])
        self.assertEqual(
            [item["grid_resolution"] for item in history[:7]],
            [25] * 7,
        )
        self.assertEqual(
            [item["grid_resolution"] for item in history[7:]],
            [100] * 8,
        )
        self.assertTrue(
            all(item["operator"].startswith("initial_") for item in history[:7])
        )
        self.assertTrue(
            all(
                parent_id < item["schedule_id"]
                for item in history[10:]
                for parent_id in item["parent_ids"]
            )
        )
        self.assertNotIn(
            7,
            [
                parent_id
                for item in history[7:]
                for parent_id in item["parent_ids"]
            ],
        )
        self.assertTrue(all(item["operator"] == "mutation" for item in history[7:]))
        self.assertIn(1, [parent_id for item in history[7:] for parent_id in item["parent_ids"]])
        self.assertEqual(sum(call[0] == 1 for call in fit_calls), 1)
        self.assertIn(1, result["frozen_schedule_ids"])
        self.assertEqual(queue_events[0]["event"], "SEARCH_STARTED")
        proposed_event = next(
            event for event in queue_events if event["event"] == "SCHEDULE_PROPOSED"
        )
        self.assertEqual(proposed_event["schedule"]["queue_state"], "pending")
        self.assertTrue(
            any(event["event"] == "PARENT_SELECTED" for event in queue_events)
        )
        self.assertTrue(
            any(
                event["event"] == "PARENT_REQUEUED"
                and event["schedule"]["queue_state"] == "waiting"
                and event["schedule"]["log_likelihood"] is not None
                for event in queue_events
            )
        )
        self.assertTrue(any(event["event"] == "QUEUE_READY" for event in queue_events))
        final_snapshot = next(
            event
            for event in queue_events
            if event["event"] == "QUEUE_SNAPSHOT"
            and event["reason"] == "stage_2_completed"
        )
        snapshot_ids = {item["schedule_id"] for item in final_snapshot["items"]}
        self.assertEqual(snapshot_ids, set(range(1, 16)))
        invalid_record = next(item for item in final_snapshot["items"] if item["schedule_id"] == 7)
        self.assertEqual(invalid_record["queue_state"], "invalid")
        self.assertIsNone(invalid_record["log_likelihood"])

        parent_usage = {}
        for item in history[7:]:
            for parent_id in item["parent_ids"]:
                previous_use = parent_usage.get(parent_id)
                if previous_use is not None:
                    self.assertGreaterEqual(
                        item["schedule_id"] - previous_use,
                        5,
                    )
                parent_usage[parent_id] = item["schedule_id"]

    def test_exhausted_crossover_falls_back_to_mutation(self) -> None:
        def fake_fit(
            candidate,
            _trees,
            _num_states,
            _free_edges,
            _device,
            _initial_parameters,
            max_steps,
            _patience,
            *,
            max_total_steps,
            improvement_tolerance,
            verbose=False,
        ):
            candidate.score = 100.0 - candidate.schedule_id
            candidate.parameters = {
                "initial_birth_rates": [candidate.schedule_id]
            }
            candidate.optimization_steps += max_steps
            candidate.fit_blocks += 1
            if candidate.optimization_steps >= max_total_steps:
                candidate.fit_status = "frozen"
                candidate.freeze_reason = "step_cap"
            else:
                candidate.fit_status = "active"

        with tempfile.TemporaryDirectory() as temporary_directory:
            queue_log_path = Path(temporary_directory) / "queue.jsonl"
            with (
                patch(
                    "src.population_transition_time_search._fit_candidate",
                    side_effect=fake_fit,
                ),
                patch(
                    "src.population_transition_time_search._unique_crossover_schedule",
                    return_value=None,
                ),
                redirect_stdout(StringIO()),
            ):
                result = population_search(
                    trees=[],
                    num_states=4,
                    free_edges=[],
                    initial_times=(0.1, 0.3, 0.5, 0.7, 0.9),
                    initial_parameters={},
                    grid_resolutions=(25, 100),
                    stage_budgets=(3, 1),
                    crossover_probability=1.0,
                    random_seed=37,
                    queue_log_path=queue_log_path,
                )
            queue_events = [
                json.loads(line)
                for line in queue_log_path.read_text().splitlines()
            ]

        child = result["schedule_history"][-1]
        self.assertEqual(child["operator"], "mutation")
        self.assertEqual(len(child["parent_ids"]), 1)
        self.assertTrue(
            any(event["event"] == "CROSSOVER_EXHAUSTED" for event in queue_events)
        )
        self.assertTrue(
            any(
                event["event"] == "QUEUE_ADD"
                and event["schedule"]["schedule_id"] == 2
                for event in queue_events
            )
        )
        self.assertTrue(
            any(
                event["event"] == "PARENT_REQUEUED"
                and event.get("reason") == "crossover_space_exhausted"
                for event in queue_events
            )
        )

    def test_fit_reuses_cut_trees_during_independent_optimization(self) -> None:
        class FakeParameters(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.birth_rates = torch.nn.Parameter(torch.zeros(2))
                self.death_rates = torch.nn.Parameter(torch.zeros(2))
                self.sampling_probabilities = torch.nn.Parameter(torch.zeros(2))
                self.transition_probabilities = torch.nn.Parameter(torch.zeros(2, 2))

        class FakeLikelihood(torch.nn.Module):
            def __init__(self, params):
                super().__init__()
                self.params = params

            def forward(self):
                return self.params.birth_rates.sum()

        params = FakeParameters()
        candidate = ScheduleCandidate(
            schedule_id=1,
            times=(0.1, 0.3),
            operator="initial_incumbent",
        )
        with (
            patch(
                "src.population_transition_time_search.PhylodynamicParameters",
                return_value=params,
            ),
            patch(
                "src.population_transition_time_search._build_datastructures",
                return_value=[],
            ) as build_datastructures,
            patch(
                "src.population_transition_time_search.LikelihoodModel",
                side_effect=lambda **kwargs: FakeLikelihood(kwargs["params"]),
            ),
        ):
            _fit_candidate(
                candidate,
                trees=[],
                num_states=2,
                free_edges=[],
                device=torch.device("cpu"),
                initial_parameters=None,
                max_steps=5,
                patience=10,
                max_total_steps=7,
            )

            _fit_candidate(
                candidate,
                trees=[],
                num_states=2,
                free_edges=[],
                device=torch.device("cpu"),
                initial_parameters=None,
                max_steps=5,
                patience=10,
                max_total_steps=7,
            )

        self.assertEqual(build_datastructures.call_count, 1)
        self.assertEqual(candidate.optimization_steps, 7)
        self.assertEqual(candidate.fit_blocks, 2)
        self.assertEqual(candidate.fit_status, "frozen")
        self.assertEqual(candidate.freeze_reason, "step_cap")
        self.assertIsNone(candidate.cached_datastructures)
        self.assertIsNotNone(candidate.parameters)

    def test_nonfinite_initial_likelihood_invalidates_only_that_schedule(self) -> None:
        class FakeParameters(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.birth_rates = torch.nn.Parameter(torch.zeros(2))
                self.death_rates = torch.nn.Parameter(torch.zeros(2))
                self.sampling_probabilities = torch.nn.Parameter(torch.zeros(2))
                self.transition_probabilities = torch.nn.Parameter(torch.zeros(2, 2))

        class NonfiniteLikelihood(torch.nn.Module):
            def forward(self):
                return torch.tensor(float("nan"))

        candidate = ScheduleCandidate(
            schedule_id=1,
            times=(0.1, 0.3),
            operator="initial_incumbent",
        )
        with (
            patch(
                "src.population_transition_time_search.PhylodynamicParameters",
                return_value=FakeParameters(),
            ),
            patch(
                "src.population_transition_time_search._build_datastructures",
                return_value=[],
            ),
            patch(
                "src.population_transition_time_search.LikelihoodModel",
                return_value=NonfiniteLikelihood(),
            ),
        ):
            _fit_candidate(
                candidate,
                trees=[],
                num_states=2,
                free_edges=[],
                device=torch.device("cpu"),
                initial_parameters=None,
                max_steps=5,
                patience=10,
            )

        self.assertEqual(candidate.fit_status, "invalid")
        self.assertEqual(candidate.optimization_steps, 1)
        self.assertEqual(candidate.score, float("-inf"))
        self.assertIsNone(candidate.parameters)
        self.assertIn("step 1", candidate.fit_error)

    def test_plateau_freezes_schedule_after_consecutive_small_improvements(self) -> None:
        class FakeParameters(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.birth_rates = torch.nn.Parameter(torch.zeros(2))
                self.death_rates = torch.nn.Parameter(torch.zeros(2))
                self.sampling_probabilities = torch.nn.Parameter(torch.zeros(2))
                self.transition_probabilities = torch.nn.Parameter(torch.zeros(2, 2))

        class ConstantLikelihood(torch.nn.Module):
            def __init__(self, params):
                super().__init__()
                self.params = params

            def forward(self):
                return self.params.birth_rates.sum() * 0

        candidate = ScheduleCandidate(
            schedule_id=1,
            times=(0.1, 0.3),
            operator="initial_incumbent",
        )
        with (
            patch(
                "src.population_transition_time_search.PhylodynamicParameters",
                return_value=FakeParameters(),
            ),
            patch(
                "src.population_transition_time_search._build_datastructures",
                return_value=[],
            ) as build_datastructures,
            patch(
                "src.population_transition_time_search.LikelihoodModel",
                side_effect=lambda **kwargs: ConstantLikelihood(kwargs["params"]),
            ),
        ):
            _fit_candidate(
                candidate,
                trees=[],
                num_states=2,
                free_edges=[],
                device=torch.device("cpu"),
                initial_parameters=None,
                max_steps=10,
                patience=3,
                max_total_steps=30,
                improvement_tolerance=1e-4,
            )

        self.assertEqual(candidate.optimization_steps, 4)
        self.assertEqual(candidate.fit_status, "frozen")
        self.assertEqual(candidate.freeze_reason, "plateau")
        self.assertEqual(candidate.steps_without_improvement, 3)
        self.assertIsNone(candidate.cached_datastructures)
        self.assertEqual(build_datastructures.call_count, 1)

    def test_nonfinite_likelihood_after_finite_steps_keeps_best_fit(self) -> None:
        class FakeParameters(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.birth_rates = torch.nn.Parameter(torch.zeros(2))
                self.death_rates = torch.nn.Parameter(torch.zeros(2))
                self.sampling_probabilities = torch.nn.Parameter(torch.zeros(2))
                self.transition_probabilities = torch.nn.Parameter(torch.zeros(2, 2))

        class IntermittentLikelihood(torch.nn.Module):
            def __init__(self, params):
                super().__init__()
                self.params = params
                self.calls = 0

            def forward(self):
                self.calls += 1
                if self.calls == 1:
                    return self.params.birth_rates.sum()
                return torch.tensor(float("nan"))

        params = FakeParameters()
        candidate = ScheduleCandidate(
            schedule_id=1,
            times=(0.1, 0.3),
            operator="initial_incumbent",
        )
        with (
            patch(
                "src.population_transition_time_search.PhylodynamicParameters",
                return_value=params,
            ),
            patch(
                "src.population_transition_time_search._build_datastructures",
                return_value=[],
            ),
            patch(
                "src.population_transition_time_search.LikelihoodModel",
                side_effect=lambda **kwargs: IntermittentLikelihood(kwargs["params"]),
            ),
        ):
            _fit_candidate(
                candidate,
                trees=[],
                num_states=2,
                free_edges=[],
                device=torch.device("cpu"),
                initial_parameters=None,
                max_steps=5,
                patience=10,
            )

        self.assertEqual(candidate.fit_status, "frozen")
        self.assertEqual(candidate.freeze_reason, "nonfinite")
        self.assertEqual(candidate.optimization_steps, 2)
        self.assertEqual(candidate.score, 0.0)
        self.assertIsNotNone(candidate.parameters)
        self.assertIn("step 2", candidate.fit_error)

    def test_joint_optimizer_initial_schedule_uses_grid(self) -> None:
        with patch(
            "src.main2_joint_transition_times._fixed_parameter_likelihood",
            side_effect=[float("-inf"), 1.0],
        ):
            schedule = _find_finite_initial_schedule(
                trees=[],
                num_states=4,
                free_edges=[],
                num_transition_times=5,
                max_time=1.0,
                device=None,
                normalize_root_likelihood=False,
                random_seed=37,
                max_attempts=1,
            )

        for time in schedule:
            self.assertAlmostEqual(time * 1000, round(time * 1000))


if __name__ == "__main__":
    unittest.main()
