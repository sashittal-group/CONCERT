from __future__ import annotations

from pathlib import Path
import random
import tempfile
import unittest

from src.results.sgootr.schedule_search import (
    GLOBAL_MUTATION_PROBABILITY,
    LOCAL_MUTATION_RADIUS,
    RANDOM_IMMIGRANT_FRACTION,
    ScheduleResult,
    _mutate_transition_time,
    _schedule_times,
    _write_summary,
    candidate_grid_indices,
    genetic_schedules,
    random_schedules,
)


class ScheduleSearchTests(unittest.TestCase):
    def test_grid_contains_only_times_strictly_before_root(self) -> None:
        indices = candidate_grid_indices(0.443103042059838)

        self.assertEqual(len(indices), 444)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 443)
        self.assertTrue(all(index / 1000 < 0.443103042059838 for index in indices))

    def test_grid_can_exclude_zero_time(self) -> None:
        indices = candidate_grid_indices(
            0.443103042059838,
            exclude_zero_time=True,
        )

        self.assertEqual(len(indices), 443)
        self.assertEqual(indices[0], 1)
        self.assertEqual(indices[-1], 443)
        self.assertTrue(all(index / 1000 > 0.0 for index in indices))

    def test_rejects_nonpositive_root_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "Root time"):
            candidate_grid_indices(0.0)

    def test_random_schedules_are_distinct_and_sorted(self) -> None:
        seen: set[tuple[int, int, int]] = set()

        schedules = random_schedules(list(range(20)), 12, random.Random(3), seen)

        self.assertEqual(len(schedules), 12)
        self.assertEqual(len(set(schedules)), 12)
        self.assertEqual(len(seen), 12)
        for schedule in schedules:
            self.assertEqual(len(schedule), 3)
            self.assertLess(schedule[0], schedule[1])
            self.assertLess(schedule[1], schedule[2])

    def test_genetic_schedules_use_elites_and_avoid_seen_schedules(self) -> None:
        output = Path("inference.txt")
        elites = [
            ScheduleResult(
                "schedule_0001",
                _schedule_times((1, 5, 10)),
                20.0,
                output,
                1,
            ),
            ScheduleResult(
                "schedule_0002",
                _schedule_times((2, 6, 11)),
                10.0,
                output,
                1,
            ),
        ]
        seen = {(1, 5, 10), (2, 6, 11)}

        children = genetic_schedules(
            elites,
            list(range(20)),
            15,
            random.Random(9),
            seen,
        )

        self.assertEqual(len(children), 15)
        self.assertEqual(len(set(children)), 15)
        for child in children:
            self.assertEqual(len(child), 3)
            self.assertLess(child[0], child[1])
            self.assertLess(child[1], child[2])
            self.assertTrue(all(index in range(20) for index in child))
            self.assertIn(child, seen)

    def test_genetic_schedules_include_random_immigrants(self) -> None:
        output = Path("inference.txt")
        elites = [
            ScheduleResult(
                "schedule_0001",
                _schedule_times((1, 5, 10)),
                20.0,
                output,
                1,
            ),
            ScheduleResult(
                "schedule_0002",
                _schedule_times((2, 6, 11)),
                10.0,
                output,
                1,
            ),
        ]
        seen = {(1, 5, 10), (2, 6, 11)}
        immigrant_count = 20

        children = genetic_schedules(
            elites,
            list(range(500)),
            100,
            random.Random(42),
            seen,
            random_immigrant_count=immigrant_count,
        )

        self.assertEqual(len(children), 100)
        self.assertTrue(
            all(max(schedule) > 100 for schedule in children[:immigrant_count])
        )
        self.assertEqual(RANDOM_IMMIGRANT_FRACTION, 0.20)

    def test_mutation_mixes_local_changes_and_global_jumps(self) -> None:
        rng = random.Random(12)
        candidate_indices = list(range(1000))

        mutated_indices = [
            _mutate_transition_time(500, candidate_indices, rng)
            for _ in range(1000)
        ]
        distances = [abs(index - 500) for index in mutated_indices]

        self.assertTrue(all(distance > 0 for distance in distances))
        self.assertGreater(
            sum(distance <= LOCAL_MUTATION_RADIUS for distance in distances),
            600,
        )
        self.assertTrue(any(distance > LOCAL_MUTATION_RADIUS for distance in distances))
        self.assertAlmostEqual(GLOBAL_MUTATION_PROBABILITY, 0.25)

    def test_grid_indices_convert_to_transition_times(self) -> None:
        self.assertEqual(_schedule_times((0, 43, 443)), (0.0, 0.043, 0.443))

    def test_summary_selects_highest_likelihood_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            results = [
                ScheduleResult(
                    "schedule_0001",
                    (0.1, 0.2, 0.3),
                    12.0,
                    output_dir / "schedule_0001" / "inference.txt",
                    1,
                ),
                ScheduleResult(
                    "schedule_0002",
                    (0.2, 0.3, 0.4),
                    15.0,
                    output_dir / "schedule_0002" / "inference.txt",
                    2,
                ),
            ]

            best = _write_summary(output_dir, results, [])

            self.assertIsNotNone(best)
            self.assertEqual(best.schedule_id, "schedule_0002")
            self.assertEqual(
                [
                    float(time)
                    for time in (output_dir / "best_schedule.txt")
                    .read_text()
                    .splitlines()
                ],
                [0.2, 0.3, 0.4],
            )


if __name__ == "__main__":
    unittest.main()
