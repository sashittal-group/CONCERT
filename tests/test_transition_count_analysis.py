from __future__ import annotations

import random
import unittest
from pathlib import Path

from src.results.sgootr.transition_count_analysis import (
    _cohort_label,
    _epoch_edges,
    _random_schedules,
)


class TransitionCountAnalysisTests(unittest.TestCase):
    def test_cohort_label_uses_input_crc(self) -> None:
        self.assertEqual(
            _cohort_label(Path("S10_CRC04_transition_chain_input")),
            "S10 CRC04",
        )
        self.assertEqual(
            _cohort_label(Path("S15_CRC11_transition_chain_input")),
            "S15 CRC11",
        )

    def test_epoch_edges_keep_one_nc_to_pt_and_repeat_pt_to_ln(self) -> None:
        self.assertEqual(_epoch_edges(1), ((1, 2), (0, 1)))
        self.assertEqual(
            _epoch_edges(5),
            ((1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (0, 1)),
        )

    def test_random_schedules_have_the_requested_size_and_unique_times(self) -> None:
        schedules = _random_schedules(
            list(range(1, 12)),
            schedule_size=6,
            count=20,
            rng=random.Random(18),
            seen=set(),
        )

        self.assertEqual(len(schedules), 20)
        self.assertEqual(len(set(schedules)), 20)
        for schedule in schedules:
            self.assertEqual(len(schedule), 6)
            self.assertEqual(tuple(sorted(schedule)), schedule)
            self.assertEqual(len(set(schedule)), 6)


if __name__ == "__main__":
    unittest.main()
