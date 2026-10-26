from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from src.results.profile_tools import ParameterProfile
from src.results.tls.profile_transition_run import (
    _fixed_sampling_profiles,
    _read_log_likelihood,
)


class TlsTransitionProfileTests(unittest.TestCase):
    def test_fixed_sampling_is_kept_as_a_single_point(self) -> None:
        profiles = [
            ParameterProfile(
                "sampling[shared]",
                "sampling",
                0,
                0.1,
                np.asarray([0.01, 0.1, 0.99]),
            ),
            ParameterProfile(
                "birth[NMPs]",
                "birth",
                0,
                2.0,
                np.asarray([1.0, 2.0, 3.0]),
            ),
        ]

        result = _fixed_sampling_profiles(profiles, np.asarray([0.1, 0.1, 0.1]))

        self.assertEqual(result[0].values.tolist(), [0.1])
        self.assertEqual(result[1], profiles[1])

    def test_fixed_sampling_requires_the_expected_probability(self) -> None:
        profile = ParameterProfile(
            "sampling[shared]",
            "sampling",
            0,
            0.2,
            np.asarray([0.01, 0.2, 0.99]),
        )

        with self.assertRaisesRegex(ValueError, "fixed sampling at 0.1"):
            _fixed_sampling_profiles([profile], np.asarray([0.2, 0.2, 0.2]))

    def test_log_likelihood_can_be_read_from_an_iteration_checkpoint(self) -> None:
        path = Path("checkpoint.tmp")
        path.write_text("ITER 4360 LL=-2803.639766855049402\n")
        try:
            self.assertAlmostEqual(_read_log_likelihood(path), -2803.639766855049402)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
