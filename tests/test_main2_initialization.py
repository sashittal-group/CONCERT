from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from src.main2 import read_initial_parameters
from src.mle_closed_form2 import PhylodynamicParameters


class Main2InitializationTests(unittest.TestCase):
    def test_loads_parameters_and_derives_transition_diagonals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "inference.tmp.out"
            path.write_text(
                "ITER 120 LL=-10\n"
                "birth: 5 6 7\n"
                "death: 1 2 3\n"
                "sampling: 0.1 0.1 0.1\n"
                "transitions:\n"
                "0->1 0.3\n"
                "0->2 0.2\n"
                "transition_times:\n"
                "0 0.0001\n"
            )

            birth, death, sampling, transitions = read_initial_parameters(
                path,
                num_states=3,
                free_edges=[(0, 1), (0, 2)],
            )

        self.assertEqual(birth, [5.0, 6.0, 7.0])
        self.assertEqual(death, [1.0, 2.0, 3.0])
        self.assertEqual(sampling, [0.1, 0.1, 0.1])
        self.assertEqual(
            transitions,
            [
                [0.5, 0.3, 0.2],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        )

    def test_rejects_initial_transition_edges_that_do_not_match_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "inference.tmp.out"
            path.write_text(
                "birth: 5 6 7\n"
                "death: 1 2 3\n"
                "sampling: 0.1 0.1 0.1\n"
                "transitions:\n"
                "0->1 0.3\n"
                "transition_times:\n"
                "0 0.0001\n"
            )

            with self.assertRaisesRegex(ValueError, "do not match"):
                read_initial_parameters(
                    path,
                    num_states=3,
                    free_edges=[(0, 1), (0, 2)],
                )

    def test_preserves_small_initial_birth_and_death_rates(self) -> None:
        expected_birth = torch.tensor([9.82136090221631e-8])
        expected_death = torch.tensor([4.00373276624008e-8])
        params = PhylodynamicParameters(
            num_states=1,
            free_edges=[],
            given_transition_times=[0.5],
            device=torch.device("cpu"),
            initial_birth_rates=expected_birth.tolist(),
            initial_death_rates=expected_death.tolist(),
            given_sampling_probabilities=[0.05],
        )

        torch.testing.assert_close(
            params.birth_rates,
            expected_birth,
            rtol=1e-5,
            atol=1e-12,
        )
        torch.testing.assert_close(
            params.death_rates,
            expected_death,
            rtol=1e-5,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
