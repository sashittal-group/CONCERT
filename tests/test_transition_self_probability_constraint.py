from __future__ import annotations

import unittest

import torch

from src.mle_closed_form2 import PhylodynamicParameters


class SelfTransitionProbabilityConstraintTests(unittest.TestCase):
    def _parameters(
        self,
        minimum_self_transition_probability: float | None = None,
    ) -> PhylodynamicParameters:
        return PhylodynamicParameters(
            num_states=3,
            free_edges=[(0, 1), (1, 0)],
            given_transition_times=[0.5],
            device=torch.device("cpu"),
            initial_transition_probabilities=[
                [0.8, 0.2, 0.0],
                [0.2, 0.8, 0.0],
                [0.0, 0.0, 1.0],
            ],
            minimum_self_transition_probability=(
                minimum_self_transition_probability
            ),
        )

    def test_constraint_keeps_every_self_probability_above_threshold(self) -> None:
        parameters = self._parameters(0.9)
        probabilities = parameters.transition_probabilities

        self.assertTrue(torch.all(probabilities.diagonal() > 0.9))
        torch.testing.assert_close(
            probabilities.sum(dim=1),
            torch.ones(3, dtype=probabilities.dtype),
        )
        self.assertTrue(torch.all(probabilities[0, 1:] >= 0.0))
        self.assertTrue(torch.all(probabilities[1, 1:] >= 0.0))
        self.assertTrue(torch.all(torch.isfinite(probabilities)))

        objective = probabilities[0, 1] + probabilities[1, 0]
        objective.backward()
        self.assertTrue(torch.all(torch.isfinite(parameters.transition_raw.grad)))

    def test_constraint_is_opt_in(self) -> None:
        probabilities = self._parameters().transition_probabilities

        torch.testing.assert_close(
            probabilities,
            torch.tensor(
                [
                    [0.8, 0.2, 0.0],
                    [0.2, 0.8, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=probabilities.dtype,
            ),
        )

    def test_threshold_must_be_less_than_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be finite"):
            self._parameters(1.0)


if __name__ == "__main__":
    unittest.main()
