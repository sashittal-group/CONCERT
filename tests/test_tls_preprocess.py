from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.results.profile_tools import ParameterProfile
from src.results.tls.config import load_config
from src.results.tls.data import DatasetContext, load_inference_parameters
from src.results.tls.preprocess import build_parameter_profiles, preprocess_dataset


class TlsPreprocessTests(unittest.TestCase):
    def test_missing_diagonal_transition_probabilities_are_inferred(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            inference_path = Path(temporary_directory) / "inference.out"
            inference_path.write_text(
                "birth: 1 2 3\n"
                "death: 0.1 0.2 0.3\n"
                "sampling: 0.1 0.1 0.1\n"
                "transitions:\n"
                "0->1 0.2\n"
                "0->2 0.3\n"
                "1->2 0.4\n"
                "2->1 0.1\n"
                "transition_times:\n"
                "0 0.1\n"
            )

            parameters = load_inference_parameters(
                inference_path,
                ("NMPs", "Somite", "NeuralTube"),
            )

        self.assertTrue(
            torch.allclose(
                parameters.transition_probabilities,
                torch.tensor(
                    [
                        [0.5, 0.2, 0.3],
                        [0.0, 0.6, 0.4],
                        [0.0, 0.1, 0.9],
                    ],
                    dtype=torch.float64,
                ),
            )
        )

    def test_fixed_sampling_config_keeps_inference_and_outputs_separate(self) -> None:
        config = load_config(
            Path("src/results/tls/config_fixed_sampling_0.1.yaml")
        )

        self.assertEqual(
            config.dataset_inference_path("TLS"),
            Path(
                "src/results/tls/fixed_sampling_0.1/inference/TLS/inference.out"
            ).resolve(),
        )
        self.assertEqual(
            config.dataset_output_dir("TLS"),
            Path("src/results/tls/fixed_sampling_0.1/profiles/TLS").resolve(),
        )
        self.assertEqual(
            config.figure_root,
            Path("src/results/tls/fixed_sampling_0.1/figures").resolve(),
        )

    def test_preprocess_writes_shared_profile_rows(self) -> None:
        base_config = load_config(Path("src/results/tls/config.yaml"))
        context = DatasetContext(
            birth_rates=torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64),
            death_rates=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
            sampling_probabilities=torch.tensor(
                [0.4, 0.4, 0.4],
                dtype=torch.float64,
            ),
            transition_probabilities=torch.tensor(
                [[0.7, 0.3, 0.0], [0.0, 0.8, 0.2], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
            transition_times=[],
            sensitivity_inputs=[],
        )
        profile = ParameterProfile(
            "birth[NMPs]",
            "birth",
            0,
            4.0,
            np.array([3.5, 4.0, 4.5]),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            config = replace(
                base_config,
                data_root=temporary_path / "input",
                preprocessing_root=temporary_path / "output",
            )
            with (
                patch(
                    "src.results.tls.preprocess.load_dataset_context",
                    return_value=context,
                ),
                patch(
                    "src.results.tls.preprocess.build_parameter_profiles",
                    return_value=[profile],
                ),
                patch(
                    "src.results.tls.preprocess.evaluate_total_log_likelihood",
                    side_effect=[-0.1, -0.1, -0.2, -0.5],
                ),
            ):
                preprocess_dataset("TLS", config)

            profile_path = (
                config.dataset_output_dir("TLS") / config.profile_filename
            )
            rows = pd.read_csv(profile_path)
            self.assertEqual(len(rows), 3)
            self.assertEqual(set(rows["parameter"]), {"birth[NMPs]"})
            self.assertEqual(set(rows["dataset"]), {"TLS"})

    def test_death_profile_grid_includes_values_below_rate_epsilon(self) -> None:
        config = load_config(Path("src/results/tls/config_fixed_sampling_0.1.yaml"))
        context = DatasetContext(
            birth_rates=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
            death_rates=torch.tensor([9.0e-9, 0.5, 1.0], dtype=torch.float64),
            sampling_probabilities=torch.tensor(
                [0.1, 0.1, 0.1],
                dtype=torch.float64,
            ),
            transition_probabilities=torch.tensor(
                [[0.9, 0.05, 0.05], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
            transition_times=[0.1],
            sensitivity_inputs=[],
        )

        profiles = build_parameter_profiles(context, config)
        death_profile = next(
            profile for profile in profiles if profile.name == "death[NMPs]"
        )

        self.assertIn(9.0e-9, death_profile.values)


if __name__ == "__main__":
    unittest.main()
