from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import torch

from src.results.sgootr.preprocess import (
    _birth_ratio_profiles,
    _candidate_parameters,
    _growth_profiles,
    _sampling_ratio_profiles,
    _sampling_profiles,
    _transition_profiles,
    build_parameter_profiles,
)
from src.results.tls.config import load_config
from src.results.tls.data import DatasetContext
from src.results.tls.plot import (
    _dataset_offsets,
    _parameter_group_label,
    _profile_box_color,
    _render_grouped_sampling_profiles,
)
from src.results.sgootr.profile_combined_inference import (
    _cohort_name,
    _plot_config,
    _plot_ratio_profiles,
    _plot_ratio_profile_intervals,
    _plot_sampling_ratio_profile_intervals_by_tree,
)


class SgootrProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(
            Path("src/results/sgootr/config.yaml"),
            section="sgootr_profiles",
        )
        self.context = DatasetContext(
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
            transition_times=[0.25, 0.75],
            sensitivity_inputs=[],
        )

    def test_profiles_include_observed_sgootr_transition_edges(self) -> None:
        profiles = build_parameter_profiles(self.context, self.config)
        transition_names = [
            profile.name
            for profile in profiles
            if profile.parameter_type == "transition"
        ]
        self.assertEqual(
            transition_names,
            ["transition[NC->PT]", "transition[PT->LN]"],
        )
        self.assertEqual(
            sum(profile.parameter_type == "transition_ratio" for profile in profiles),
            0,
        )

    def test_sampling_ratio_profiles_hold_pair_totals_constant(self) -> None:
        profiles = _sampling_ratio_profiles(self.context, self.config)

        self.assertEqual(
            [profile.name for profile in profiles],
            ["sampling_ratio[PT/NC]", "sampling_ratio[LN/NC]"],
        )
        for profile in profiles:
            candidate = 2.0
            _, _, sampling, _ = _candidate_parameters(
                self.context,
                profile,
                candidate,
                self.config,
            )
            self.assertAlmostEqual(
                float(sampling[profile.index] / sampling[0]),
                candidate,
            )
            self.assertAlmostEqual(
                float(sampling[profile.index] + sampling[0]),
                float(
                    self.context.sampling_probabilities[profile.index]
                    + self.context.sampling_probabilities[0]
                ),
            )

    def test_per_tree_sampling_ratio_profiles_change_only_target_pair(self) -> None:
        context = replace(
            self.context,
            sampling_probabilities=torch.tensor(
                [[0.4, 0.2, 0.1], [0.3, 0.15, 0.05]],
                dtype=torch.float64,
            ),
            tree_sampling_scope="per_tree_by_state",
        )

        profiles = _sampling_ratio_profiles(context, self.config)

        self.assertEqual(len(profiles), 4)
        profile = profiles[2]
        self.assertEqual(profile.name, "sampling_ratio[tree_1,PT/NC]")
        candidate = 2.0
        _, _, sampling, _ = _candidate_parameters(
            context,
            profile,
            candidate,
            self.config,
        )
        self.assertAlmostEqual(float(sampling[1, 1] / sampling[1, 0]), candidate)
        self.assertAlmostEqual(
            float(sampling[1, 0] + sampling[1, 1]),
            float(context.sampling_probabilities[1, 0] + context.sampling_probabilities[1, 1]),
        )
        self.assertTrue(
            torch.equal(sampling[0], context.sampling_probabilities[0])
        )
        self.assertEqual(float(sampling[1, 2]), float(context.sampling_probabilities[1, 2]))

    def test_birth_ratio_profiles_hold_pair_totals_constant(self) -> None:
        profiles = _birth_ratio_profiles(self.context, self.config)

        self.assertEqual(
            [profile.name for profile in profiles],
            ["birth_ratio[PT/NC]", "birth_ratio[LN/NC]"],
        )
        for profile in profiles:
            candidate = 2.0
            birth, _, _, _ = _candidate_parameters(
                self.context,
                profile,
                candidate,
                self.config,
            )
            self.assertAlmostEqual(float(birth[profile.index] / birth[0]), candidate)
            self.assertAlmostEqual(
                float(birth[profile.index] + birth[0]),
                float(
                    self.context.birth_rates[profile.index]
                    + self.context.birth_rates[0]
                ),
            )
            self.assertGreater(float(birth[0]), float(self.context.death_rates[0]))
            self.assertGreater(
                float(birth[profile.index]),
                float(self.context.death_rates[profile.index]),
            )

    def test_transition_profiles_adjust_each_edge_independently(self) -> None:
        profiles = [
            profile
            for profile in build_parameter_profiles(self.context, self.config)
            if profile.parameter_type == "transition"
        ]
        for profile in profiles:
            candidate = profile.original_value * 0.5
            _, _, _, transitions = _candidate_parameters(
                self.context,
                profile,
                candidate,
                self.config,
            )
            source, target = divmod(profile.index, len(self.config.state_names))
            self.assertAlmostEqual(float(transitions[source, target]), candidate)
            self.assertTrue(
                torch.allclose(
                    transitions.sum(dim=1),
                    torch.ones(3, dtype=torch.float64),
                )
            )

    def test_per_tree_transition_profiles_change_only_the_target_tree(self) -> None:
        context = replace(
            self.context,
            transition_probabilities=torch.stack(
                [
                    self.context.transition_probabilities,
                    self.context.transition_probabilities,
                ]
            ),
            tree_transition_times=[[0.25], [0.75]],
        )
        profiles = _transition_profiles(context, self.config)

        self.assertEqual(len(profiles), 4)
        self.assertEqual(profiles[2].name, "transition[tree_1,NC->PT]")
        candidate = float(profiles[2].original_value) * 0.5
        _, _, _, transitions = _candidate_parameters(
            context,
            profiles[2],
            candidate,
            self.config,
        )
        self.assertAlmostEqual(float(transitions[1, 0, 1]), candidate)
        self.assertAlmostEqual(
            float(transitions[0, 0, 1]),
            float(context.transition_probabilities[0, 0, 1]),
        )

    def test_per_tree_transition_profiles_follow_each_graph_mask(self) -> None:
        context = replace(
            self.context,
            transition_probabilities=torch.tensor(
                [
                    [
                        [0.7, 0.3, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    [
                        [0.7, 0.0, 0.3],
                        [0.0, 0.8, 0.2],
                        [0.0, 0.2, 0.8],
                    ],
                ],
                dtype=torch.float64,
            ),
            tree_transition_times=[[0.25], [0.75]],
        )
        profiles = _transition_profiles(context, self.config)

        self.assertEqual(
            [profile.name for profile in profiles],
            [
                "transition[tree_0,NC->PT]",
                "transition[tree_1,NC->LN]",
                "transition[tree_1,PT->LN]",
                "transition[tree_1,LN->PT]",
            ],
        )

    def test_per_tree_sampling_profiles_adjust_all_states_for_one_tree(self) -> None:
        context = replace(
            self.context,
            sampling_probabilities=torch.tensor(
                [[0.4, 0.4, 0.4], [0.2, 0.2, 0.2]],
                dtype=torch.float64,
            ),
            tree_sampling_scope="per_tree_shared_across_states",
        )
        profiles = _sampling_profiles(context, self.config)

        self.assertEqual(len(profiles), 2)
        self.assertEqual(profiles[1].name, "sampling[tree_1]")
        candidate = 0.55
        _, _, sampling, _ = _candidate_parameters(
            context,
            profiles[1],
            candidate,
            self.config,
        )
        self.assertTrue(
            torch.allclose(sampling[1], torch.full_like(sampling[1], candidate))
        )
        self.assertTrue(
            torch.allclose(sampling[0], torch.full_like(sampling[0], 0.4))
        )

    def test_state_specific_tree_sampling_profiles_adjust_one_cell_rate(self) -> None:
        context = replace(
            self.context,
            sampling_probabilities=torch.tensor(
                [[0.4, 0.3, 0.5], [0.2, 0.7, 0.6]],
                dtype=torch.float64,
            ),
            tree_sampling_scope="per_tree_by_state",
        )
        profiles = _sampling_profiles(context, self.config)

        self.assertEqual(len(profiles), 6)
        self.assertEqual(profiles[4].name, "sampling[tree_1,PT]")
        self.assertEqual(profiles[4].index, 4)
        candidate = 0.55
        _, _, sampling, _ = _candidate_parameters(
            context,
            profiles[4],
            candidate,
            self.config,
        )
        self.assertAlmostEqual(float(sampling[1, 1]), candidate)
        self.assertAlmostEqual(float(sampling[1, 0]), 0.2)
        self.assertAlmostEqual(float(sampling[1, 2]), 0.6)
        self.assertAlmostEqual(float(sampling[0, 1]), 0.3)

    def test_sampling_profiles_render_grouped_by_tree_and_by_state(self) -> None:
        dataset = "combined_S15_per_tree_state_sampling"
        tree_names = ("CRC04", "CRC10", "CRC11", "CRC12", "CRC13", "CRC14")
        rows = []
        ranges = []
        for tree_index, _ in enumerate(tree_names):
            for state_index, state_name in enumerate(self.config.state_names):
                parameter = f"sampling[tree_{tree_index},{state_name}]"
                rows.append(
                    {
                        "dataset": dataset,
                        "parameter": parameter,
                        "parameter_type": "sampling",
                        "parameter_index": tree_index * 3 + state_index,
                        "original_value": 0.2 + tree_index * 0.01 + state_index * 0.02,
                    }
                )
                for fraction, width in ((0.1, 0.04), (1e-5, 0.08)):
                    ranges.append(
                        {
                            "dataset": dataset,
                            "parameter": parameter,
                            "likelihood_fraction": fraction,
                            "lower_parameter_value": (
                                rows[-1]["original_value"] - width
                            ),
                            "higher_parameter_value": (
                                rows[-1]["original_value"] + width
                            ),
                        }
                    )

        plotting = dict(self.config.plotting)
        plotting.update(
            {
                "tree_names": list(tree_names),
                "formats": ["svg", "png"],
                "parameter_output_basenames": {
                    **self.config.plotting["parameter_output_basenames"],
                    "sampling": "sampling_profiles",
                },
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = replace(
                self.config,
                datasets=(dataset,),
                figure_root=Path(temporary_directory),
                plotting=plotting,
            )
            outputs = _render_grouped_sampling_profiles(
                config,
                {dataset: pd.DataFrame(ranges)},
                {dataset: pd.DataFrame(rows)},
            )
            output_names = {path.name for path in outputs}
            tree_group_svg = (
                Path(temporary_directory) / "sampling_profiles_by_tree.svg"
            ).read_text()
            state_group_svg = (
                Path(temporary_directory) / "sampling_profiles_by_state.svg"
            ).read_text()

        self.assertEqual(
            output_names,
            {
                "sampling_profiles_by_tree.svg",
                "sampling_profiles_by_tree.png",
                "sampling_profiles_by_state.svg",
                "sampling_profiles_by_state.png",
            },
        )
        self.assertIn("CRC04", tree_group_svg)
        self.assertIn("CRC14", tree_group_svg)
        self.assertIn("NC", state_group_svg)
        self.assertIn("PT", state_group_svg)
        self.assertIn("LN", state_group_svg)

    def test_dataset_offsets_center_any_number_of_datasets(self) -> None:
        self.assertEqual(_dataset_offsets(("only",), 0.23), {"only": 0.0})
        self.assertEqual(
            _dataset_offsets(("first", "second"), 0.23),
            {"first": -0.23, "second": 0.23},
        )
        self.assertEqual(
            _dataset_offsets(("first", "middle", "last"), 0.23),
            {"first": -0.23, "middle": 0.0, "last": 0.23},
        )

    def test_profile_box_colors_follow_cell_type(self) -> None:
        self.assertEqual(
            _profile_box_color(self.config, "birth", 0, "birth_shifted_min_branch"),
            "#6f91bd",
        )
        self.assertEqual(
            _profile_box_color(self.config, "scaled_sampling", 1, "transition_anchored"),
            "#e49a78",
        )
        self.assertEqual(
            _profile_box_color(self.config, "transition", 3, "transition_anchored"),
            "#e49a78",
        )
        self.assertEqual(
            _profile_box_color(self.config, "sampling", 4, "transition_anchored"),
            "#e49a78",
        )
        self.assertEqual(
            _profile_box_color(
                self.config,
                "sampling",
                0,
                "transition_anchored",
                "sampling[tree_0]",
            ),
            self.config.plotting["dataset_colors"]["transition_anchored"],
        )
        self.assertEqual(
            _profile_box_color(
                self.config,
                "sampling",
                4,
                "transition_anchored",
                "sampling[tree_1,PT]",
            ),
            "#e49a78",
        )

    def test_profile_layout_uses_narrow_panels_and_wider_boxes(self) -> None:
        minimum_dataset_spacing = (
            2.0
            * float(self.config.plotting["dataset_offset"])
            / (len(self.config.datasets) - 1)
        )
        self.assertLess(
            float(self.config.plotting["box_width"]),
            minimum_dataset_spacing,
        )
        self.assertGreaterEqual(
            minimum_dataset_spacing
            - float(self.config.plotting["box_width"])
            + 1e-12,
            float(self.config.plotting["box_width"]),
        )
        between_state_spacing = (
            float(self.config.plotting["group_spacing"])
            - 2.0 * float(self.config.plotting["dataset_offset"])
        )
        self.assertGreaterEqual(
            between_state_spacing
            - float(self.config.plotting["box_width"])
            + 1e-12,
            float(self.config.plotting["box_width"]),
        )
        self.assertEqual(tuple(self.config.plotting["figsize"]), (16.0, 4.2))

        combined_plotting = _plot_config(
            self.config,
            Path("combined_profiles"),
            "combined_S15_upgma_min_subtracted_fixed_sampling_0.1",
            "S15",
            "fixed",
        ).plotting
        self.assertEqual(tuple(combined_plotting["figsize"]), (18.5, 8.4))
        self.assertEqual(float(combined_plotting["box_width"]), 0.3)
        self.assertEqual(
            combined_plotting["composite_panel_types"],
            [
                "growth",
                "transition",
                "scaled_sampling",
                "sampling_ratio",
                "birth_ratio",
            ],
        )
        self.assertEqual(
            combined_plotting["composite_panel_rows"],
            [
                ["growth", "transition", "scaled_sampling"],
                ["sampling_ratio", "birth_ratio"],
            ],
        )
        self.assertEqual(
            combined_plotting["composite_panel_gaps"],
            [0.18, 0.4, 0.16],
        )
        self.assertGreater(
            combined_plotting["composite_panel_gaps"][1],
            combined_plotting["composite_panel_gaps"][2],
        )
        self.assertGreaterEqual(
            float(self.config.plotting["group_spacing"])
            - float(combined_plotting["box_width"]),
            float(combined_plotting["box_width"]),
        )
        self.assertIn("SGOOTR_combined_S15", combined_plotting["output_basename"])
        inferred_plotting = _plot_config(
            self.config,
            Path("inferred_profiles"),
            "combined_S15_upgma_min_subtracted_inferred_sampling",
            "S15",
            "inferred",
        ).plotting
        self.assertIn("inferred_sampling", inferred_plotting["output_basename"])
        per_tree_plotting = _plot_config(
            self.config,
            Path("per_tree_profiles"),
            "combined_S15_upgma_min_subtracted_per_tree_sampling",
            "S15",
            "per_tree",
        ).plotting
        self.assertIn("per_tree_sampling", per_tree_plotting["output_basename"])
        per_tree_state_plotting = _plot_config(
            self.config,
            Path("per_tree_state_profiles"),
            "combined_S15_upgma_min_subtracted_per_tree_state_sampling",
            "S15",
            "per_tree_state",
            ("CRC04", "CRC10", "CRC11", "CRC12", "CRC13", "CRC14"),
        ).plotting
        self.assertIn("png", per_tree_state_plotting["formats"])
        self.assertEqual(
            per_tree_state_plotting["tree_names"],
            ["CRC04", "CRC10", "CRC11", "CRC12", "CRC13", "CRC14"],
        )
        multi_tree_plotting = _plot_config(
            self.config,
            Path("multi_tree_profiles"),
            "combined_S10_CRC04_CRC11_CRC12",
            "S10",
            "per_tree_transition",
            ("CRC04", "CRC11", "CRC12"),
            include_rate_ratio_profiles=False,
        ).plotting
        self.assertEqual(
            multi_tree_plotting["composite_panel_types"],
            ["growth", "transition", "scaled_sampling"],
        )
        self.assertEqual(
            multi_tree_plotting["composite_panel_rows"],
            [["growth", "transition", "scaled_sampling"]],
        )
        self.assertEqual(
            multi_tree_plotting["composite_panel_gaps"],
            [0.18, 0.4],
        )
        self.assertEqual(
            _parameter_group_label(
                "sampling[tree_0,NC]",
                0,
                "sampling",
                replace(self.config, plotting=per_tree_state_plotting),
            ),
            "CRC04-NC",
        )
        self.assertEqual(
            _cohort_name(Path("inference/S15_upgma_min_subtracted/result.txt")),
            "S15",
        )

    def test_ratio_profile_intervals_are_rendered(self) -> None:
        dataset = "combined_S10_CRC11"
        parameter_rows = []
        range_rows = []
        for parameter_type, original_values in (
            ("sampling_ratio", {"PT": 0.3, "LN": 0.2}),
            ("birth_ratio", {"PT": 1.5, "LN": 1.7}),
        ):
            for parameter_index, (state_name, original_value) in enumerate(
                original_values.items(), start=1
            ):
                parameter = f"{parameter_type}[{state_name}/NC]"
                parameter_rows.append(
                    {
                        "dataset": dataset,
                        "parameter": parameter,
                        "parameter_type": parameter_type,
                        "parameter_index": parameter_index,
                        "original_value": original_value,
                    }
                )
                for fraction, width in ((0.1, 0.1), (1e-5, 0.2)):
                    range_rows.append(
                        {
                            "dataset": dataset,
                            "parameter": parameter,
                            "likelihood_fraction": fraction,
                            "lower_parameter_value": original_value - width,
                            "higher_parameter_value": original_value + width,
                        }
                    )

        with tempfile.TemporaryDirectory() as temp_dir:
            plot_config = _plot_config(
                self.config,
                Path(temp_dir) / "profiles",
                dataset,
                "S10",
                "per_tree_transition",
            )
            plot_config = replace(
                plot_config,
                plotting={**plot_config.plotting, "formats": ["png"]},
            )
            outputs = _plot_ratio_profile_intervals(
                plot_config,
                {dataset: pd.DataFrame(range_rows)},
                {dataset: pd.DataFrame(parameter_rows)},
            )

            self.assertEqual(len(outputs), 1)
            self.assertTrue(outputs[0].is_file())

    def test_per_tree_ratio_profiles_and_intervals_are_rendered(self) -> None:
        dataset = "combined_S10_CRC04_CRC11_CRC12"
        tree_names = ("CRC04", "CRC11", "CRC12")
        profile_rows = []
        parameter_rows = []
        range_rows = []
        for tree_index in range(len(tree_names)):
            for state_name, original_value in (("PT", 0.2), ("LN", 0.1)):
                parameter = (
                    f"sampling_ratio[tree_{tree_index},{state_name}/NC]"
                )
                parameter_rows.append(
                    {
                        "dataset": dataset,
                        "parameter": parameter,
                        "parameter_type": "sampling_ratio",
                        "parameter_index": tree_index * 3 + (
                            1 if state_name == "PT" else 2
                        ),
                        "original_value": original_value,
                    }
                )
                for fraction, width in ((0.1, 0.1), (1e-5, 0.2)):
                    range_rows.append(
                        {
                            "dataset": dataset,
                            "parameter": parameter,
                            "likelihood_fraction": fraction,
                            "lower_parameter_value": original_value - width,
                            "higher_parameter_value": original_value + width,
                        }
                    )
                for value, likelihood in (
                    (original_value / 2.0, -2.0),
                    (original_value, 0.0),
                    (original_value * 2.0, -2.0),
                ):
                    profile_rows.append(
                        {
                            "parameter": parameter,
                            "parameter_type": "sampling_ratio",
                            "parameter_value": value,
                            "summed_log_likelihood": likelihood,
                            "original_value": original_value,
                        }
                    )
        for parameter_index, (state_name, original_value) in enumerate(
            (("PT", 1.5), ("LN", 1.7)),
            start=1,
        ):
            parameter = f"birth_ratio[{state_name}/NC]"
            parameter_rows.append(
                {
                    "dataset": dataset,
                    "parameter": parameter,
                    "parameter_type": "birth_ratio",
                    "parameter_index": parameter_index,
                    "original_value": original_value,
                }
            )
            for fraction, width in ((0.1, 0.1), (1e-5, 0.2)):
                range_rows.append(
                    {
                        "dataset": dataset,
                        "parameter": parameter,
                        "likelihood_fraction": fraction,
                        "lower_parameter_value": original_value - width,
                        "higher_parameter_value": original_value + width,
                    }
                )
            for value, likelihood in (
                (original_value / 2.0, -2.0),
                (original_value, 0.0),
                (original_value * 2.0, -2.0),
            ):
                profile_rows.append(
                    {
                        "parameter": parameter,
                        "parameter_type": "birth_ratio",
                        "parameter_value": value,
                        "summed_log_likelihood": likelihood,
                        "original_value": original_value,
                    }
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            plot_config = _plot_config(
                self.config,
                Path(temp_dir),
                dataset,
                "S10",
                "per_tree_transition",
                tree_names,
            )
            plot_config = replace(
                plot_config,
                plotting={**plot_config.plotting, "formats": ["png"]},
            )
            sampling_outputs = _plot_ratio_profiles(
                plot_config,
                profile_rows,
                "sampling_ratio",
                r"\rho",
            )
            birth_outputs = _plot_ratio_profiles(
                plot_config,
                profile_rows,
                "birth_ratio",
                r"\beta",
            )
            interval_outputs = _plot_ratio_profile_intervals(
                plot_config,
                {dataset: pd.DataFrame(range_rows)},
                {dataset: pd.DataFrame(parameter_rows)},
            )
            sampling_interval_outputs = (
                _plot_sampling_ratio_profile_intervals_by_tree(
                    plot_config,
                    {dataset: pd.DataFrame(range_rows)},
                    {dataset: pd.DataFrame(parameter_rows)},
                )
            )

            self.assertEqual(len(sampling_outputs), 1)
            self.assertEqual(len(birth_outputs), 1)
            self.assertEqual(len(interval_outputs), 1)
            self.assertEqual(len(sampling_interval_outputs), 1)
            self.assertTrue(all(path.is_file() for path in (
                *sampling_outputs,
                *birth_outputs,
                *interval_outputs,
                *sampling_interval_outputs,
            )))

    def test_profile_ranges_include_inferred_rates_above_configured_ceiling(self) -> None:
        context = DatasetContext(
            birth_rates=torch.tensor([100.0, 80.0, 60.0], dtype=torch.float64),
            death_rates=torch.tensor([90.0, 70.0, 50.0], dtype=torch.float64),
            sampling_probabilities=torch.tensor(
                [0.1, 0.1, 0.1],
                dtype=torch.float64,
            ),
            transition_probabilities=torch.tensor(
                [[0.7, 0.3, 0.0], [0.0, 0.8, 0.2], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
            transition_times=[0.25, 0.75],
            sensitivity_inputs=[],
        )

        growth_profiles = _growth_profiles(context, self.config)

        self.assertGreater(growth_profiles[0].values.max(), 10.0)
        self.assertAlmostEqual(growth_profiles[0].values.max(), 20.0)


if __name__ == "__main__":
    unittest.main()
