from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt

from src.plot_star_transition_rates_vs_probabilities import RateObservation
from src.plot_star_diversitree_errors import (
    AbsoluteError,
    CombinedErrorPlotRow,
    Mach2BatchTransitionTimeError,
    TransitionRankCorrelation,
    TransitionPearsonCorrelation,
    TransitionTimeError,
    _derived_rates,
    _sampling_adjusted_star_transition_probabilities,
    _transition_pearson_correlation,
    _transition_rank_correlation,
    load_linear_transition_probability_comparison_errors,
    load_matched_absolute_errors,
    load_matched_linear_absolute_errors,
    load_mach2_transition_time_errors,
    load_star_transition_time_errors,
    load_star_transition_probability_errors,
    load_transition_rank_correlations,
    plot_combined_error_boxplots,
    plot_error_boxplots,
    plot_topology_comparison,
    restrict_row_to_common_batches,
)


class PlotStarDiversitreeErrorsTests(unittest.TestCase):
    def test_derives_net_growth_and_scaled_birth(self) -> None:
        rates = _derived_rates(
            (8.0, 7.0, 6.0, 5.0),
            (2.0, 3.0, 1.0, 4.0),
            (0.5, 0.4, 0.3, 0.2),
        )

        self.assertEqual(rates["net_growth"], (6.0, 4.0, 5.0, 1.0))
        for actual, expected in zip(
            rates["scaled_birth"],
            (4.0, 2.8, 1.8, 1.0),
        ):
            self.assertAlmostEqual(actual, expected)

    def test_transition_rank_correlations_are_computed_once_per_batch(self) -> None:
        comparison = SimpleNamespace(
            setting="setting_1",
            batch="batch_0",
            truth={"p_A_to_B": 0.1, "p_A_to_C": 0.4, "p_A_to_D": 0.9},
            inferred={
                "p_A_to_B": 0.2,
                "p_A_to_C": 0.3,
                "p_A_to_D": 0.8,
            },
        )
        probabilities = {"B": 0.1, "C": 0.4, "D": 0.9}
        rate_observations = [
            RateObservation("diversitree", "setting_1", "B", 0.1, 0.3, "batch_0"),
            RateObservation("diversitree", "setting_1", "C", 0.4, 0.2, "batch_0"),
            RateObservation("diversitree", "setting_1", "D", 0.9, 0.1, "batch_0"),
        ]
        for transition, probability, rates in (
            ("B", 0.1, (1.0, 2.0)),
            ("C", 0.4, (3.0, 2.0)),
            ("D", 0.9, (2.0, 4.0)),
        ):
            rate_observations.extend(
                RateObservation(
                    "BEAM", "setting_1", transition, probability, rate, "batch_0"
                )
                for rate in rates
            )

        with (
            patch(
                "src.plot_star_diversitree_errors.load_completed_progressive_comparisons",
                return_value=[comparison],
            ),
            patch(
                "src.plot_star_diversitree_errors.collect_rate_observations",
                return_value=rate_observations,
            ),
        ):
            correlations = load_transition_rank_correlations(Path("unused"), "star", 3)

        by_method = {item.method: item.value for item in correlations}
        self.assertEqual(set(by_method), {"CONCERT", "diversitree", "BEAM"})
        self.assertAlmostEqual(by_method["CONCERT"], 1.0)
        self.assertAlmostEqual(by_method["diversitree"], -1.0)
        self.assertAlmostEqual(by_method["BEAM"], 1.0)
        order_errors = {item.method: item.order_error_rate for item in correlations}
        self.assertAlmostEqual(order_errors["CONCERT"], 0.0)
        self.assertAlmostEqual(order_errors["diversitree"], 1.0)
        self.assertAlmostEqual(order_errors["BEAM"], 0.0)
        self.assertEqual(len(correlations), 3)

    def test_order_error_counts_inferred_ties_as_half_and_omits_truth_ties(self) -> None:
        result = _transition_rank_correlation(
            "setting_1",
            "batch_0",
            "BEAM",
            {"B": 0.1, "C": 0.1, "D": 0.9},
            {"B": 0.4, "C": 0.5, "D": 0.4},
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.order_error_rate, 0.75)

    def test_transition_pearson_correlation_uses_values_across_transitions(self) -> None:
        result = _transition_pearson_correlation(
            "setting_1",
            "batch_0",
            "CONCERT",
            {"B": 0.1, "C": 0.4, "D": 0.9},
            {"B": 0.2, "C": 0.3, "D": 0.8},
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.value, 0.9750002110024925)

        undefined = _transition_pearson_correlation(
            "setting_1",
            "batch_0",
            "BEAM",
            {"B": 0.1, "C": 0.4, "D": 0.9},
            {"B": 2.0, "C": 2.0, "D": 2.0},
        )
        self.assertIsNone(undefined)

    def test_adjusts_star_concert_probabilities_by_inferred_sampling(self) -> None:
        adjusted = _sampling_adjusted_star_transition_probabilities(
            {
                "sampling_A": 0.5,
                "sampling_B": 0.25,
                "sampling_C": 0.4,
                "sampling_D": 0.75,
                "p_A_to_B": 0.4,
                "p_A_to_C": 0.5,
                "p_A_to_D": 0.6,
            }
        )

        self.assertEqual(
            adjusted,
            {"B": 0.2, "C": 0.4, "D": 0.8999999999999999},
        )

    def test_rejects_nonpositive_root_sampling_for_adjusted_probabilities(self) -> None:
        with self.assertRaisesRegex(ValueError, "root sampling probability"):
            _sampling_adjusted_star_transition_probabilities(
                {
                    "sampling_A": 0.0,
                    "sampling_B": 0.25,
                    "sampling_C": 0.4,
                    "sampling_D": 0.75,
                    "p_A_to_B": 0.4,
                    "p_A_to_C": 0.5,
                    "p_A_to_D": 0.6,
                }
            )

    def test_restricts_plot_data_to_batches_with_completed_all_methods(self) -> None:
        errors = [
            AbsoluteError(setting, batch, "A", method, "net_growth", 0.1)
            for setting, batch, methods in (
                ("setting_1", "batch_1", ("CONCERT", "MuSSE", "BEAM")),
                ("setting_1", "batch_2", ("CONCERT", "MuSSE", "BEAM")),
            )
            for method in methods
        ]
        row = CombinedErrorPlotRow(
            label="star",
            errors=errors,
            transition_time_errors=[
                TransitionTimeError("setting_1", "batch_1", "edge", 0.1),
                TransitionTimeError("setting_1", "batch_2", "edge", 0.2),
            ],
            mach2_transition_time_errors=[],
            transition_probability_errors=[],
            diversitree_transition_probability_errors=[],
            beam_transition_probability_errors=[],
            transition_rank_correlations=[
                TransitionRankCorrelation(
                    "setting_1", batch, "CONCERT", 1.0, 0.0
                )
                for batch in ("batch_1", "batch_2")
            ],
            transition_correlations=[
                TransitionPearsonCorrelation(
                    "setting_1", batch, "CONCERT", 0.9
                )
                for batch in ("batch_1", "batch_2")
            ],
        )

        filtered = restrict_row_to_common_batches(
            row,
            {("setting_1", "batch_1")},
        )

        self.assertEqual(filtered.common_batch_keys, [("setting_1", "batch_1")])
        self.assertEqual({error.batch for error in filtered.errors}, {"batch_1"})
        self.assertEqual(
            {error.batch for error in filtered.transition_time_errors},
            {"batch_1"},
        )
        self.assertEqual(
            {error.batch for error in filtered.transition_rank_correlations},
            {"batch_1"},
        )
        self.assertEqual(
            {correlation.batch for correlation in filtered.transition_correlations},
            {"batch_1"},
        )

    def test_plot_omits_unavailable_methods_and_notes_them(self) -> None:
        errors = [
            AbsoluteError(
                setting="setting_1",
                batch="batch_0",
                state="A",
                method=method,
                metric=metric,
                value=0.5,
            )
            for method in ("CONCERT", "MuSSE")
            for metric in ("net_growth", "scaled_birth")
        ]
        captured_figures = []
        original_subplots = plt.subplots

        def capture_figure(*args, **kwargs):
            figure, axes = original_subplots(*args, **kwargs)
            captured_figures.append(figure)
            return figure, axes

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "linear_errors.png"
            with patch(
                "src.plot_star_diversitree_errors.plt.subplots",
                side_effect=capture_figure,
            ):
                plot_error_boxplots(
                    errors,
                    output_path,
                )
            self.assertTrue(output_path.is_file())
            self.assertEqual(
                [
                    tick.get_text()
                    for tick in captured_figures[0].axes[0].get_xticklabels()
                ],
                [],
            )
            self.assertLessEqual(captured_figures[0].get_size_inches()[0], 8)
            self.assertEqual(
                captured_figures[0].axes[0].title.get_fontsize(),
                9,
            )
            self.assertEqual(
                [
                    text.get_text()
                    for text in captured_figures[0].legends[0].get_texts()
                ],
                ["CONCERT", "diversitree"],
            )
            self.assertGreaterEqual(
                captured_figures[0].legends[0].get_texts()[0].get_fontsize(),
                8.5,
            )
            self.assertFalse(captured_figures[0].legends[0].get_frame_on())
            self.assertFalse(captured_figures[0].texts)

    def test_plot_adds_one_aggregate_transition_time_panel(self) -> None:
        errors = [
            AbsoluteError(
                setting="setting_1",
                batch="batch_0",
                state="A",
                method=method,
                metric=metric,
                value=0.5,
            )
            for method in ("CONCERT", "MuSSE")
            for metric in ("net_growth", "scaled_birth")
        ]
        transition_errors = [
            TransitionTimeError("setting_1", "batch_0", edge, value)
            for edge, value in (("A→B", 0.1), ("B→C", 0.2), ("C→D", 0.3))
        ]
        captured_figures = []
        original_subplots = plt.subplots

        def capture_figure(*args, **kwargs):
            figure, axes = original_subplots(*args, **kwargs)
            captured_figures.append(figure)
            return figure, axes

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "linear_errors.png"
            with patch(
                "src.plot_star_diversitree_errors.plt.subplots",
                side_effect=capture_figure,
            ):
                plot_error_boxplots(
                    errors,
                    output_path,
                    transition_time_errors=transition_errors,
                    transition_probability_errors=[0.1, 0.2, 0.3],
                    diversitree_transition_probability_errors=[0.15, 0.25, 0.35],
                )
            self.assertTrue(output_path.is_file())
            self.assertEqual(len(captured_figures[0].axes), 4)
            self.assertLess(captured_figures[0].get_size_inches()[0], 6)
            self.assertEqual(len(captured_figures[0].axes[2].patches), 1)
            self.assertEqual(len(captured_figures[0].axes[3].patches), 2)
            self.assertEqual(
                captured_figures[0].axes[0].get_xlim(),
                captured_figures[0].axes[2].get_xlim(),
            )
            self.assertEqual(
                captured_figures[0].axes[0].get_xlim(),
                captured_figures[0].axes[3].get_xlim(),
            )
            self.assertEqual(
                [
                    tick.get_text()
                    for tick in captured_figures[0].axes[2].get_xticklabels()
                ],
                [],
            )
            self.assertEqual(
                [
                    tick.get_text()
                    for tick in captured_figures[0].axes[3].get_xticklabels()
                ],
                [],
            )
            self.assertEqual(
                captured_figures[0].axes[3].get_title(),
                "Transition probability",
            )
            self.assertFalse(captured_figures[0].texts)
            self.assertEqual(
                captured_figures[0].axes[0].get_ylim(),
                captured_figures[0].axes[1].get_ylim(),
            )
            self.assertEqual(
                captured_figures[0].axes[2].get_ylim(),
                (0.0, 0.2),
            )
            boxes = [
                patch.get_path().vertices[:, 0]
                for patch in captured_figures[0].axes[0].patches
            ]
            box_bounds = sorted((vertices.min(), vertices.max()) for vertices in boxes)
            self.assertAlmostEqual(box_bounds[1][0] - box_bounds[0][1], 0.28)
            self.assertAlmostEqual(
                captured_figures[0].axes[0].patches[0].get_linewidth(),
                0.8,
            )
            self.assertAlmostEqual(
                max(line.get_linewidth() for line in captured_figures[0].axes[0].lines),
                1.7,
            )

    def test_plot_combines_star_and_linear_in_one_row_with_one_legend(self) -> None:
        def make_row(label: str, methods: tuple[str, ...]) -> CombinedErrorPlotRow:
            errors = [
                AbsoluteError(
                    setting="setting_1",
                    batch="batch_0",
                    state="Mean across states" if method == "BEAM" else "A",
                    method=method,
                    metric=metric,
                    value=0.25 * (method_index + 1),
                )
                for method_index, method in enumerate(methods)
                for metric in ("net_growth", "scaled_birth")
            ]
            return CombinedErrorPlotRow(
                label=label,
                errors=errors,
                transition_time_errors=[
                    TransitionTimeError("setting_1", "batch_0", "A→B", 0.1),
                    TransitionTimeError("setting_1", "batch_0", "B→C", 0.2),
                ],
                mach2_transition_time_errors=[
                    Mach2BatchTransitionTimeError(
                        "setting_1",
                        "batch_0",
                        0.15,
                        tree_count=2,
                        transition_count=3,
                    )
                ],
                transition_probability_errors=[0.1, 0.2],
                diversitree_transition_probability_errors=[0.15, 0.25],
                beam_transition_probability_errors=[0.2, 0.3],
            )

        rows = (
            make_row("star_3", ("CONCERT", "MuSSE", "BEAM")),
            make_row("linear A→D", ("CONCERT", "MuSSE")),
        )
        captured_figures = []
        original_subplots = plt.subplots

        def capture_figure(*args, **kwargs):
            figure, axes = original_subplots(*args, **kwargs)
            captured_figures.append(figure)
            return figure, axes

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "combined_errors.png"
            with patch(
                "src.plot_star_diversitree_errors.plt.subplots",
                side_effect=capture_figure,
            ):
                plot_combined_error_boxplots(rows, output_path)

            figure = captured_figures[0]
            self.assertTrue(output_path.is_file())
            self.assertTrue(output_path.with_suffix(".pdf").is_file())
            self.assertEqual(len(figure.axes), 8)
            self.assertLess(figure.get_size_inches()[0], 9)
            self.assertLess(
                figure.axes[0].get_position().width * figure.get_size_inches()[0],
                0.85,
            )
            self.assertEqual(len(figure.axes[0].patches), 3)
            self.assertEqual(len(figure.axes[4].patches), 2)
            self.assertEqual(len(figure.axes[3].patches), 3)
            self.assertEqual(len(figure.axes[7].patches), 3)
            self.assertEqual(len(figure.axes[2].patches), 2)
            self.assertEqual(len(figure.axes[6].patches), 2)
            self.assertAlmostEqual(
                max(
                    vertices.max() - vertices.min()
                    for vertices in (
                        patch.get_path().vertices[:, 0]
                        for patch in figure.axes[0].patches
                    )
                ),
                0.22,
            )
            self.assertAlmostEqual(figure.axes[0].patches[0].get_linewidth(), 0.8)
            self.assertAlmostEqual(
                max(line.get_linewidth() for line in figure.axes[0].lines),
                1.7,
            )
            self.assertEqual(
                [text.get_text() for text in figure.legends[0].get_texts()],
                [
                    "CONCERT",
                    "diversitree",
                    "BEAM",
                    "MACH2 (edge midpoint)",
                ],
            )
            self.assertEqual(
                {text.get_text() for text in figure.texts},
                {"star_3", "linear A→D"},
            )
            self.assertTrue(
                all(not axis.get_xticklabels() for axis in figure.axes)
            )
            self.assertEqual(figure.axes[0].get_ylabel(), "Absolute error")
            self.assertEqual(figure.axes[1].get_ylabel(), "")
            self.assertTrue(
                all(label.get_visible() for label in figure.axes[0].get_yticklabels())
            )
            self.assertTrue(
                all(
                    label.get_visible()
                    for index in (2, 3, 4, 6, 7)
                    for label in figure.axes[index].get_yticklabels()
                )
            )
            self.assertTrue(
                all(
                    not label.get_visible()
                    for index in (1, 5)
                    for label in figure.axes[index].get_yticklabels()
                )
            )
            self.assertEqual(
                figure.axes[0].get_ylim(),
                figure.axes[5].get_ylim(),
            )
            self.assertEqual(figure.axes[2].get_ylim(), (0.0, 0.2))
            self.assertEqual(figure.axes[6].get_ylim(), (0.0, 0.2))
            self.assertEqual(
                figure.axes[3].get_ylim(),
                figure.axes[7].get_ylim(),
            )
            self.assertEqual(figure.axes[3].get_xlim(), figure.axes[7].get_xlim())
            self.assertEqual(
                [
                    round(
                        (patch.get_path().vertices[:, 0].min()
                        + patch.get_path().vertices[:, 0].max())
                        / 2,
                        2,
                    )
                    for patch in figure.axes[3].patches
                ],
                [1.0, 1.5, 2.0],
            )

    def test_mach2_transition_errors_use_edge_midpoints_and_batch_means(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)

            def add_tree(
                tree: str,
                nodes: str,
                edges: str,
                truth: str,
                refined: str,
                node_origins: str,
            ) -> None:
                tree_dir = data_dir / "setting_1" / "batch_0" / tree
                mach2_output = tree_dir / "mach2_full" / "mach2_output"
                mach2_output.mkdir(parents=True)
                (tree_dir / "nodes.csv").write_text(nodes)
                (tree_dir / "edges.csv").write_text(edges)
                (tree_dir / "transition_times_with_edges.csv").write_text(truth)
                (mach2_output / "0-T-0.refined.tree").write_text(refined)
                (mach2_output / "0-T-0.nodes").write_text(node_origins)

            add_tree(
                "tree_0",
                "node_id,T,is_leaf,state_id\n"
                "10,1,0,-1\n20,0.5,0,-1\n30,0,1,0\n40,0,1,1\n",
                "parent,child,branch_length\n10,20,0.5\n20,30,0.5\n20,40,0.5\n",
                "2\n0.8,10,20,0,1\n0.2,20,30,1,0\n",
                "1\t2\t0\n2\t3\t-1\n2\t4\t1\n",
                "1\t1\n2\t2\n3\t3\n4\t4\n",
            )
            add_tree(
                "tree_1",
                "node_id,T,is_leaf,state_id\n10,1,0,-1\n20,0,1,0\n",
                "parent,child,branch_length\n10,20,1\n",
                "1\n0.9,10,20,0,1\n",
                "3\t4\t0\n",
                "3\t1\n4\t1\n",
            )

            errors = load_mach2_transition_time_errors(data_dir)

        self.assertEqual(len(errors), 1)
        self.assertAlmostEqual(errors[0].value, 0.075)
        self.assertEqual(errors[0].tree_count, 2)
        self.assertEqual(errors[0].transition_count, 3)

    def test_topology_comparison_draws_simulation_results_in_reference_layout(self) -> None:
        def make_row(label: str, offset: float) -> CombinedErrorPlotRow:
            rate_errors = [
                AbsoluteError(
                    setting=f"setting_{index}",
                    batch="batch_0",
                    state=state,
                    method=method,
                    metric=metric,
                    value=offset + index / 100,
                )
                for index, method in enumerate(("CONCERT", "MuSSE", "BEAM"), start=1)
                for metric in ("net_growth", "scaled_birth")
                for state in (("Mean across states",) if method == "BEAM" else ("A", "B"))
            ]
            return CombinedErrorPlotRow(
                label=label,
                errors=rate_errors,
                transition_time_errors=[
                    TransitionTimeError("setting_1", "batch_0", "A→B", 0.02 + offset),
                    TransitionTimeError("setting_1", "batch_0", "B→C", 0.04 + offset),
                ],
                mach2_transition_time_errors=[
                    Mach2BatchTransitionTimeError(
                        "setting_1", "batch_0", 0.06 + offset, 1, 2
                    ),
                    Mach2BatchTransitionTimeError(
                        "setting_2", "batch_0", 0.08 + offset, 1, 2
                    ),
                ],
                transition_probability_errors=[0.1 + offset, 0.2 + offset],
                diversitree_transition_probability_errors=[
                    0.15 + offset,
                    0.25 + offset,
                ],
                beam_transition_probability_errors=[0.3 + offset, 0.4 + offset],
                transition_correlations=[
                    TransitionPearsonCorrelation(
                        f"setting_{index}",
                        "batch_0",
                        method,
                        value + offset,
                    )
                    for method, value in (
                        ("CONCERT", 0.9),
                        ("diversitree", 0.4),
                        ("BEAM", -0.5),
                    )
                    for index in range(2)
                ],
            )

        rows = (make_row("star_3", 0.0), make_row("linear A→D", 0.01))
        captured_figures = []
        original_figure = plt.figure

        def capture_figure(*args, **kwargs):
            figure = original_figure(*args, **kwargs)
            captured_figures.append(figure)
            return figure

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "topology_errors.png"
            with patch(
                "src.plot_star_diversitree_errors.plt.figure",
                side_effect=capture_figure,
            ):
                plot_topology_comparison(rows, output_path)
            self.assertTrue(output_path.is_file())
            self.assertTrue(output_path.with_suffix(".pdf").is_file())

        figure = captured_figures[0]
        self.assertEqual(len(figure.axes), 10)
        self.assertEqual(figure.get_size_inches().tolist(), [19.0, 5.2])
        self.assertEqual(figure.axes[0].get_title(), "Linear topology")
        self.assertEqual(figure.axes[5].get_title(), "Star topology")
        self.assertEqual(figure.axes[0].texts[0].get_text(), "(a)")
        self.assertEqual(figure.axes[5].texts[0].get_text(), "(b)")
        self.assertEqual(len(figure.axes[0].patches), 10)
        self.assertEqual(len(figure.axes[5].patches), 8)
        self.assertGreater(
            figure.axes[0].get_position().y0,
            figure.axes[1].get_position().y0,
        )
        self.assertGreater(
            figure.axes[5].get_position().y0,
            figure.axes[6].get_position().y0,
        )
        self.assertLess(
            figure.axes[0].get_position().x0,
            figure.axes[5].get_position().x0,
        )
        self.assertGreaterEqual(
            figure.axes[2].get_position().x0 - figure.axes[1].get_position().x1,
            0.055,
        )
        linear_nodes = {
            text.get_text(): text.get_position()
            for text in figure.axes[0].texts[1:]
        }
        self.assertLess(linear_nodes["A"][0], linear_nodes["B"][0])
        self.assertLess(linear_nodes["B"][0], linear_nodes["C"][0])
        self.assertLess(linear_nodes["C"][0], linear_nodes["D"][0])
        for node in linear_nodes.values():
            self.assertAlmostEqual(node[1], 0.5)
        star_nodes = {
            text.get_text(): text.get_position()
            for text in figure.axes[5].texts[1:]
        }
        self.assertLess(star_nodes["A"][0], star_nodes["B"][0])
        self.assertGreater(star_nodes["B"][1], star_nodes["C"][1])
        self.assertGreater(star_nodes["C"][1], star_nodes["D"][1])
        self.assertEqual(
            [figure.axes[index].get_title() for index in (1, 2, 3, 4)],
            [
                "Net\ngrowth",
                "Scaled\nbirth",
                "Transition\ntime",
                "Transition\ncorrelation",
            ],
        )
        self.assertEqual(
            [figure.axes[index].get_ylabel() for index in range(1, 5)],
            ["Absolute error", "Absolute error", "Absolute error", "Pearson's r"],
        )
        self.assertEqual(
            [figure.axes[index].get_ylabel() for index in range(6, 10)],
            ["Absolute error", "Absolute error", "Absolute error", "Pearson's r"],
        )
        self.assertEqual(figure.axes[3].get_ylim(), (0.0, 0.2))
        self.assertEqual(figure.axes[8].get_ylim(), (0.0, 0.2))
        self.assertEqual(figure.axes[4].get_ylim(), (-1.05, 1.05))
        self.assertEqual(figure.axes[9].get_ylim(), (-1.05, 1.05))
        self.assertEqual(len(figure.axes[1].patches), 3)
        self.assertEqual(len(figure.axes[3].patches), 2)
        self.assertEqual(len(figure.axes[4].patches), 3)
        self.assertEqual(len(figure.axes[6].patches), 3)
        self.assertIn(
            figure.axes[1].title.get_fontfamily()[0],
            ("Arial", "Nimbus Sans", "DejaVu Sans"),
        )
        self.assertGreaterEqual(figure.axes[1].title.get_fontsize(), 16)
        self.assertGreaterEqual(figure.axes[1].get_yticklabels()[0].get_fontsize(), 12)
        self.assertEqual(
            figure.axes[1].spines["left"].get_edgecolor(),
            (0.18823529411764706, 0.18823529411764706, 0.18823529411764706, 1.0),
        )
        FigureCanvasAgg(figure)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        transition_title_bbox = figure.axes[3].title.get_window_extent(renderer)
        for node in figure.axes[0].patches[3:]:
            self.assertFalse(
                node.get_window_extent(renderer).overlaps(transition_title_bbox)
            )
        for start in (1, 6):
            for left_axis, right_axis in zip(
                figure.axes[start : start + 3],
                figure.axes[start + 1 : start + 4],
            ):
                label_left_edges = [
                    label.get_window_extent(renderer).x0
                    for label in right_axis.get_yticklabels()
                    if label.get_visible()
                ]
                self.assertGreaterEqual(
                    min(label_left_edges),
                    left_axis.get_window_extent(renderer).x1,
                )
        self.assertEqual(
            [text.get_text() for text in figure.legends[0].get_texts()],
            ["CONCERT", "diversitree", "BEAM", "MACH2"],
        )

    def test_loads_matched_linear_probability_errors_with_cross_validation(self) -> None:
        fieldnames = [
            "setting",
            "sampling_rate",
            *(f"birth_{state}" for state in "ABCD"),
            *(f"death_{state}" for state in "ABCD"),
            "p_A_to_B",
            "p_B_to_C",
            "p_C_to_D",
            "transition_time_A",
            "transition_time_B",
            "transition_time_C",
        ]
        probabilities = {
            "setting_1": (0.2, 0.3, 0.1),
            "setting_2": (0.4, 0.5, 0.3),
            "setting_3": (0.6, 0.7, 0.5),
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            with (data_dir / "generated_settings.csv").open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for setting, (p_ab, p_bc, p_cd) in probabilities.items():
                    writer.writerow(
                        {
                            "setting": setting,
                            "sampling_rate": 0.5,
                            **{f"birth_{state}": 4 for state in "ABCD"},
                            **{f"death_{state}": 1 for state in "ABCD"},
                            "p_A_to_B": p_ab,
                            "p_B_to_C": p_bc,
                            "p_C_to_D": p_cd,
                            "transition_time_A": 0.1,
                            "transition_time_B": 0.2,
                            "transition_time_C": 0.3,
                        }
                    )

            for setting, edge_probabilities in probabilities.items():
                batch_dir = data_dir / setting / "batch_0"
                fit_dir = batch_dir / "diversitree_musse_first8"
                fit_dir.mkdir(parents=True)
                (fit_dir / "joint_musse_fit.txt").write_text(
                    "transition_model=linear\n"
                    "convergence_code=0\n"
                    "sampling_fraction=0.5,0.5,0.5,0.5\n"
                )
                with (fit_dir / "parameter_estimates.csv").open(
                    "w",
                    newline="",
                ) as output:
                    writer = csv.writer(output)
                    writer.writerow(("parameter", "estimate"))
                    for index in range(1, 5):
                        writer.writerow((f"lambda{index}", 4.0))
                        writer.writerow((f"mu{index}", 1.0))
                    for parameter, probability in zip(
                        ("q12", "q23", "q34"),
                        edge_probabilities,
                    ):
                        writer.writerow((parameter, probability * 2.0))

                beam_dir = batch_dir / "tree_0" / "beam_snakemake"
                beam_dir.mkdir(parents=True)
                (beam_dir / "run_status.tsv").write_text(
                    "tree_id\tstatus\ttip_count\tobserved_tissues\tmessage\n"
                    f"{setting}/batch_0/tree_0\tcompleted\t20\tA,B,C,D\t\n"
                )
                (beam_dir / "parameter_summary.tsv").write_text(
                    "tree_id\tparameter\tmean\tmedian\tHPD95_lower\t"
                    "HPD95_upper\tsamples_after_burnin\n"
                    f"{setting}/batch_0/tree_0\tbirthDiffRate\t1\t1\t0.5\t1.5\t2\n"
                )
                (beam_dir / "result.log").write_text(
                    "Sample\ttissueClockRate\t"
                    "tissueSubstModelLogger.relMigRate_A_B\t"
                    "tissueSubstModelLogger.relMigRate_B_C\t"
                    "tissueSubstModelLogger.relMigRate_C_D\n"
                    + "".join(
                        f"{index}\t1\t"
                        + "\t".join(str(probability * 2) for probability in edge_probabilities)
                        + "\n"
                        for index in range(2)
                    )
                )

            concert_dir = data_dir / "setting_2" / "batch_0"
            (concert_dir / "inference_joint_coordinate_first_8_trees.out").write_text(
                "LL=-1\n"
                "birth: 4 5 6 7\n"
                "death: 1 1 2 2\n"
                "sampling: 0.5 0.4 0.3 0.2\n"
                "transition_probabilities_by_epoch:\n"
                "epoch_1 0->1\n"
                "0->0 0.5\n"
                "0->1 0.5\n"
                "epoch_2 1->2\n"
                "1->1 0.4\n"
                "1->2 0.6\n"
                "epoch_3 2->3\n"
                "2->2 0.3\n"
                "2->3 0.4\n"
                "transition_times_by_edge:\n"
                "0->1 0.3\n"
                "1->2 0.2\n"
                "2->3 0.1\n"
            )

            concert_errors, diversitree_errors, beam_errors, slopes, clipped = (
                load_linear_transition_probability_comparison_errors(data_dir)
            )

        for error in concert_errors:
            self.assertAlmostEqual(error, 0.1)
        self.assertEqual(diversitree_errors, [0.0, 0.0, 0.0])
        self.assertEqual(len(beam_errors), 3)
        self.assertTrue(all(error < 1e-12 for error in beam_errors))
        self.assertEqual(slopes, {"B": 2.0, "C": 2.0, "D": 2.0})
        self.assertEqual(clipped, 0)

    def test_linear_plot_loader_includes_completed_beam_trees(self) -> None:
        fieldnames = [
            "setting",
            "sampling_rate",
            *(f"birth_{state}" for state in "ABCD"),
            *(f"death_{state}" for state in "ABCD"),
            "p_A_to_B",
            "p_B_to_C",
            "p_C_to_D",
            "transition_time_A",
            "transition_time_B",
            "transition_time_C",
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            with (data_dir / "generated_settings.csv").open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "setting": "setting_1",
                        "sampling_rate": 0.5,
                        **{f"birth_{state}": 4 for state in "ABCD"},
                        **{f"death_{state}": 1 for state in "ABCD"},
                        "p_A_to_B": 0.8,
                        "p_B_to_C": 0.7,
                        "p_C_to_D": 0.6,
                        "transition_time_A": 0.1,
                        "transition_time_B": 0.2,
                        "transition_time_C": 0.3,
                    }
                )

            batch_dir = data_dir / "setting_1" / "batch_0"
            batch_dir.mkdir(parents=True)
            concert_dir = batch_dir
            (
                concert_dir / "inference_joint_coordinate_first_8_trees.out"
            ).write_text(
                "LL=-1\n"
                "birth: 4 5 6 7\n"
                "death: 1 1 2 2\n"
                "sampling: 0.5 0.4 0.3 0.2\n"
                "transition_probabilities_by_epoch:\n"
                "epoch_1 0->1\n"
                "0->0 0.2\n"
                "0->1 0.8\n"
                "epoch_2 1->2\n"
                "1->1 0.3\n"
                "1->2 0.7\n"
                "epoch_3 2->3\n"
                "2->2 0.4\n"
                "2->3 0.6\n"
                "transition_times_by_edge:\n"
                "0->1 0.9\n"
                "1->2 0.8\n"
                "2->3 0.7\n"
            )
            fit_dir = batch_dir / "diversitree_musse_first8"
            fit_dir.mkdir()
            (fit_dir / "joint_musse_fit.txt").write_text(
                "transition_model=linear\n"
                "convergence_code=0\n"
                "sampling_fraction=0.5,0.5,0.5,0.5\n"
            )
            with (fit_dir / "parameter_estimates.csv").open(
                "w",
                newline="",
            ) as output:
                writer = csv.writer(output)
                writer.writerow(("parameter", "estimate"))
                for index in range(1, 5):
                    writer.writerow((f"lambda{index}", 4.0))
                    writer.writerow((f"mu{index}", 1.0))
                for parameter in ("q12", "q23", "q34"):
                    writer.writerow((parameter, 0.5))

            beam_dir = batch_dir / "tree_0" / "beam_snakemake"
            beam_dir.mkdir(parents=True)
            (beam_dir / "run_status.tsv").write_text(
                "tree_id\tstatus\ttip_count\tobserved_tissues\tmessage\n"
                "setting_1/batch_0/tree_0\tcompleted\t100\tA,B,C,D\t\n"
            )
            (beam_dir / "parameter_summary.tsv").write_text(
                "tree_id\tparameter\tmean\n"
                "setting_1/batch_0/tree_0\tbirthDiffRate\t5\n"
                "setting_1/batch_0/tree_0\trelativeDeathRate\t0.2\n"
            )

            errors = load_matched_linear_absolute_errors(data_dir)

        beam_errors = [error for error in errors if error.method == "BEAM"]
        self.assertEqual(len(beam_errors), 2)
        self.assertEqual(
            {(error.metric, error.state, error.tree) for error in beam_errors},
            {
                ("net_growth", "Mean across states", "tree_0"),
                ("scaled_birth", "Mean across states", "tree_0"),
            },
        )
        self.assertAlmostEqual(
            next(error.value for error in beam_errors if error.metric == "net_growth"),
            2.0,
        )
        self.assertAlmostEqual(
            next(error.value for error in beam_errors if error.metric == "scaled_birth"),
            1.125,
        )

    def test_loads_matched_errors_and_omits_nonconverged_fits(self) -> None:
        header = [
            "setting",
            *(f"birth_{state}" for state in "ABCD"),
            *(f"death_{state}" for state in "ABCD"),
            *(f"p_A_to_{state}" for state in "ABCD"),
            "transition_time_1",
            "transition_time_2",
            "sampling_rate",
            "num_trees",
        ]
        truth_row = [
            "setting_1",
            *("8" for _ in "ABCD"),
            *("2" for _ in "ABCD"),
            *("0.25" for _ in "ABCD"),
            "0.1",
            "0.2",
            "0.5",
            "100",
        ]
        coordinated_output = """\
LL=1.0
birth: 9 8 7 6
death: 2 2 2 2
sampling: 0.5 0.5 0.5 0.5
transitions:
0->0 0.25
0->1 0.25
0->2 0.25
0->3 0.25
transition_times:
0 0.1
1 0.2
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            with (data_dir / "generated_settings.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerow(truth_row)
            batch_dir = data_dir / "setting_1" / "batch_0"
            batch_dir.mkdir(parents=True)
            (
                batch_dir / "inference_progressive_joint_first_8_trees.out"
            ).write_text(coordinated_output)

            fit_dir = batch_dir / "diversitree_musse"
            fit_dir.mkdir()
            (fit_dir / "joint_musse_fit.txt").write_text(
                "convergence_code=0\nsampling_fraction=0.5,0.5,0.5,0.5\n"
            )
            with (fit_dir / "parameter_estimates.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(("parameter", "estimate"))
                for index, state in enumerate("ABCD", start=1):
                    writer.writerow((f"lambda{index}", 8))
                    writer.writerow((f"mu{index}", 2))

            failed_fit = data_dir / "setting_1" / "batch_1" / "diversitree_musse"
            failed_fit.mkdir(parents=True)
            (failed_fit / "joint_musse_fit.txt").write_text(
                "convergence_code=52\n"
            )

            beam_dir = batch_dir / "tree_0" / "beam_snakemake"
            beam_dir.mkdir(parents=True)
            (beam_dir / "run_status.tsv").write_text(
                "tree_id\tstatus\ttip_count\tobserved_tissues\tmessage\n"
                "setting_1/batch_0/tree_0\tcompleted\t100\tA,B,C,D\t\n"
            )
            (beam_dir / "parameter_summary.tsv").write_text(
                "tree_id\tparameter\tmean\tmedian\tHPD95_lower\t"
                "HPD95_upper\tsamples_after_burnin\n"
                "setting_1/batch_0/tree_0\tbirthDiffRate\t5\t5\t4\t6\t9001\n"
                "setting_1/batch_0/tree_0\trelativeDeathRate\t0.2\t0.2\t"
                "0.1\t0.3\t9001\n"
            )

            errors = load_matched_absolute_errors(data_dir)
            transition_errors = load_star_transition_time_errors(data_dir, 2)
            probability_errors = load_star_transition_probability_errors(data_dir, 2)
            self.assertEqual(probability_errors, [0.0, 0.0, 0.0, 0.0])
            plot_errors = [
                *errors,
                AbsoluteError(
                    setting="setting_1",
                    batch="batch_0",
                    state="A",
                    method="CONCERT",
                    metric="net_growth",
                    value=1000,
                ),
            ]
            output_path = data_dir / "figures" / "errors.png"
            original_subplots = plt.subplots
            plotted_axes = []

            def capture_axes(*args, **kwargs):
                figure, axes = original_subplots(*args, **kwargs)
                plotted_axes.extend(axes)
                return figure, axes

            with patch(
                "src.plot_star_diversitree_errors.plt.subplots",
                side_effect=capture_axes,
            ):
                plot_error_boxplots(plot_errors, output_path)
            self.assertTrue(output_path.is_file())
            self.assertEqual(len(plotted_axes[0].patches), 3)
            self.assertTrue(
                all(axis.get_yscale() == "linear" for axis in plotted_axes)
            )
            self.assertLess(plotted_axes[0].get_ylim()[1], 1000)
            self.assertEqual(
                [axis.get_title() for axis in plotted_axes],
                ["Net growth", "Scaled birth"],
            )
            self.assertEqual(
                [text.get_text() for text in plotted_axes[0].texts],
                ["(a)"],
            )
            self.assertEqual(
                [tick.get_text() for tick in plotted_axes[0].get_xticklabels()],
                [],
            )

        self.assertEqual(len(errors), 22)
        self.assertEqual(len(transition_errors), 2)
        self.assertEqual(
            {
                (error.method, error.metric, error.state)
                for error in errors
            },
            {
                *(
                    (method, metric, state)
                    for method in ("CONCERT", "MuSSE")
                    for metric in ("net_growth", "scaled_birth")
                    for state in (*"ABCD", "Mean across states")
                ),
                *(
                    ("BEAM", metric, "Mean across states")
                    for metric in ("net_growth", "scaled_birth")
                ),
            },
        )
        coordinated_net_growth_a = next(
            error.value
            for error in errors
            if error.method == "CONCERT"
            and error.metric == "net_growth"
            and error.state == "A"
        )
        self.assertEqual(coordinated_net_growth_a, 1.0)
        beam_mean_net_growth = next(
            error.value
            for error in errors
            if error.method == "BEAM"
            and error.metric == "net_growth"
            and error.state == "Mean across states"
        )
        self.assertEqual(beam_mean_net_growth, 1.0)
        beam_mean_scaled_birth = next(
            error.value
            for error in errors
            if error.method == "BEAM"
            and error.metric == "scaled_birth"
            and error.state == "Mean across states"
        )
        self.assertAlmostEqual(beam_mean_scaled_birth, 0.875)


if __name__ == "__main__":
    unittest.main()
