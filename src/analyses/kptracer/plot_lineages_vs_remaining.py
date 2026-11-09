from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

from src.analyses.kptracer.plot_transition_type_counts import (
    TIME_TOLERANCE,
    _analysis_config,
    _ensure_segment_states,
    _fit_specs,
    _transition_types,
    _unique_segments,
)
from src.analyses.common.profile_data import load_inference_parameters

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "data/KPTracer-Data/processed_data/"
    "transition_type_counts/lineages_vs_remaining"
)
CSV_PATH = OUTPUT_DIRECTORY / "lineages_transitioned_vs_remaining.csv"
PDF_PATH = OUTPUT_DIRECTORY / "lineages_transitioned_vs_remaining.pdf"
COMBINED_PLOT_PATH = (
    OUTPUT_DIRECTORY / "lineages_transitioned_vs_remaining_all_transitions.png"
)
ALLOWED_TRANSITIONS = _transition_types()


def _count_at_boundaries(
    segments: pd.DataFrame,
    transition_times: tuple[float, ...],
) -> tuple[
    dict[tuple[int, str, str], int],
    dict[tuple[int, str], int],
    dict[tuple[int, str], int],
]:
    transitioned: dict[tuple[int, str, str], int] = {}
    remaining: dict[tuple[int, str], int] = {}
    source_before: dict[tuple[int, str], int] = {}
    group_columns = ["tree", "subtree", "original_edge"]
    for _, edge_segments in segments.groupby(group_columns, sort=False):
        ordered = edge_segments.sort_values(
            "segment_start_time",
            ascending=False,
            kind="stable",
        ).to_dict("records")
        for older_segment, younger_segment in zip(ordered, ordered[1:]):
            older_end = float(older_segment["segment_end_time"])
            younger_start = float(younger_segment["segment_start_time"])
            if not np.isclose(
                older_end,
                younger_start,
                rtol=0.0,
                atol=TIME_TOLERANCE,
            ):
                raise ValueError(
                    "Adjacent colored segments do not meet on the same "
                    f"original edge: {older_segment['original_edge']}"
                )
            event_time = (older_end + younger_start) / 2.0
            time_index = min(
                range(len(transition_times)),
                key=lambda index: abs(transition_times[index] - event_time),
            )
            if (
                abs(transition_times[time_index] - event_time)
                > TIME_TOLERANCE
            ):
                raise ValueError(
                    f"Colored boundary at {event_time:.15g} does not match "
                    "a fitted transition time."
                )

            source = str(older_segment["most_likely_state"])
            target = str(younger_segment["most_likely_state"])
            source_key = (time_index, source)
            source_before[source_key] = source_before.get(source_key, 0) + 1
            if source == target:
                remaining[source_key] = remaining.get(source_key, 0) + 1
                continue
            if (source, target) not in ALLOWED_TRANSITIONS:
                raise ValueError(
                    f"Colored state change {source}->{target} is not in the "
                    "configured transition graph."
                )
            key = (time_index, source, target)
            transitioned[key] = transitioned.get(key, 0) + 1
    return transitioned, remaining, source_before


def _collect_records() -> list[dict[str, object]]:
    config = _analysis_config()
    records: list[dict[str, object]] = []
    for fit in _fit_specs():
        parameters = load_inference_parameters(
            fit.inference_path,
            ("T1", "L1", "L2", "L3", "S1"),
        )
        transition_times = tuple(
            float(value) for value in parameters.transition_times
        )
        if not transition_times or not np.all(np.diff(transition_times) > 0):
            raise ValueError(
                f"Transition times for {fit.label} must be strictly increasing."
            )
        segment_path = _ensure_segment_states(fit, config)
        segments = _unique_segments(segment_path)
        transitioned, remaining, source_before = _count_at_boundaries(
            segments,
            transition_times,
        )
        for time_index, carta_time in enumerate(transition_times):
            for source, target in ALLOWED_TRANSITIONS:
                records.append(
                    {
                        "fit": fit.label,
                        "transition_count": len(transition_times),
                        "carta_time": carta_time,
                        "elapsed_time_root_to_tip": 1.0 - carta_time,
                        "transition_type": f"{source}->{target}",
                        "source_state": source,
                        "target_state": target,
                        "lineages_transitioned": transitioned.get(
                            (time_index, source, target),
                            0,
                        ),
                        "lineages_remaining_in_source": remaining.get(
                            (time_index, source),
                            0,
                        ),
                        "source_lineages_before_transition": source_before.get(
                            (time_index, source),
                            0,
                        ),
                    }
                )
    return records


def _correlation_points(
    fit_schedule: pd.DataFrame,
) -> tuple[pd.DataFrame, float | None]:
    zero_transition_points = fit_schedule.loc[
        fit_schedule["lineages_transitioned"] == 0
    ]
    if zero_transition_points.empty:
        correlation_data = fit_schedule
    else:
        maximum_zero_x = zero_transition_points[
            "lineages_remaining_in_source"
        ].max()
        correlation_data = fit_schedule.loc[
            (fit_schedule["lineages_transitioned"] > 0)
            | (
                (fit_schedule["lineages_transitioned"] == 0)
                & (
                    fit_schedule["lineages_remaining_in_source"]
                    >= maximum_zero_x
                )
            )
        ]
    x_values = correlation_data[
        "source_lineages_before_transition"
    ].to_numpy(dtype=float)
    y_values = correlation_data["lineages_transitioned"].to_numpy(dtype=float)
    if (
        len(correlation_data) >= 3
        and np.ptp(x_values) > 0
        and np.ptp(y_values) > 0
    ):
        correlation = float(np.corrcoef(x_values, y_values)[0, 1])
    else:
        correlation = None
    return correlation_data, correlation


def _plot_correlation_line(
    axis,
    correlation_data: pd.DataFrame,
    correlation: float | None,
) -> None:
    if correlation is None:
        return
    x_values = correlation_data[
        "source_lineages_before_transition"
    ].to_numpy(dtype=float)
    y_values = correlation_data["lineages_transitioned"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x_values, y_values, 1)
    line_x = np.array([x_values.min(), x_values.max()])
    axis.plot(
        line_x,
        intercept + slope * line_x,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        zorder=1,
    )


def _plot_transition(
    data: pd.DataFrame,
    source: str,
    target: str,
    fit_labels: list[str],
    pdf: PdfPages,
) -> Path:
    transition_type = f"{source}->{target}"
    figure, axes = plt.subplots(
        len(fit_labels),
        1,
        figsize=(5.25, 2.4 * len(fit_labels)),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    for axis, fit_label in zip(axes, fit_labels):
        fit_schedule = data.loc[
            (data["fit"] == fit_label)
            & (data["transition_type"] == transition_type)
        ]
        correlation_data, correlation = _correlation_points(fit_schedule)
        observation_count = len(correlation_data)
        if correlation is not None:
            correlation_text = f"Pearson r={correlation:.2f}"
        else:
            correlation_text = "Pearson r=N/A"
        axis.scatter(
            fit_schedule["source_lineages_before_transition"],
            fit_schedule["lineages_transitioned"],
            color="black",
            s=35,
            alpha=0.8,
            edgecolors="none",
            zorder=2,
        )
        _plot_correlation_line(axis, correlation_data, correlation)
        axis.set_title(
            f"{fit_label} | {correlation_text} "
            f"(correlation n={observation_count})",
            loc="left",
            fontsize=10,
        )
        if fit_schedule.empty:
            axis.text(
                0.5,
                0.5,
                "No transitions of this type",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.grid(True, alpha=0.25)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_ylabel("Lineages transitioned")

    axes[-1].set_xlabel(
        f"Total lineages in {source} immediately before transition"
    )
    figure.suptitle(
        f"{source}->{target}: transitioned vs. total source lineages",
        fontsize=14,
        y=0.995,
    )
    figure.subplots_adjust(
        left=0.12,
        right=0.96,
        top=0.96,
        bottom=0.06,
        hspace=0.45,
    )
    filename = f"lineages_transitioned_vs_remaining_{source}_to_{target}.png"
    output_path = OUTPUT_DIRECTORY / filename
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_all_transitions(data: pd.DataFrame, fit_labels: list[str]) -> Path:
    figure, axes = plt.subplots(
        len(ALLOWED_TRANSITIONS),
        len(fit_labels),
        figsize=(18, 18),
        squeeze=False,
    )
    for row_index, (source, target) in enumerate(ALLOWED_TRANSITIONS):
        transition_type = f"{source}->{target}"
        transition_data = data.loc[
            data["transition_type"] == transition_type
        ]
        maximum_x = float(
            transition_data["source_lineages_before_transition"].max()
        )
        maximum_y = int(transition_data["lineages_transitioned"].max())
        for column_index, fit_label in enumerate(fit_labels):
            axis = axes[row_index, column_index]
            fit_schedule = transition_data.loc[
                transition_data["fit"] == fit_label
            ]
            correlation_data, correlation = _correlation_points(fit_schedule)
            observation_count = len(correlation_data)
            correlation_text = (
                f"r={correlation:.2f}"
                if correlation is not None
                else "r=N/A"
            )
            axis.scatter(
                fit_schedule["source_lineages_before_transition"],
                fit_schedule["lineages_transitioned"],
                color="black",
                s=18,
                alpha=0.8,
                edgecolors="none",
                zorder=2,
            )
            _plot_correlation_line(axis, correlation_data, correlation)
            axis.text(
                0.04,
                0.94,
                f"{correlation_text} (n={observation_count})",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
            axis.set_xlim(0, max(1.0, maximum_x * 1.04))
            axis.set_ylim(-0.25, maximum_y + 0.5)
            axis.yaxis.set_major_locator(
                MaxNLocator(nbins=4, integer=True)
            )
            axis.tick_params(labelsize=7)
            axis.grid(True, alpha=0.22)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if row_index == 0:
                short_label = (
                    fit_label
                    .replace(", CONCERT initialized", "")
                    .replace(", best completed LL", "")
                )
                axis.set_title(short_label, fontsize=9)
            if column_index == 0:
                axis.set_ylabel(
                    f"{source}->{target}\nTransitioned",
                    fontsize=8,
                )
            if row_index == len(ALLOWED_TRANSITIONS) - 1:
                axis.set_xlabel(
                    f"{source} total before",
                    fontsize=8,
                )
    figure.suptitle(
        "Lineages transitioned vs. total source lineages",
        fontsize=15,
        y=0.995,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.995,
        top=0.96,
        bottom=0.045,
        wspace=0.28,
        hspace=0.45,
    )
    figure.savefig(COMBINED_PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return COMBINED_PLOT_PATH


def main() -> None:
    records = _collect_records()
    data = pd.DataFrame(records)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    data.to_csv(CSV_PATH, index=False)
    fit_labels = list(dict.fromkeys(data["fit"]))
    with PdfPages(PDF_PATH) as pdf:
        plot_paths = [
            _plot_transition(data, source, target, fit_labels, pdf)
            for source, target in ALLOWED_TRANSITIONS
        ]
    combined_plot_path = _plot_all_transitions(data, fit_labels)
    print(f"Wrote {len(plot_paths)} transition-type plots under {OUTPUT_DIRECTORY}")
    print(f"Wrote combined plots to {PDF_PATH}")
    print(f"Wrote all-transition PNG to {combined_plot_path}")
    print(f"Wrote exact-time scatter data to {CSV_PATH}")


if __name__ == "__main__":
    main()
