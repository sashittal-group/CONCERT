from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re
import warnings


STATES = ("A", "B", "C", "D")
JOINT_METHOD = "Joint coordinate search"
MUSSE_METHOD = "diversitree MuSSE"
METHOD_COLORS = {
    JOINT_METHOD: "#0072B2",
    MUSSE_METHOD: "#E69F00",
}
EDGE_PARAMETERS = (
    (0, 1, "A", "B", "p_A_to_B", "transition_time_A_to_B", "transition_time_A"),
    (1, 2, "B", "C", "p_B_to_C", "transition_time_B_to_C", "transition_time_B"),
    (2, 3, "C", "D", "p_C_to_D", "transition_time_C_to_D", "transition_time_C"),
)
LINEAR_TRANSITIONS = (
    ("B", "A_to_B", "p_A_to_B", "q12"),
    ("C", "B_to_C", "p_B_to_C", "q23"),
    ("D", "C_to_D", "p_C_to_D", "q34"),
)
RATE_PARAMETERS = [
    *((f"Net growth {state}", f"net_growth_{state}") for state in STATES),
    *((f"Scaled birth {state}", f"scaled_birth_{state}") for state in STATES),
]
TRANSITION_PARAMETERS = [
    (
        f"Transition probability {source}→{target}",
        probability_key,
    )
    for _, _, source, target, probability_key, _, _ in EDGE_PARAMETERS
]
TIME_PARAMETERS = [
    (
        f"Transition age {source}→{target}",
        time_key,
    )
    for _, _, source, target, _, time_key, _ in EDGE_PARAMETERS
]
PLOT_PARAMETERS = [*RATE_PARAMETERS, *TRANSITION_PARAMETERS, *TIME_PARAMETERS]
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
VECTOR_LINE = re.compile(rf"^(birth|death|sampling):\s+(.+)$")
EDGE_LINE = re.compile(rf"^(\d+)->(\d+)\s+({NUMBER})$")
EPOCH_LINE = re.compile(r"^epoch_\d+\s+(\d+)->(\d+)$")
TIME_LINE = re.compile(rf"^(\d+)->(\d+)\s+({NUMBER})$")


@dataclass(frozen=True)
class ParameterComparison:
    setting: str
    batch: str
    method: str
    parameter: str
    truth: float
    estimate: float


@dataclass(frozen=True)
class TransitionTimeError:
    setting: str
    batch: str
    edge: str
    value: float


@dataclass(frozen=True)
class TransitionProbabilityError:
    setting: str
    batch: str
    edge: str
    value: float


def _read_ground_truth(path: Path) -> dict[str, dict[str, float]]:
    required_columns = {
        "setting",
        "sampling_rate",
        *(f"birth_{state}" for state in STATES),
        *(f"death_{state}" for state in STATES),
        "p_A_to_B",
        "p_B_to_C",
        "p_C_to_D",
        "transition_time_A",
        "transition_time_B",
        "transition_time_C",
    }
    truth_by_setting = {}
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"Missing linear-model ground-truth columns in {path}")
        for row in reader:
            setting = row["setting"]
            if setting in truth_by_setting:
                raise ValueError(f"Duplicate ground-truth row for {setting}")
            try:
                values = {
                    key: float(row[key])
                    for key in required_columns
                    if key != "setting"
                }
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid ground-truth value for {setting}") from error
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Non-finite ground-truth value for {setting}")

            sampling_rate = values["sampling_rate"]
            for state in STATES:
                birth = values[f"birth_{state}"]
                death = values[f"death_{state}"]
                values[f"net_growth_{state}"] = birth - death
                values[f"scaled_birth_{state}"] = birth * sampling_rate

            for (
                source_id,
                target_id,
                source,
                target,
                _,
                time_key,
                source_time_key,
            ) in EDGE_PARAMETERS:
                values[time_key] = 1.0 - values[source_time_key]
            truth_by_setting[setting] = values
    return truth_by_setting


def _read_joint_estimates(path: Path) -> dict[str, float]:
    lines = path.read_text().splitlines()
    likelihood_lines = [line for line in lines if line.startswith("LL=")]
    if len(likelihood_lines) != 1 or not math.isfinite(
        float(likelihood_lines[0].split("=", maxsplit=1)[1])
    ):
        raise ValueError(f"Missing or invalid likelihood in {path}")

    vectors = {}
    for line in lines:
        match = VECTOR_LINE.fullmatch(line)
        if match is None:
            continue
        name, raw_values = match.groups()
        try:
            values = tuple(float(value) for value in raw_values.split())
        except ValueError as error:
            raise ValueError(f"Invalid {name} vector in {path}") from error
        if len(values) != len(STATES) or not all(
            math.isfinite(value) for value in values
        ):
            raise ValueError(f"Expected four finite {name} values in {path}")
        vectors[name] = values
    if set(vectors) != {"birth", "death", "sampling"}:
        raise ValueError(f"Missing rate vectors in {path}")

    matrices_by_edge: dict[tuple[int, int], dict[tuple[int, int], float]] = {}
    times_by_edge: dict[tuple[int, int], float] = {}
    current_epoch_edge = None
    section = None
    for line in lines:
        stripped = line.strip()
        if stripped == "transition_probabilities_by_epoch:":
            section = "probabilities"
            continue
        if stripped == "transition_times_by_edge:":
            section = "times"
            current_epoch_edge = None
            continue
        if not stripped:
            continue
        if section == "probabilities":
            epoch_match = EPOCH_LINE.fullmatch(stripped)
            if epoch_match is not None:
                current_epoch_edge = tuple(
                    int(value) for value in epoch_match.groups()
                )
                if current_epoch_edge in matrices_by_edge:
                    raise ValueError(f"Duplicate epoch edge in {path}")
                matrices_by_edge[current_epoch_edge] = {}
                continue
            edge_match = EDGE_LINE.fullmatch(stripped)
            if edge_match is None or current_epoch_edge is None:
                continue
            source, target, raw_probability = edge_match.groups()
            matrices_by_edge[current_epoch_edge][
                (int(source), int(target))
            ] = float(raw_probability)
        elif section == "times":
            time_match = TIME_LINE.fullmatch(stripped)
            if time_match is None:
                continue
            source, target, raw_time = time_match.groups()
            edge = (int(source), int(target))
            if edge in times_by_edge:
                raise ValueError(f"Duplicate transition time for {edge} in {path}")
            times_by_edge[edge] = float(raw_time)

    expected_edges = {
        (source_id, target_id)
        for source_id, target_id, *_ in EDGE_PARAMETERS
    }
    if set(matrices_by_edge) != expected_edges or set(times_by_edge) != expected_edges:
        raise ValueError(f"Missing linear transition estimates in {path}")

    estimates = {}
    for index, state in enumerate(STATES):
        estimates[f"birth_{state}"] = vectors["birth"][index]
        estimates[f"death_{state}"] = vectors["death"][index]
        estimates[f"sampling_{state}"] = vectors["sampling"][index]
        estimates[f"net_growth_{state}"] = (
            vectors["birth"][index] - vectors["death"][index]
        )
        estimates[f"scaled_birth_{state}"] = (
            vectors["birth"][index] * vectors["sampling"][index]
        )

    for (
        source_id,
        target_id,
        source,
        target,
        probability_key,
        time_key,
        _,
    ) in EDGE_PARAMETERS:
        edge = (source_id, target_id)
        try:
            estimates[probability_key] = matrices_by_edge[edge][edge]
            estimates[time_key] = times_by_edge[edge]
        except KeyError as error:
            raise ValueError(f"Missing estimate for edge {edge} in {path}") from error
    if not all(math.isfinite(value) for value in estimates.values()):
        raise ValueError(f"Non-finite estimate in {path}")
    return estimates


def _read_diversitree_estimates(
    summary_path: Path,
) -> dict[str, float]:
    summary = {
        key: value
        for line in summary_path.read_text().splitlines()
        if "=" in line
        for key, value in [line.split("=", maxsplit=1)]
    }
    if summary.get("transition_model") != "linear":
        raise ValueError(f"Expected a linear MuSSE fit in {summary_path}")
    if summary.get("convergence_code") != "0":
        raise ValueError(f"MuSSE fit did not converge in {summary_path}")
    try:
        sampling_fractions = tuple(
            float(value) for value in summary["sampling_fraction"].split(",")
        )
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid sampling fractions in {summary_path}") from error
    if len(sampling_fractions) != len(STATES) or not all(
        math.isfinite(value) for value in sampling_fractions
    ):
        raise ValueError(f"Expected four finite sampling fractions in {summary_path}")

    estimates_path = summary_path.with_name("parameter_estimates.csv")
    with estimates_path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if any(not row.get("parameter") or not row.get("estimate") for row in rows):
        raise ValueError(f"Invalid parameter estimate row in {estimates_path}")
    estimates = {row["parameter"]: float(row["estimate"]) for row in rows}
    if len(estimates) != len(rows):
        raise ValueError(f"Duplicate parameter names in {estimates_path}")

    derived = {}
    for index, state in enumerate(STATES, start=1):
        birth = estimates.get(f"lambda{index}")
        death = estimates.get(f"mu{index}")
        if birth is None or death is None:
            raise ValueError(f"Missing MuSSE rate estimates for state {index}")
        sampling = sampling_fractions[index - 1]
        derived[f"net_growth_{state}"] = birth - death
        derived[f"scaled_birth_{state}"] = birth * sampling
    for _, edge, _, parameter in LINEAR_TRANSITIONS:
        rate = estimates.get(parameter)
        if rate is None or not math.isfinite(rate) or rate < 0.0:
            raise ValueError(f"Missing or invalid {parameter} estimate in {estimates_path}")
        derived[f"transition_rate_{edge}"] = rate
    if not all(math.isfinite(value) for value in derived.values()):
        raise ValueError(f"Non-finite MuSSE estimate in {estimates_path}")
    return derived


def load_parameter_comparisons(data_dir: Path) -> list[ParameterComparison]:
    truth_by_setting = _read_ground_truth(data_dir / "generated_settings.csv")
    comparisons = []

    joint_paths = sorted(
        data_dir.glob(
            "setting_*/batch_*/inference_joint_coordinate_first_8_trees.out"
        )
    )
    for path in joint_paths:
        setting = path.parent.parent.name
        if setting not in truth_by_setting:
            warnings.warn(f"No ground truth for {setting}; skipping {path}")
            continue
        try:
            estimates = _read_joint_estimates(path)
        except (OSError, ValueError) as error:
            warnings.warn(f"Skipping incomplete joint output {path}: {error}")
            continue
        truth = truth_by_setting[setting]
        for _, parameter in PLOT_PARAMETERS:
            comparisons.append(
                ParameterComparison(
                    setting=setting,
                    batch=path.parent.name,
                    method=JOINT_METHOD,
                    parameter=parameter,
                    truth=truth[parameter],
                    estimate=estimates[parameter],
                )
            )

    diversitree_paths = sorted(
        data_dir.glob(
            "setting_*/batch_*/diversitree_musse/joint_musse_fit.txt"
        )
    )
    for path in diversitree_paths:
        setting = path.parent.parent.parent.name
        if setting not in truth_by_setting:
            warnings.warn(f"No ground truth for {setting}; skipping {path}")
            continue
        try:
            estimates = _read_diversitree_estimates(path)
        except (OSError, ValueError) as error:
            warnings.warn(f"Skipping incomplete MuSSE output {path}: {error}")
            continue
        truth = truth_by_setting[setting]
        for _, parameter in RATE_PARAMETERS:
            comparisons.append(
                ParameterComparison(
                    setting=setting,
                    batch=path.parent.parent.name,
                    method=MUSSE_METHOD,
                    parameter=parameter,
                    truth=truth[parameter],
                    estimate=estimates[parameter],
                )
            )

    if not comparisons:
        raise FileNotFoundError(
            f"No complete linear joint or MuSSE fits found in {data_dir}"
        )
    return comparisons


def _r_squared(truth: list[float], estimates: list[float]) -> float | None:
    if len(truth) != len(estimates) or not truth:
        raise ValueError("R² requires equally sized, non-empty values")
    mean = sum(truth) / len(truth)
    total_sum_squares = sum((value - mean) ** 2 for value in truth)
    if math.isclose(total_sum_squares, 0.0, abs_tol=1e-15):
        return None
    residual_sum_squares = sum(
        (actual - estimate) ** 2 for actual, estimate in zip(truth, estimates)
    )
    return 1.0 - residual_sum_squares / total_sum_squares


def load_concert_transition_time_errors(
    data_dir: Path,
) -> list[TransitionTimeError]:
    truth_by_setting = _read_ground_truth(data_dir / "generated_settings.csv")
    edge_parameters = tuple(
        (f"{source}→{target}", time_key)
        for _, _, source, target, _, time_key, _ in EDGE_PARAMETERS
    )
    errors = []
    for path in sorted(
        data_dir.glob(
            "setting_*/batch_*/inference_joint_coordinate_first_8_trees.out"
        )
    ):
        setting = path.parent.parent.name
        truth = truth_by_setting.get(setting)
        if truth is None:
            warnings.warn(f"No ground truth for {setting}; skipping {path}")
            continue
        try:
            estimates = _read_joint_estimates(path)
        except (OSError, ValueError) as error:
            warnings.warn(f"Skipping incomplete joint output {path}: {error}")
            continue
        for edge, parameter in edge_parameters:
            errors.append(
                TransitionTimeError(
                    setting=setting,
                    batch=path.parent.name,
                    edge=edge,
                    value=abs(estimates[parameter] - truth[parameter]),
                )
            )

    if not errors:
        raise FileNotFoundError(
            f"No complete CONCERT transition-time fits found in {data_dir}"
        )
    return errors


def load_concert_transition_probability_errors(
    data_dir: Path,
) -> list[TransitionProbabilityError]:
    truth_by_setting = _read_ground_truth(data_dir / "generated_settings.csv")
    edge_parameters = tuple(
        (f"{source}→{target}", probability_key)
        for _, _, source, target, probability_key, _, _ in EDGE_PARAMETERS
    )
    errors = []
    for path in sorted(
        data_dir.glob(
            "setting_*/batch_*/inference_joint_coordinate_first_8_trees.out"
        )
    ):
        setting = path.parent.parent.name
        truth = truth_by_setting.get(setting)
        if truth is None:
            warnings.warn(f"No ground truth for {setting}; skipping {path}")
            continue
        try:
            estimates = _read_joint_estimates(path)
        except (OSError, ValueError) as error:
            warnings.warn(f"Skipping incomplete joint output {path}: {error}")
            continue
        for edge, parameter in edge_parameters:
            errors.append(
                TransitionProbabilityError(
                    setting=setting,
                    batch=path.parent.name,
                    edge=edge,
                    value=abs(estimates[parameter] - truth[parameter]),
                )
            )

    if not errors:
        raise FileNotFoundError(
            f"No complete CONCERT transition-probability fits found in {data_dir}"
        )
    return errors


def plot_concert_transition_time_errors(
    data_dir: Path,
    output_path: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cbook import boxplot_stats

    errors = load_concert_transition_time_errors(data_dir)
    values = [error.value for error in errors]
    color = "#0072B2"
    figure, axis = plt.subplots(figsize=(5.8, 3.8))
    box = axis.boxplot(
        [values],
        positions=[1.5],
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "#222222", "linewidth": 1.5},
        whiskerprops={"color": color, "linewidth": 1.1},
        capprops={"color": color, "linewidth": 1.1},
        flierprops={
            "marker": "o",
            "markerfacecolor": color,
            "markeredgecolor": color,
            "markersize": 2.5,
            "markeredgewidth": 0.35,
            "alpha": 0.55,
        },
    )
    box["boxes"][0].set_facecolor(color)
    box["boxes"][0].set_alpha(0.75)
    box["boxes"][0].set_edgecolor("#333333")
    box["boxes"][0].set_linewidth(0.9)
    upper_whisker = boxplot_stats([values])[0]["whishi"]
    axis.set_xticks([1.5], [f"All transition times\n(n={len(values)} estimates)"])
    axis.set_xlim(0.5, 2.5)
    axis.set_xlabel("A→B, B→C, and C→D pooled")
    axis.set_ylabel("Absolute error in transition age")
    upper_limit = upper_whisker
    axis.set_ylim(0, upper_limit * 1.06 if upper_limit > 0 else 1.0)
    axis.set_title("CONCERT transition-time inference error", fontsize=10, pad=8)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.65)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=8, length=3)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)

    run_count = len({(error.setting, error.batch) for error in errors})
    figure.subplots_adjust(left=0.16, right=0.98, bottom=0.27, top=0.86)
    figure.text(
        0.5,
        0.015,
        f"Linear A-D: {run_count} CONCERT batches. Errors use backward-time "
        "ages; y-limits end at the largest 1.5-IQR whisker, omitting higher outliers.",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Plotted {len(errors)} transition-time errors to {output_path}")
    return output_path


def plot_identifiable_parameters(
    data_dir: Path,
    output_path: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    comparisons = load_parameter_comparisons(data_dir)
    row_count = math.ceil(len(PLOT_PARAMETERS) / 4)
    figure, axes = plt.subplots(
        row_count,
        4,
        figsize=(16, row_count * 3.4),
        squeeze=False,
    )

    for index, (label, parameter) in enumerate(PLOT_PARAMETERS):
        axis = axes.flat[index]
        points = [
            item for item in comparisons if item.parameter == parameter
        ]
        if not points:
            axis.set_visible(False)
            continue
        all_values = [value for point in points for value in (point.truth, point.estimate)]
        lower = min(all_values)
        upper = max(all_values)
        padding = max((upper - lower) * 0.05, 0.05)
        lower -= padding
        upper += padding

        score_lines = []
        for method, color in METHOD_COLORS.items():
            method_points = [point for point in points if point.method == method]
            if not method_points:
                continue
            truth = [point.truth for point in method_points]
            estimates = [point.estimate for point in method_points]
            axis.scatter(
                truth,
                estimates,
                s=22,
                alpha=0.72,
                color=color,
                edgecolors="none",
            )
            score = _r_squared(truth, estimates)
            score_text = "undefined" if score is None else f"{score:.3f}"
            score_lines.append(
                f"{method}: R²={score_text}, n={len(method_points)}"
            )

        axis.plot(
            [lower, upper],
            [lower, upper],
            color="#444444",
            linestyle="--",
            linewidth=1,
        )
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(label)
        axis.set_xlabel("Ground truth")
        axis.set_ylabel("Inferred")
        axis.text(
            0.04,
            0.96,
            "\n".join(score_lines),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        axis.grid(True, alpha=0.2)

    for axis in axes.flat[len(PLOT_PARAMETERS):]:
        axis.set_visible(False)

    handles = [
        Patch(
            facecolor=color,
            edgecolor=color,
            alpha=0.72,
            label=method,
        )
        for method, color in METHOD_COLORS.items()
        if any(point.method == method for point in comparisons)
    ]
    figure.legend(handles=handles, loc="upper center", ncol=len(handles))
    figure.suptitle(
        "Identifiable parameters: ground truth vs. inferred\n"
        "Transition ages use backward-time coordinates; MuSSE transition "
        "rates are not directly comparable to discrete transition probabilities."
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Plotted {len(comparisons)} parameter comparisons to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot identifiable linear-transition model estimates against "
            "simulator ground truth."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/simulation/coord_diff/linear_A_to_D"),
    )
    parser.add_argument(
        "--transition-time-errors",
        action="store_true",
        help="Plot absolute CONCERT transition-time errors instead of all parameters.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.transition_time_errors:
        output_path = args.output or (
            Path("outputs/simulation/linear_A_to_D/figures")
            / "concert_transition_time_error_boxplots.png"
        )
        plot_concert_transition_time_errors(args.data_dir, output_path)
        return
    output_path = args.output or (
        Path("outputs/simulation/linear_A_to_D/figures")
        / "identifiable_parameters_ground_truth_vs_inferred.png"
    )
    plot_identifiable_parameters(args.data_dir, output_path)


if __name__ == "__main__":
    main()
