from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
from matplotlib.cbook import boxplot_stats
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from src.plot_star_inference_accuracy import (
    STATES,
    _add_identifiable_parameters,
    _read_ground_truth,
    load_completed_progressive_comparisons,
)
from src.plot_linear_transition_inference_accuracy import (
    LINEAR_TRANSITIONS,
    TransitionTimeError,
    _read_diversitree_estimates as _read_linear_diversitree_estimates,
    _read_ground_truth as _read_linear_ground_truth,
    _read_joint_estimates as _read_linear_joint_estimates,
    load_concert_transition_probability_errors,
    load_concert_transition_time_errors,
)
from src.plot_star_transition_rates_vs_probabilities import (
    RateObservation,
    _read_beam_rates,
    calibrate_rates_to_probabilities,
    collect_rate_observations,
)
from src.coord_diff_top_batches import (
    LINEAR_TOP_TWENTY_BATCHES,
    STAR3_TOP_TWENTY_BATCHES,
)


DEFAULT_DATA_ROOT = Path("data/simulation/coord_diff")
TRANSITION_TIME_ERROR_YLIM = (0.0, 0.2)
METHODS = ("CONCERT", "MuSSE", "BEAM")
BOX_WIDTH = 0.22
BOX_EDGE_COLOR = "#3F3F3F"
BOX_EDGE_WIDTH = 0.8
BOX_FILL_ALPHA = 0.62
MEDIAN_LINE_WIDTH = 1.7
WHISKER_LINE_WIDTH = 1.2
METHOD_DISPLAY_NAMES = {
    "CONCERT": "CONCERT",
    "MuSSE": "diversitree",
    "BEAM": "BEAM",
    "MACH2": "MACH2 (edge midpoint)",
}
METHOD_COLORS = {
    "CONCERT": "#0072B2",
    "MuSSE": "#E69F00",
    "BEAM": "#009E73",
    "MACH2": "#D55E00",
}
METRICS = (
    ("Net growth", "net_growth"),
    ("Scaled birth", "scaled_birth"),
)
MEAN_STATE = "Mean across states"
LINEAR_BEAM_RATE_COLUMNS = {
    "B": "tissueSubstModelLogger.relMigRate_A_B",
    "C": "tissueSubstModelLogger.relMigRate_B_C",
    "D": "tissueSubstModelLogger.relMigRate_C_D",
}


@dataclass(frozen=True)
class AbsoluteError:
    setting: str
    batch: str
    state: str
    method: str
    metric: str
    value: float
    tree: str | None = None


@dataclass(frozen=True)
class CombinedErrorPlotRow:
    label: str
    errors: list[AbsoluteError]
    transition_time_errors: list[TransitionTimeError]
    mach2_transition_time_errors: list[Mach2BatchTransitionTimeError]
    transition_probability_errors: list[float]
    diversitree_transition_probability_errors: list[float]
    beam_transition_probability_errors: list[float]
    transition_rank_correlations: list[TransitionRankCorrelation] = field(
        default_factory=list
    )
    transition_correlations: list[TransitionPearsonCorrelation] = field(
        default_factory=list
    )
    common_batch_keys: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class Mach2BatchTransitionTimeError:
    setting: str
    batch: str
    value: float
    tree_count: int
    transition_count: int


@dataclass(frozen=True)
class TransitionRankCorrelation:
    setting: str
    batch: str
    method: str
    value: float | None
    order_error_rate: float


@dataclass(frozen=True)
class TransitionPearsonCorrelation:
    setting: str
    batch: str
    method: str
    value: float


def _read_diversitree_estimates(
    summary_path: Path,
) -> tuple[dict[str, float], tuple[float, ...]]:
    summary = {
        key: value
        for line in summary_path.read_text().splitlines()
        if "=" in line
        for key, value in [line.split("=", maxsplit=1)]
    }
    if summary.get("convergence_code") != "0":
        raise ValueError(f"Expected a converged diversitree fit in {summary_path}")

    try:
        sampling_fractions = tuple(
            float(value) for value in summary["sampling_fraction"].split(",")
        )
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid sampling fractions in {summary_path}") from error
    if len(sampling_fractions) != len(STATES):
        raise ValueError(f"Expected four sampling fractions in {summary_path}")

    estimates_path = summary_path.with_name("parameter_estimates.csv")
    with estimates_path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if any(not row.get("parameter") or not row.get("estimate") for row in rows):
        raise ValueError(f"Invalid parameter estimate row in {estimates_path}")
    estimates = {row["parameter"]: float(row["estimate"]) for row in rows}
    if len(estimates) != len(rows):
        raise ValueError(f"Duplicate parameter names in {estimates_path}")
    for parameter in (
        *(f"lambda{index}" for index in range(1, len(STATES) + 1)),
        *(f"mu{index}" for index in range(1, len(STATES) + 1)),
    ):
        if parameter not in estimates:
            raise ValueError(f"Missing {parameter} in {estimates_path}")
    return estimates, sampling_fractions


def _derived_rates(
    birth: tuple[float, ...],
    death: tuple[float, ...],
    sampling: tuple[float, ...],
) -> dict[str, tuple[float, ...]]:
    return {
        "net_growth": tuple(
            birth_rate - death_rate
            for birth_rate, death_rate in zip(birth, death)
        ),
        "scaled_birth": tuple(
            birth_rate * sampling_rate
            for birth_rate, sampling_rate in zip(birth, sampling)
        ),
    }


def _read_beam_rate_estimates(
    summary_path: Path,
) -> tuple[float, float]:
    values = {}
    with summary_path.open(newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            parameter = row.get("parameter")
            if parameter not in {"birthDiffRate", "relativeDeathRate"}:
                continue
            if parameter in values:
                raise ValueError(
                    f"Duplicate BEAM parameter {parameter} in {summary_path}"
                )
            try:
                values[parameter] = float(row["mean"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid BEAM estimate for {parameter} in {summary_path}"
                ) from error

    if set(values) != {"birthDiffRate", "relativeDeathRate"}:
        raise ValueError(f"Missing BEAM birth/death estimates in {summary_path}")
    net_growth = values["birthDiffRate"]
    relative_death = values["relativeDeathRate"]
    if (
        not math.isfinite(net_growth)
        or not math.isfinite(relative_death)
        or net_growth <= 0.0
        or not 0.0 <= relative_death < 1.0
    ):
        raise ValueError(f"Invalid BEAM birth/death estimates in {summary_path}")
    birth_rate = net_growth / (1.0 - relative_death)
    if not math.isfinite(birth_rate):
        raise ValueError(f"Non-finite BEAM birth rate in {summary_path}")
    return net_growth, birth_rate


def _append_rate_errors(
    errors: list[AbsoluteError],
    setting: str,
    batch: str,
    method: str,
    estimates: dict[str, tuple[float, ...]],
    truth: dict[str, tuple[float, ...]],
    tree: str | None = None,
) -> None:
    for _, metric in METRICS:
        estimated_values = estimates[metric]
        truth_values = truth[metric]
        for state_index, state in enumerate(STATES):
            errors.append(
                AbsoluteError(
                    setting=setting,
                    batch=batch,
                    state=state,
                    method=method,
                    metric=metric,
                    value=abs(
                        estimated_values[state_index] - truth_values[state_index]
                    ),
                    tree=tree,
                )
            )
        errors.append(
            AbsoluteError(
                setting=setting,
                batch=batch,
                state=MEAN_STATE,
                method=method,
                metric=metric,
                value=abs(
                    sum(estimated_values) / len(STATES)
                    - sum(truth_values) / len(STATES)
                ),
                tree=tree,
            )
        )


def load_matched_absolute_errors(
    data_dir: Path,
    num_transition_times: int = 2,
) -> list[AbsoluteError]:
    truth_by_setting = _read_ground_truth(
        data_dir / "generated_settings.csv",
        num_transition_times=num_transition_times,
    )
    truth_by_setting = {
        setting: _add_identifiable_parameters(values)
        for setting, values in truth_by_setting.items()
    }
    coordinated = {
        (comparison.setting, comparison.batch): comparison
        for comparison in load_completed_progressive_comparisons(
            data_dir,
            num_transition_times=num_transition_times,
        )
    }

    errors = []
    matched_batches = set()
    for summary_path in sorted(
        data_dir.glob("setting_*/batch_*/diversitree_musse/joint_musse_fit.txt")
    ):
        summary_values = {
            key: value
            for line in summary_path.read_text().splitlines()
            if "=" in line
            for key, value in [line.split("=", maxsplit=1)]
        }
        if summary_values.get("convergence_code") != "0":
            continue

        batch_dir = summary_path.parent.parent
        setting = batch_dir.parent.name
        batch = batch_dir.name
        key = (setting, batch)
        comparison = coordinated.get(key)
        truth = truth_by_setting.get(setting)
        if comparison is None or truth is None:
            continue
        matched_batches.add(key)

        estimates, sampling_fractions = _read_diversitree_estimates(summary_path)
        diversitree_rates = _derived_rates(
            tuple(estimates[f"lambda{index}"] for index in range(1, 5)),
            tuple(estimates[f"mu{index}"] for index in range(1, 5)),
            sampling_fractions,
        )
        coordinated_values = comparison.inferred
        coordinated_rates = _derived_rates(
            tuple(coordinated_values[f"birth_{state}"] for state in STATES),
            tuple(coordinated_values[f"death_{state}"] for state in STATES),
            tuple(coordinated_values[f"sampling_{state}"] for state in STATES),
        )
        truth_rates = {
            "net_growth": tuple(
                truth[f"net_growth_{state}"] for state in STATES
            ),
            "scaled_birth": tuple(
                truth[f"birth_{state}"] * truth[f"sampling_{state}"]
                for state in STATES
            ),
        }

        for method, rates in (
            ("CONCERT", coordinated_rates),
            ("MuSSE", diversitree_rates),
        ):
            _append_rate_errors(
                errors,
                setting,
                batch,
                method,
                rates,
                truth_rates,
            )

    for summary_path in sorted(
        data_dir.glob(
            "setting_*/batch_*/tree_*/beam_snakemake/parameter_summary.tsv"
        )
    ):
        tree_dir = summary_path.parent.parent
        batch_dir = tree_dir.parent
        setting = batch_dir.parent.name
        batch = batch_dir.name
        tree = tree_dir.name
        if (setting, batch) not in matched_batches:
            continue
        if setting not in truth_by_setting:
            warnings.warn(f"No ground truth for {setting}; skipping {summary_path}")
            continue

        status_path = summary_path.with_name("run_status.tsv")
        try:
            with status_path.open(newline="") as source:
                status_rows = list(csv.DictReader(source, delimiter="\t"))
            if len(status_rows) != 1 or status_rows[0].get("status") != "completed":
                raise ValueError(f"BEAM run is not marked completed in {status_path}")
            net_growth, birth_rate = _read_beam_rate_estimates(summary_path)
        except (OSError, ValueError) as error:
            warnings.warn(f"Skipping incomplete BEAM output {summary_path}: {error}")
            continue

        truth = truth_by_setting[setting]
        sampling_rate = sum(
            truth[f"sampling_{state}"] for state in STATES
        ) / len(STATES)
        beam_rates = {
            "net_growth": (net_growth,),
            "scaled_birth": (birth_rate * sampling_rate,),
        }
        mean_truth = {
            "net_growth": (
                sum(truth[f"net_growth_{state}"] for state in STATES)
                / len(STATES),
            ),
            "scaled_birth": (
                sum(
                    truth[f"birth_{state}"] * truth[f"sampling_{state}"]
                    for state in STATES
                )
                / len(STATES),
            ),
        }
        for _, metric in METRICS:
            errors.append(
                AbsoluteError(
                    setting=setting,
                    batch=batch,
                    state=MEAN_STATE,
                    method="BEAM",
                    metric=metric,
                    value=abs(beam_rates[metric][0] - mean_truth[metric][0]),
                    tree=tree,
                )
            )

    if not errors:
        raise ValueError(
            f"No matched, converged star inference fits found in {data_dir}"
        )
    return errors


def load_matched_linear_absolute_errors(data_dir: Path) -> list[AbsoluteError]:
    truth_by_setting = _read_linear_ground_truth(data_dir / "generated_settings.csv")
    concert_by_batch = {}
    for path in sorted(
        data_dir.glob(
            "setting_*/batch_*/inference_joint_coordinate_first_8_trees.out"
        )
    ):
        setting = path.parent.parent.name
        if setting not in truth_by_setting:
            warnings.warn(f"No ground truth for {setting}; skipping {path}")
            continue
        try:
            estimates = _read_linear_joint_estimates(path)
        except (OSError, ValueError) as error:
            warnings.warn(f"Skipping incomplete CONCERT output {path}: {error}")
            continue
        concert_by_batch[(setting, path.parent.name)] = estimates

    errors = []
    matched_batches = set()
    for summary_path in sorted(
        data_dir.glob(
            "setting_*/batch_*/diversitree_musse_first8/joint_musse_fit.txt"
        )
    ):
        batch_dir = summary_path.parent.parent
        setting = batch_dir.parent.name
        batch = batch_dir.name
        key = (setting, batch)
        concert_estimates = concert_by_batch.get(key)
        truth = truth_by_setting.get(setting)
        if concert_estimates is None or truth is None:
            continue
        try:
            musse_estimates = _read_linear_diversitree_estimates(summary_path)
        except (OSError, ValueError) as error:
            warnings.warn(f"Skipping incomplete MuSSE output {summary_path}: {error}")
            continue

        matched_batches.add(key)
        truth_rates = {
            metric: tuple(truth[f"{metric}_{state}"] for state in STATES)
            for _, metric in METRICS
        }
        for method, rate_estimates in (
            ("CONCERT", concert_estimates),
            ("MuSSE", musse_estimates),
        ):
            rates = {
                metric: tuple(
                    rate_estimates[f"{metric}_{state}"] for state in STATES
                )
                for _, metric in METRICS
            }
            _append_rate_errors(
                errors,
                setting,
                batch,
                method,
                rates,
                truth_rates,
            )

    for summary_path in sorted(
        data_dir.glob(
            "setting_*/batch_*/tree_*/beam_snakemake/parameter_summary.tsv"
        )
    ):
        tree_dir = summary_path.parent.parent
        batch_dir = tree_dir.parent
        setting = batch_dir.parent.name
        batch = batch_dir.name
        tree = tree_dir.name
        if (setting, batch) not in matched_batches:
            continue

        status_path = summary_path.with_name("run_status.tsv")
        try:
            with status_path.open(newline="") as source:
                status_rows = list(csv.DictReader(source, delimiter="\t"))
            if len(status_rows) != 1 or status_rows[0].get("status") != "completed":
                raise ValueError(f"BEAM run is not marked completed in {status_path}")
            net_growth, birth_rate = _read_beam_rate_estimates(summary_path)
        except (OSError, ValueError) as error:
            warnings.warn(f"Skipping incomplete BEAM output {summary_path}: {error}")
            continue

        truth = truth_by_setting[setting]
        sampling_rate = truth["sampling_rate"]
        beam_rates = {
            "net_growth": (net_growth,),
            "scaled_birth": (birth_rate * sampling_rate,),
        }
        mean_truth = {
            "net_growth": (
                sum(truth[f"net_growth_{state}"] for state in STATES) / len(STATES),
            ),
            "scaled_birth": (
                sum(
                    truth[f"birth_{state}"] * sampling_rate for state in STATES
                )
                / len(STATES),
            ),
        }
        for _, metric in METRICS:
            errors.append(
                AbsoluteError(
                    setting=setting,
                    batch=batch,
                    state=MEAN_STATE,
                    method="BEAM",
                    metric=metric,
                    value=abs(beam_rates[metric][0] - mean_truth[metric][0]),
                    tree=tree,
                )
            )

    if not errors:
        raise ValueError(
            f"No matched, converged linear inference fits found in {data_dir}"
        )
    return errors


def load_star_transition_time_errors(
    data_dir: Path,
    num_transition_times: int,
) -> list[TransitionTimeError]:
    comparisons = load_completed_progressive_comparisons(
        data_dir,
        num_transition_times=num_transition_times,
    )
    return [
        TransitionTimeError(
            setting=comparison.setting,
            batch=comparison.batch,
            edge=f"Transition time {index}",
            value=abs(
                comparison.inferred[f"transition_time_{index}"]
                - comparison.truth[f"transition_time_{index}"]
            ),
        )
        for comparison in comparisons
        for index in range(1, num_transition_times + 1)
    ]


def _read_mach2_tree_transition_time_error(
    refined_tree_path: Path,
) -> tuple[float, int] | None:
    tree_dir = refined_tree_path.parents[2]
    node_times: dict[int, float] = {}
    mach2_to_tree_node: dict[int, int] = {}
    with (tree_dir / "nodes.csv").open(newline="") as source:
        for mach2_node_id, row in enumerate(csv.DictReader(source), start=1):
            tree_node_id = int(row["node_id"])
            node_time = float(row["T"])
            if not math.isfinite(node_time):
                raise ValueError(f"Non-finite node age in {tree_dir / 'nodes.csv'}")
            mach2_to_tree_node[mach2_node_id] = tree_node_id
            node_times[tree_node_id] = node_time

    tree_edges = set()
    with (tree_dir / "edges.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            tree_edges.add((int(row["parent"]), int(row["child"])))

    truth_path = tree_dir / "transition_times_with_edges.csv"
    with truth_path.open(newline="") as source:
        truth_rows = list(csv.reader(source))
    try:
        expected_count = int(truth_rows[0][0])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Invalid ground-truth transition count in {truth_path}") from error
    if expected_count != len(truth_rows) - 1:
        raise ValueError(f"Transition count does not match rows in {truth_path}")
    truth_times = []
    for row in truth_rows[1:]:
        if len(row) != 5:
            raise ValueError(f"Invalid ground-truth transition row in {truth_path}")
        try:
            transition_time = float(row[0])
            int(row[1])
            int(row[2])
            int(row[3])
            int(row[4])
        except ValueError as error:
            raise ValueError(
                f"Invalid ground-truth transition row in {truth_path}"
            ) from error
        if not math.isfinite(transition_time):
            raise ValueError(f"Non-finite transition time in {truth_path}")
        truth_times.append(transition_time)
    if not truth_times:
        raise ValueError(f"No ground-truth transitions in {truth_path}")

    prefix = Path(str(refined_tree_path)[: -len(".refined.tree")])
    origin_path = Path(f"{prefix}.nodes")
    node_of_origin = {}
    with origin_path.open(newline="") as source:
        for line in source:
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"Invalid node-of-origin row in {origin_path}")
            output_node, mach2_node = fields
            node_of_origin[output_node] = int(mach2_node)

    tree_error_values = []
    for line in refined_tree_path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"Invalid refined-tree edge in {refined_tree_path}")
        source, target, raw_timestamp = fields
        try:
            timestamp = int(raw_timestamp)
        except ValueError as error:
            raise ValueError(
                f"Invalid migration timestamp in {refined_tree_path}"
            ) from error
        if timestamp == -1:
            continue
        try:
            original_source = mach2_to_tree_node[
                node_of_origin[source]
            ]
            original_target = mach2_to_tree_node[
                node_of_origin[target]
            ]
        except KeyError as error:
            raise ValueError(
                f"Missing node mapping for migration in {refined_tree_path}"
            ) from error

        if original_source == original_target:
            estimated_time = node_times[original_source]
        elif (
            (original_source, original_target) in tree_edges
            or (original_target, original_source) in tree_edges
        ):
            estimated_time = (
                node_times[original_source] + node_times[original_target]
            ) / 2
        else:
            raise ValueError(
                f"Migration edge {original_source}->{original_target} does not "
                f"map to an input-tree edge in {refined_tree_path}"
            )
        tree_error_values.append(
            min(abs(estimated_time - truth_time) for truth_time in truth_times)
        )

    if not tree_error_values:
        return None
    return sum(tree_error_values) / len(tree_error_values), len(tree_error_values)


def load_mach2_transition_time_errors(
    data_dir: Path,
    tree_indices: set[int] | None = None,
) -> list[Mach2BatchTransitionTimeError]:
    """Score MACH2 migration edges at their temporal midpoint, then average by batch."""
    errors_by_batch: dict[tuple[str, str], list[tuple[float, int]]] = {}
    no_transition_count = 0
    paths = sorted(
        data_dir.glob(
            "setting_*/batch_*/tree_*/mach2_full/mach2_output/*-T-0.refined.tree"
        )
    )
    for path in paths:
        tree_dir = path.parents[2]
        if tree_indices is not None:
            try:
                tree_index = int(tree_dir.name.removeprefix("tree_"))
            except ValueError:
                warnings.warn(f"Skipping MACH2 output with invalid tree name: {path}")
                continue
            if tree_index not in tree_indices:
                continue
        setting = tree_dir.parent.parent.name
        batch = tree_dir.parent.name
        try:
            result = _read_mach2_tree_transition_time_error(path)
        except (OSError, ValueError) as error:
            warnings.warn(f"Skipping MACH2 transition-time output {path}: {error}")
            continue
        if result is None:
            no_transition_count += 1
            continue
        tree_error, transition_count = result
        errors_by_batch.setdefault((setting, batch), []).append(
            (tree_error, transition_count)
        )

    if no_transition_count:
        warnings.warn(
            f"Skipped {no_transition_count} MACH2 trees with no inferred "
            f"transitions in {data_dir}"
        )
    return [
        Mach2BatchTransitionTimeError(
            setting=setting,
            batch=batch,
            value=sum(tree_error for tree_error, _ in tree_errors)
            / len(tree_errors),
            tree_count=len(tree_errors),
            transition_count=sum(count for _, count in tree_errors),
        )
        for (setting, batch), tree_errors in sorted(errors_by_batch.items())
    ]


def _transition_rank_correlation(
    setting: str,
    batch: str,
    method: str,
    probabilities: dict[str, float],
    estimates: dict[str, float],
) -> TransitionRankCorrelation | None:
    transitions = ("B", "C", "D")
    true_values = [probabilities[transition] for transition in transitions]
    inferred_values = [estimates[transition] for transition in transitions]
    comparable_pairs = [
        (left, right)
        for left in range(len(transitions))
        for right in range(left + 1, len(transitions))
        if true_values[left] != true_values[right]
    ]
    if not comparable_pairs:
        return None
    order_error_rate = sum(
        0.5
        if inferred_values[left] == inferred_values[right]
        else float(
            (true_values[left] - true_values[right])
            * (inferred_values[left] - inferred_values[right])
            < 0
        )
        for left, right in comparable_pairs
    ) / len(comparable_pairs)
    correlation = (
        float(spearmanr(true_values, inferred_values).statistic)
        if len(set(inferred_values)) > 1
        else None
    )
    if correlation is not None and not math.isfinite(correlation):
        correlation = None
    return TransitionRankCorrelation(
        setting,
        batch,
        method,
        correlation,
        order_error_rate,
    )


def _transition_pearson_correlation(
    setting: str,
    batch: str,
    method: str,
    true_values_by_transition: dict[str, float],
    inferred_values_by_transition: dict[str, float],
) -> TransitionPearsonCorrelation | None:
    transitions = ("B", "C", "D")
    true_values = [true_values_by_transition[transition] for transition in transitions]
    inferred_values = [
        inferred_values_by_transition[transition] for transition in transitions
    ]
    if (
        not all(math.isfinite(value) for value in true_values + inferred_values)
        or len(set(true_values)) < 2
        or len(set(inferred_values)) < 2
    ):
        return None

    correlation = float(pearsonr(true_values, inferred_values).statistic)
    if not math.isfinite(correlation):
        return None
    return TransitionPearsonCorrelation(setting, batch, method, correlation)


def _sampling_adjusted_star_transition_probabilities(
    inferred: dict[str, float],
) -> dict[str, float]:
    root_sampling = inferred["sampling_A"]
    if not math.isfinite(root_sampling) or root_sampling <= 0:
        raise ValueError(
            "CONCERT root sampling probability must be positive and finite"
        )
    return {
        transition: (
            inferred[f"p_A_to_{transition}"]
            * inferred[f"sampling_{transition}"]
            / root_sampling
        )
        for transition in ("B", "C", "D")
    }


def load_transition_rank_correlations(
    data_dir: Path,
    topology: str,
    num_transition_times: int = 3,
) -> list[TransitionRankCorrelation]:
    if topology not in {"star", "linear"}:
        raise ValueError(f"Unsupported topology for transition ranking: {topology}")

    correlations = []
    rate_observations: list[RateObservation] = []
    undefined_counts: dict[str, int] = {}
    if topology == "star":
        for comparison in load_completed_progressive_comparisons(
            data_dir,
            num_transition_times=num_transition_times,
        ):
            probabilities = {
                transition: comparison.truth[f"p_A_to_{transition}"]
                for transition in ("B", "C", "D")
            }
            estimates = {
                transition: comparison.inferred[f"p_A_to_{transition}"]
                for transition in ("B", "C", "D")
            }
            correlation = _transition_rank_correlation(
                comparison.setting,
                comparison.batch,
                "CONCERT",
                probabilities,
                estimates,
            )
            if correlation is not None:
                correlations.append(correlation)
                if correlation.value is None:
                    undefined_counts["CONCERT"] = (
                        undefined_counts.get("CONCERT", 0) + 1
                    )
            else:
                undefined_counts["CONCERT"] = undefined_counts.get("CONCERT", 0) + 1
        rate_observations = collect_rate_observations(data_dir)
    else:
        truth_by_setting = _read_linear_ground_truth(
            data_dir / "generated_settings.csv"
        )
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
                estimates = _read_linear_joint_estimates(path)
            except (OSError, ValueError) as error:
                warnings.warn(f"Skipping incomplete CONCERT output {path}: {error}")
                continue
            probabilities = {
                target: truth[probability_key]
                for target, _, probability_key, _ in LINEAR_TRANSITIONS
            }
            transition_estimates = {
                target: estimates[probability_key]
                for target, _, probability_key, _ in LINEAR_TRANSITIONS
            }
            correlation = _transition_rank_correlation(
                setting,
                path.parent.name,
                "CONCERT",
                probabilities,
                transition_estimates,
            )
            if correlation is not None:
                correlations.append(correlation)
                if correlation.value is None:
                    undefined_counts["CONCERT"] = (
                        undefined_counts.get("CONCERT", 0) + 1
                    )
            else:
                undefined_counts["CONCERT"] = undefined_counts.get("CONCERT", 0) + 1

        for path in sorted(
            data_dir.glob(
                "setting_*/batch_*/diversitree_musse_first8/joint_musse_fit.txt"
            )
        ):
            setting = path.parent.parent.parent.name
            truth = truth_by_setting.get(setting)
            if truth is None:
                warnings.warn(f"No ground truth for {setting}; skipping {path}")
                continue
            try:
                estimates = _read_linear_diversitree_estimates(path)
            except (OSError, ValueError) as error:
                warnings.warn(f"Skipping incomplete diversitree output {path}: {error}")
                continue
            batch = path.parent.parent.name
            rate_observations.extend(
                RateObservation(
                    method="diversitree",
                    setting=setting,
                    transition=target,
                    probability=truth[probability_key],
                    rate=estimates[f"transition_rate_{edge}"],
                    batch=batch,
                )
                for target, edge, probability_key, _ in LINEAR_TRANSITIONS
            )

        for path in sorted(
            data_dir.glob(
                "setting_*/batch_*/tree_*/beam_snakemake/parameter_summary.tsv"
            )
        ):
            setting = path.parent.parent.parent.parent.name
            truth = truth_by_setting.get(setting)
            if truth is None:
                warnings.warn(f"No ground truth for {setting}; skipping {path}")
                continue
            try:
                rates = _read_beam_rates(path, LINEAR_BEAM_RATE_COLUMNS)
            except (OSError, ValueError) as error:
                warnings.warn(f"Skipping incomplete BEAM output {path}: {error}")
                continue
            batch = path.parent.parent.parent.name
            rate_observations.extend(
                RateObservation(
                    method="BEAM",
                    setting=setting,
                    transition=target,
                    probability=truth[probability_key],
                    rate=rates[target],
                    batch=batch,
                )
                for target, _, probability_key, _ in LINEAR_TRANSITIONS
            )

    grouped_rates: dict[
        tuple[str, str, str], dict[str, list[RateObservation]]
    ] = {}
    for observation in rate_observations:
        if observation.batch is None:
            continue
        key = (observation.method, observation.setting, observation.batch)
        grouped_rates.setdefault(key, {}).setdefault(
            observation.transition, []
        ).append(observation)

    for (method, setting, batch), transitions in sorted(grouped_rates.items()):
        if any(transition not in transitions for transition in ("B", "C", "D")):
            warnings.warn(
                f"Incomplete transition-rate set for {method} {setting}/{batch}",
                stacklevel=2,
            )
            continue
        probabilities = {
            transition: transitions[transition][0].probability
            for transition in ("B", "C", "D")
        }
        mean_rates = {
            transition: sum(
                observation.rate for observation in transitions[transition]
            )
            / len(transitions[transition])
            for transition in ("B", "C", "D")
        }
        correlation = _transition_rank_correlation(
            setting,
            batch,
            method,
            probabilities,
            mean_rates,
        )
        if correlation is not None:
            correlations.append(correlation)
            if correlation.value is None:
                undefined_counts[method] = undefined_counts.get(method, 0) + 1
        else:
            undefined_counts[method] = undefined_counts.get(method, 0) + 1

    for method, count in sorted(undefined_counts.items()):
        warnings.warn(
            f"Spearman correlation is undefined for {count} {method} batches; "
            "they are retained in the order-error-rate plot",
            stacklevel=2,
        )
    return sorted(
        correlations,
        key=lambda item: (item.method, item.setting, item.batch),
    )


def restrict_row_to_common_batches(
    row: CombinedErrorPlotRow,
    mach2_completed_batch_keys: set[tuple[str, str]],
) -> CombinedErrorPlotRow:
    parameter_batches = {
        method: {
            (error.setting, error.batch)
            for error in row.errors
            if error.method == source_method
        }
        for method, source_method in (
            ("CONCERT", "CONCERT"),
            ("diversitree", "MuSSE"),
            ("BEAM", "BEAM"),
        )
    }
    common_batches = (
        parameter_batches["CONCERT"]
        & parameter_batches["diversitree"]
        & parameter_batches["BEAM"]
        & mach2_completed_batch_keys
    )
    if not common_batches:
        raise ValueError(
            f"No batches with completed outputs from all methods for {row.label}"
        )
    return CombinedErrorPlotRow(
        label=row.label,
        errors=[
            error
            for error in row.errors
            if (error.setting, error.batch) in common_batches
        ],
        transition_time_errors=[
            error
            for error in row.transition_time_errors
            if (error.setting, error.batch) in common_batches
        ],
        mach2_transition_time_errors=[
            error
            for error in row.mach2_transition_time_errors
            if (error.setting, error.batch) in common_batches
        ],
        transition_probability_errors=[],
        diversitree_transition_probability_errors=[],
        beam_transition_probability_errors=[],
        transition_rank_correlations=[
            correlation
            for correlation in row.transition_rank_correlations
            if (correlation.setting, correlation.batch) in common_batches
        ],
        transition_correlations=[
            correlation
            for correlation in row.transition_correlations
            if (correlation.setting, correlation.batch) in common_batches
        ],
        common_batch_keys=sorted(common_batches),
    )


def load_topology_comparison_row(
    data_dir: Path,
    label: str,
    topology: str,
) -> CombinedErrorPlotRow:
    if topology == "star":
        errors = load_matched_absolute_errors(data_dir, 3)
        transition_time_errors = load_star_transition_time_errors(data_dir, 3)
        rank_correlations = load_transition_rank_correlations(data_dir, "star", 3)
    elif topology == "linear":
        errors = load_matched_linear_absolute_errors(data_dir)
        transition_time_errors = load_concert_transition_time_errors(data_dir)
        rank_correlations = load_transition_rank_correlations(data_dir, "linear")
    else:
        raise ValueError(f"Unsupported topology: {topology}")

    mach2_completed_batch_keys = set()
    for path in data_dir.glob(
        "setting_*/batch_*/tree_*/mach2_full/mach2_output/*-T-0.refined.tree"
    ):
        try:
            _read_mach2_tree_transition_time_error(path)
        except (OSError, ValueError) as error:
            warnings.warn(f"Skipping incomplete MACH2 output {path}: {error}")
            continue
        mach2_completed_batch_keys.add(
            (path.parents[4].name, path.parents[3].name)
        )

    return restrict_row_to_common_batches(
        CombinedErrorPlotRow(
            label=label,
            errors=errors,
            transition_time_errors=transition_time_errors,
            mach2_transition_time_errors=load_mach2_transition_time_errors(data_dir),
            transition_probability_errors=[],
            diversitree_transition_probability_errors=[],
            beam_transition_probability_errors=[],
            transition_rank_correlations=rank_correlations,
        ),
        mach2_completed_batch_keys,
    )


def load_selected_topology_comparison_row(
    data_dir: Path,
    label: str,
    topology: str,
    selected_batches: tuple[tuple[str, str], ...],
) -> CombinedErrorPlotRow:
    if topology not in {"star", "linear"}:
        raise ValueError(f"Unsupported topology: {topology}")

    selected_keys = {
        (f"setting_{setting}", f"batch_{batch}")
        for setting, batch in selected_batches
    }
    if topology == "star":
        truth_by_setting = {
            setting: _add_identifiable_parameters(values)
            for setting, values in _read_ground_truth(
                data_dir / "generated_settings.csv",
                num_transition_times=3,
            ).items()
        }
        comparisons = {
            (comparison.setting, comparison.batch): comparison
            for comparison in load_completed_progressive_comparisons(
                data_dir,
                num_transition_times=3,
            )
            if (comparison.setting, comparison.batch) in selected_keys
        }
    else:
        truth_by_setting = _read_linear_ground_truth(
            data_dir / "generated_settings.csv"
        )
        comparisons = {}

    errors: list[AbsoluteError] = []
    transition_correlations: list[TransitionPearsonCorrelation] = []
    if topology == "star":
        for key, comparison in comparisons.items():
            truth = truth_by_setting[key[0]]
            coordinated_values = comparison.inferred
            coordinated_rates = _derived_rates(
                tuple(coordinated_values[f"birth_{state}"] for state in STATES),
                tuple(coordinated_values[f"death_{state}"] for state in STATES),
                tuple(
                    coordinated_values[f"sampling_{state}"] for state in STATES
                ),
            )
            truth_rates = {
                "net_growth": tuple(
                    truth[f"net_growth_{state}"] for state in STATES
                ),
                "scaled_birth": tuple(
                    truth[f"birth_{state}"] * truth[f"sampling_{state}"]
                    for state in STATES
                ),
            }
            _append_rate_errors(
                errors,
                key[0],
                key[1],
                "CONCERT",
                coordinated_rates,
                truth_rates,
            )
            correlation = _transition_pearson_correlation(
                key[0],
                key[1],
                "CONCERT",
                {
                    transition: comparison.truth[f"p_A_to_{transition}"]
                    for transition in ("B", "C", "D")
                },
                {
                    **_sampling_adjusted_star_transition_probabilities(
                        comparison.inferred
                    )
                },
            )
            if correlation is not None:
                transition_correlations.append(correlation)
    else:
        for key in selected_keys:
            path = data_dir / key[0] / key[1] / (
                "inference_joint_coordinate_first_8_trees.out"
            )
            truth = truth_by_setting.get(key[0])
            if truth is None or not path.is_file():
                continue
            try:
                estimates = _read_linear_joint_estimates(path)
            except (OSError, ValueError) as error:
                warnings.warn(f"Skipping incomplete CONCERT output {path}: {error}")
                continue
            rates = {
                metric: tuple(estimates[f"{metric}_{state}"] for state in STATES)
                for _, metric in METRICS
            }
            truth_rates = {
                metric: tuple(truth[f"{metric}_{state}"] for state in STATES)
                for _, metric in METRICS
            }
            _append_rate_errors(
                errors, key[0], key[1], "CONCERT", rates, truth_rates
            )
            probabilities = {
                target: truth[probability_key]
                for target, _, probability_key, _ in LINEAR_TRANSITIONS
            }
            transition_estimates = {
                target: estimates[probability_key]
                for target, _, probability_key, _ in LINEAR_TRANSITIONS
            }
            correlation = _transition_pearson_correlation(
                key[0],
                key[1],
                "CONCERT",
                probabilities,
                transition_estimates,
            )
            if correlation is not None:
                transition_correlations.append(correlation)

    beam_rank_rates: dict[
        tuple[str, str], dict[str, list[float]]
    ] = {}
    for key in selected_keys:
        truth = truth_by_setting.get(key[0])
        if truth is None:
            continue

        if topology == "star":
            summary_path = (
                data_dir / key[0] / key[1]
                / "diversitree_musse_first8" / "joint_musse_fit.txt"
            )
            if summary_path.is_file():
                try:
                    estimates, sampling_fractions = _read_diversitree_estimates(
                        summary_path
                    )
                    rates = _derived_rates(
                        tuple(estimates[f"lambda{index}"] for index in range(1, 5)),
                        tuple(estimates[f"mu{index}"] for index in range(1, 5)),
                        sampling_fractions,
                    )
                    truth_rates = {
                        "net_growth": tuple(
                            truth[f"net_growth_{state}"] for state in STATES
                        ),
                        "scaled_birth": tuple(
                            truth[f"birth_{state}"] * truth[f"sampling_{state}"]
                            for state in STATES
                        ),
                    }
                    _append_rate_errors(
                        errors, key[0], key[1], "MuSSE", rates, truth_rates
                    )
                    correlation = _transition_pearson_correlation(
                        key[0],
                        key[1],
                        "diversitree",
                        {
                            transition: truth[f"p_A_to_{transition}"]
                            for transition in ("B", "C", "D")
                        },
                        {
                            transition: estimates[parameter]
                            for transition, parameter in zip(
                                ("B", "C", "D"), ("q12", "q13", "q14")
                            )
                        },
                    )
                    if correlation is not None:
                        transition_correlations.append(correlation)
                except (OSError, ValueError, KeyError) as error:
                    warnings.warn(
                        f"Skipping incomplete diversitree output {summary_path}: {error}"
                    )
        else:
            summary_path = (
                data_dir / key[0] / key[1]
                / "diversitree_musse_first8" / "joint_musse_fit.txt"
            )
            if summary_path.is_file():
                try:
                    estimates = _read_linear_diversitree_estimates(summary_path)
                    rates = {
                        metric: tuple(
                            estimates[f"{metric}_{state}"] for state in STATES
                        )
                        for _, metric in METRICS
                    }
                    truth_rates = {
                        metric: tuple(
                            truth[f"{metric}_{state}"] for state in STATES
                        )
                        for _, metric in METRICS
                    }
                    _append_rate_errors(
                        errors, key[0], key[1], "MuSSE", rates, truth_rates
                    )
                    correlation = _transition_pearson_correlation(
                        key[0],
                        key[1],
                        "diversitree",
                        {
                            target: truth[probability_key]
                            for target, _, probability_key, _ in LINEAR_TRANSITIONS
                        },
                        {
                            target: estimates[f"transition_rate_{edge}"]
                            for target, edge, _, _ in LINEAR_TRANSITIONS
                        },
                    )
                    if correlation is not None:
                        transition_correlations.append(correlation)
                except (OSError, ValueError, KeyError) as error:
                    warnings.warn(
                        f"Skipping incomplete diversitree output {summary_path}: {error}"
                    )

        for tree_index in range(8):
            summary_path = (
                data_dir / key[0] / key[1] / f"tree_{tree_index}"
                / "beam_snakemake" / "parameter_summary.tsv"
            )
            if not summary_path.is_file():
                continue
            status_path = summary_path.with_name("run_status.tsv")
            try:
                with status_path.open(newline="") as source:
                    status_rows = list(csv.DictReader(source, delimiter="\t"))
                if len(status_rows) != 1 or status_rows[0].get("status") != "completed":
                    raise ValueError(f"BEAM run is not marked completed in {status_path}")
                net_growth, birth_rate = _read_beam_rate_estimates(summary_path)
            except (OSError, ValueError) as error:
                warnings.warn(f"Skipping incomplete BEAM output {summary_path}: {error}")
            else:
                if topology == "star":
                    sampling_rate = sum(
                        truth[f"sampling_{state}"] for state in STATES
                    ) / len(STATES)
                    mean_truth = {
                        "net_growth": sum(
                            truth[f"net_growth_{state}"] for state in STATES
                        ) / len(STATES),
                        "scaled_birth": sum(
                            truth[f"birth_{state}"] * truth[f"sampling_{state}"]
                            for state in STATES
                        ) / len(STATES),
                    }
                else:
                    sampling_rate = truth["sampling_rate"]
                    mean_truth = {
                        "net_growth": sum(
                            truth[f"net_growth_{state}"] for state in STATES
                        ) / len(STATES),
                        "scaled_birth": sum(
                            truth[f"birth_{state}"] * sampling_rate
                            for state in STATES
                        ) / len(STATES),
                    }
                for _, metric in METRICS:
                    estimate = (
                        net_growth if metric == "net_growth"
                        else birth_rate * sampling_rate
                    )
                    errors.append(
                        AbsoluteError(
                            key[0],
                            key[1],
                            MEAN_STATE,
                            "BEAM",
                            metric,
                            abs(estimate - mean_truth[metric]),
                            tree=f"tree_{tree_index}",
                        )
                    )

            try:
                if topology == "star":
                    rates = _read_beam_rates(summary_path)
                else:
                    rates = _read_beam_rates(
                        summary_path,
                        LINEAR_BEAM_RATE_COLUMNS,
                    )
            except (OSError, ValueError) as error:
                warnings.warn(
                    f"Skipping incomplete BEAM transition rates {summary_path}: {error}"
                )
                continue
            targets = ("B", "C", "D")
            beam_rank_rates.setdefault(
                key, {target: [] for target in targets}
            )
            for target in targets:
                beam_rank_rates[key][target].append(rates[target])

    for key, transition_rates in beam_rank_rates.items():
        truth = truth_by_setting[key[0]]
        if topology == "star":
            probabilities = {
                target: truth[f"p_A_to_{target}"] for target in ("B", "C", "D")
            }
        else:
            probabilities = {
                target: truth[probability_key]
                for target, _, probability_key, _ in LINEAR_TRANSITIONS
            }
        mean_rates = {
            target: sum(values) / len(values)
            for target, values in transition_rates.items()
        }
        correlation = _transition_pearson_correlation(
            key[0], key[1], "BEAM", probabilities, mean_rates
        )
        if correlation is not None:
            transition_correlations.append(correlation)

    if topology == "star":
        transition_time_errors = [
            error
            for error in load_star_transition_time_errors(data_dir, 3)
            if (error.setting, error.batch) in selected_keys
        ]
    else:
        transition_time_errors = [
            error
            for error in load_concert_transition_time_errors(data_dir)
            if (error.setting, error.batch) in selected_keys
        ]

    mach2_transition_time_errors = [
        error
        for error in load_mach2_transition_time_errors(
            data_dir,
            tree_indices=set(range(8)),
        )
        if (error.setting, error.batch) in selected_keys
    ]
    mach2_completed_batch_keys: set[tuple[str, str]] = set()
    for path in data_dir.glob(
        "setting_*/batch_*/tree_*/mach2_full/mach2_output/*-T-0.refined.tree"
    ):
        setting = path.parents[4].name
        batch = path.parents[3].name
        if (setting, batch) not in selected_keys:
            continue
        tree_dir = path.parents[2]
        try:
            tree_index = int(tree_dir.name.removeprefix("tree_"))
        except ValueError:
            warnings.warn(f"Skipping MACH2 output with invalid tree name: {path}")
            continue
        if tree_index not in range(8):
            continue
        try:
            _read_mach2_tree_transition_time_error(path)
        except (OSError, ValueError) as error:
            warnings.warn(f"Skipping incomplete MACH2 output {path}: {error}")
            continue
        mach2_completed_batch_keys.add((setting, batch))

    row = CombinedErrorPlotRow(
        label=label,
        errors=errors,
        transition_time_errors=transition_time_errors,
        mach2_transition_time_errors=mach2_transition_time_errors,
        transition_probability_errors=[],
        diversitree_transition_probability_errors=[],
        beam_transition_probability_errors=[],
        transition_correlations=transition_correlations,
        common_batch_keys=sorted(selected_keys),
    )
    return restrict_row_to_common_batches(row, mach2_completed_batch_keys)


def load_star_transition_probability_errors(
    data_dir: Path,
    num_transition_times: int,
) -> list[float]:
    comparisons = load_completed_progressive_comparisons(
        data_dir,
        num_transition_times=num_transition_times,
    )
    errors = [
        abs(
            comparison.inferred[f"p_A_to_{state}"]
            - comparison.truth[f"p_A_to_{state}"]
        )
        for comparison in comparisons
        for state in STATES
    ]
    if not errors:
        raise ValueError(f"No CONCERT transition-probability estimates found in {data_dir}")
    return errors


def load_star_transition_probability_comparison_errors(
    data_dir: Path,
    num_transition_times: int,
) -> tuple[list[float], list[float], list[float], dict[str, float], int]:
    comparisons = load_completed_progressive_comparisons(
        data_dir,
        num_transition_times=num_transition_times,
    )
    rate_observations = collect_rate_observations(data_dir)
    calibrated_rates = calibrate_rates_to_probabilities(
        rate_observations,
        "diversitree",
    )
    beam_rate_observations = [
        observation
        for observation in rate_observations
        if observation.method == "BEAM"
    ]
    calibrated_beam_rates = (
        calibrate_rates_to_probabilities(beam_rate_observations, "BEAM")
        if len({observation.setting for observation in beam_rate_observations}) > 1
        else []
    )
    calibrated_keys = {
        (observation.setting, observation.batch)
        for observation in calibrated_rates
        if observation.batch is not None
    }
    comparisons_by_key = {
        (comparison.setting, comparison.batch): comparison
        for comparison in comparisons
    }
    matched_keys = calibrated_keys.intersection(comparisons_by_key)
    if not matched_keys:
        raise ValueError(
            f"No matched CONCERT/diversitree probability fits in {data_dir}"
        )

    concert_errors = [
        abs(
            comparison.inferred[f"p_A_to_{state}"]
            - comparison.truth[f"p_A_to_{state}"]
        )
        for key in sorted(matched_keys)
        for comparison in (comparisons_by_key[key],)
        for state in STATES[1:]
    ]
    diversitree_errors = [
        abs(observation.estimate - observation.probability)
        for observation in calibrated_rates
        if (observation.setting, observation.batch) in matched_keys
    ]
    beam_errors = [
        abs(observation.estimate - observation.probability)
        for observation in calibrated_beam_rates
        if (observation.setting, observation.batch) in matched_keys
    ]
    if len(concert_errors) != len(diversitree_errors):
        raise ValueError("Matched probability error counts differ between methods")

    slopes = {}
    for transition in ("B", "C", "D"):
        transition_slopes = [
            observation.slope
            for observation in calibrated_rates
            if observation.transition == transition
            and (observation.setting, observation.batch) in matched_keys
        ]
        if not transition_slopes:
            raise ValueError(f"No matched diversitree rates for transition A->{transition}")
        slopes[transition] = sum(transition_slopes) / len(transition_slopes)
    clipped_count = sum(
        observation.clipped
        for observation in calibrated_rates
        if (observation.setting, observation.batch) in matched_keys
    )
    return concert_errors, diversitree_errors, beam_errors, slopes, clipped_count


def load_linear_transition_probability_comparison_errors(
    data_dir: Path,
) -> tuple[list[float], list[float], list[float], dict[str, float], int]:
    truth_by_setting = _read_linear_ground_truth(data_dir / "generated_settings.csv")
    concert_by_batch = {}
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
            estimates = _read_linear_joint_estimates(path)
        except (OSError, ValueError) as error:
            warnings.warn(f"Skipping incomplete CONCERT output {path}: {error}")
            continue
        concert_by_batch[(setting, path.parent.name)] = estimates

    rate_observations = []
    for path in sorted(
        data_dir.glob(
            "setting_*/batch_*/diversitree_musse_first8/joint_musse_fit.txt"
        )
    ):
        setting = path.parent.parent.parent.name
        truth = truth_by_setting.get(setting)
        if truth is None:
            warnings.warn(f"No ground truth for {setting}; skipping {path}")
            continue
        try:
            estimates = _read_linear_diversitree_estimates(path)
        except (OSError, ValueError) as error:
            warnings.warn(f"Skipping incomplete diversitree output {path}: {error}")
            continue
        batch = path.parent.parent.name
        rate_observations.extend(
            RateObservation(
                method="diversitree",
                setting=setting,
                transition=target,
                probability=truth[probability_key],
                rate=estimates[f"transition_rate_{edge}"],
                batch=batch,
            )
            for target, edge, probability_key, _ in LINEAR_TRANSITIONS
        )

    beam_rate_observations = []
    for path in sorted(
        data_dir.glob(
            "setting_*/batch_*/tree_*/beam_snakemake/parameter_summary.tsv"
        )
    ):
        setting = path.parent.parent.parent.parent.name
        truth = truth_by_setting.get(setting)
        if truth is None:
            warnings.warn(f"No ground truth for {setting}; skipping {path}")
            continue
        try:
            rates = _read_beam_rates(path, LINEAR_BEAM_RATE_COLUMNS)
        except (OSError, ValueError) as error:
            warnings.warn(f"Skipping incomplete BEAM output {path}: {error}")
            continue
        batch = path.parent.parent.parent.name
        beam_rate_observations.extend(
            RateObservation(
                method="BEAM",
                setting=setting,
                transition=target,
                probability=truth[probability_key],
                rate=rates[target],
                batch=batch,
            )
            for target, _, probability_key, _ in LINEAR_TRANSITIONS
        )

    calibrated_rates = calibrate_rates_to_probabilities(
        rate_observations,
        "diversitree",
    )
    calibrated_beam_rates = (
        calibrate_rates_to_probabilities(beam_rate_observations, "BEAM")
        if len({observation.setting for observation in beam_rate_observations}) > 1
        else []
    )
    calibrated_keys = {
        (observation.setting, observation.batch)
        for observation in calibrated_rates
        if observation.batch is not None
    }
    matched_keys = calibrated_keys.intersection(concert_by_batch)
    if not matched_keys:
        raise ValueError(
            f"No matched CONCERT/diversitree probability fits in {data_dir}"
        )

    concert_errors = [
        abs(
            concert_by_batch[key][probability_key]
            - truth_by_setting[key[0]][probability_key]
        )
        for key in sorted(matched_keys)
        for _, _, probability_key, _ in LINEAR_TRANSITIONS
    ]
    diversitree_errors = [
        abs(observation.estimate - observation.probability)
        for observation in calibrated_rates
        if (observation.setting, observation.batch) in matched_keys
    ]
    beam_errors = [
        abs(observation.estimate - observation.probability)
        for observation in calibrated_beam_rates
        if (observation.setting, observation.batch) in matched_keys
    ]
    if len(concert_errors) != len(diversitree_errors):
        raise ValueError("Matched probability error counts differ between methods")

    slopes = {}
    for target, _, _, _ in LINEAR_TRANSITIONS:
        transition_slopes = [
            observation.slope
            for observation in calibrated_rates
            if observation.transition == target
            and (observation.setting, observation.batch) in matched_keys
        ]
        if not transition_slopes:
            raise ValueError(f"No matched diversitree rates for linear edge ending {target}")
        slopes[target] = sum(transition_slopes) / len(transition_slopes)
    clipped_count = sum(
        observation.clipped
        for observation in calibrated_rates
        if (observation.setting, observation.batch) in matched_keys
    )
    return concert_errors, diversitree_errors, beam_errors, slopes, clipped_count


def _plot_error_distribution_panel(
    axis: plt.Axes,
    groups: list[tuple[float, str, list[float], str]],
    *,
    title: str,
    panel: str,
    x_limits: tuple[float, float],
    font_family: str,
) -> None:
    upper_whiskers = []
    for position, _, values, color in groups:
        box = axis.boxplot(
            [values],
            positions=[position],
            widths=BOX_WIDTH,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "#222222", "linewidth": MEDIAN_LINE_WIDTH},
            whiskerprops={"color": "#333333", "linewidth": WHISKER_LINE_WIDTH},
            capprops={"color": BOX_EDGE_COLOR, "linewidth": 1.0},
            flierprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": color,
                "markersize": 1.8,
                "markeredgewidth": 0.7,
                "alpha": 0.75,
            },
        )
        box["boxes"][0].set_facecolor(color)
        box["boxes"][0].set_alpha(BOX_FILL_ALPHA)
        box["boxes"][0].set_edgecolor(BOX_EDGE_COLOR)
        box["boxes"][0].set_linewidth(BOX_EDGE_WIDTH)
        upper_whiskers.append(boxplot_stats([values])[0]["whishi"])
    axis.set_xticks([])
    axis.set_xlim(*x_limits)
    axis.set_ylabel("Absolute error", fontsize=9, labelpad=4)
    upper_limit = max(upper_whiskers, default=0.0)
    axis.set_ylim(0, upper_limit * 1.06 if upper_limit > 0 else 1.0)
    axis.set_title(title, fontsize=9, fontweight="normal", pad=6)
    axis.text(
        -0.16,
        1.12,
        f"({panel})",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        clip_on=False,
    )
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.tick_params(
        axis="both",
        labelsize=8,
        length=2.5,
        width=0.65,
        direction="out",
        pad=2,
    )
    for text in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        text.set_fontfamily(font_family)
    axis.yaxis.label.set_fontfamily(font_family)
    axis.title.set_fontfamily(font_family)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_linewidth(0.7)


def plot_error_boxplots(
    errors: list[AbsoluteError],
    output_path: Path,
    transition_time_errors: list[TransitionTimeError] | None = None,
    transition_probability_errors: list[float] | None = None,
    diversitree_transition_probability_errors: list[float] | None = None,
    beam_transition_probability_errors: list[float] | None = None,
    transition_probability_label: str = "A→A–D pooled",
    mach2_transition_time_errors: list[Mach2BatchTransitionTimeError] | None = None,
) -> Path:
    from matplotlib.patches import Patch

    sample_counts = {}
    for method in METHODS:
        sample_counts[method] = len(
            {
                (error.setting, error.batch, error.tree)
                for error in errors
                if error.method == method
            }
        )
    plot_methods = tuple(method for method in METHODS if sample_counts[method] > 0)
    if not plot_methods:
        raise ValueError("No inference-method errors available to plot")
    if transition_time_errors is not None and not transition_time_errors:
        raise ValueError("No transition-time errors available to plot")
    if mach2_transition_time_errors is not None and not mach2_transition_time_errors:
        raise ValueError("No MACH2 transition-time errors available to plot")
    if transition_probability_errors is not None and not transition_probability_errors:
        raise ValueError("No transition-probability errors available to plot")
    if diversitree_transition_probability_errors is not None:
        if not diversitree_transition_probability_errors:
            raise ValueError("No diversitree probability errors available to plot")
        if transition_probability_errors is None:
            raise ValueError(
                "CONCERT errors are required to compare diversitree probabilities"
            )
    if beam_transition_probability_errors is not None:
        if not beam_transition_probability_errors:
            raise ValueError("No BEAM probability errors available to plot")
        if transition_probability_errors is None:
            raise ValueError("CONCERT errors are required to compare BEAM probabilities")
    has_time_panel = (
        transition_time_errors is not None
        or mach2_transition_time_errors is not None
    )
    panel_count = (
        2
        + has_time_panel
        + (transition_probability_errors is not None)
    )
    figure_width = 3.9 + 0.9 * (panel_count - 2)
    font_family = "Nimbus Sans"
    figure, axes = plt.subplots(
        1,
        panel_count,
        figsize=(figure_width, 2.9),
        sharey=False,
    )
    figure.patch.set_facecolor("white")
    method_positions = [1 + 0.5 * index for index in range(len(plot_methods))]
    x_limits = (method_positions[0] - 0.25, method_positions[-1] + 0.25)
    if mach2_transition_time_errors is not None:
        x_limits = (x_limits[0], max(x_limits[1], 2.25))

    for axis, (panel, (label, metric)) in zip(
        axes[:2],
        zip(("a", "b"), METRICS),
    ):
        upper_whiskers = []
        for method_index, method in enumerate(plot_methods):
            position = method_positions[method_index]
            states = STATES if method != "BEAM" else (MEAN_STATE,)
            values = [
                error.value
                for error in errors
                if error.metric == metric
                and error.state in states
                and error.method == method
            ]
            if not values:
                continue
            box = axis.boxplot(
                [values],
                positions=[position],
                widths=BOX_WIDTH,
                patch_artist=True,
                showfliers=True,
                medianprops={"color": "#222222", "linewidth": MEDIAN_LINE_WIDTH},
                whiskerprops={
                    "color": "#333333",
                    "linewidth": WHISKER_LINE_WIDTH,
                },
                capprops={
                    "color": BOX_EDGE_COLOR,
                    "linewidth": 1.0,
                },
                flierprops={
                    "marker": "o",
                    "markerfacecolor": "white",
                    "markeredgecolor": METHOD_COLORS[method],
                    "markersize": 1.8,
                    "markeredgewidth": 0.7,
                    "alpha": 0.75,
                },
            )
            box["boxes"][0].set_facecolor(METHOD_COLORS[method])
            box["boxes"][0].set_alpha(BOX_FILL_ALPHA)
            box["boxes"][0].set_edgecolor(BOX_EDGE_COLOR)
            box["boxes"][0].set_linewidth(BOX_EDGE_WIDTH)
            upper_whiskers.append(boxplot_stats([values])[0]["whishi"])

        axis.set_xticks([])
        axis.set_xlim(*x_limits)
        axis.set_yscale("linear")
        upper_limit = max(upper_whiskers, default=0.0)
        axis.set_ylim(0, upper_limit * 1.06 if upper_limit > 0 else 1.0)
        axis.set_ylabel("Absolute error", fontsize=9, labelpad=4)
        axis.set_title(label, fontsize=9, fontweight="normal", pad=6)
        axis.text(
            -0.16,
            1.12,
            f"({panel})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            clip_on=False,
        )
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.55)
        axis.set_axisbelow(True)
        axis.tick_params(
            axis="both",
            labelsize=8,
            length=2.5,
            width=0.65,
            direction="out",
            pad=2,
        )
        for text in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            text.set_fontfamily(font_family)
        axis.xaxis.label.set_fontfamily(font_family)
        axis.yaxis.label.set_fontfamily(font_family)
        axis.title.set_fontfamily(font_family)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_linewidth(0.7)

    aggregate_axis_index = 2
    if has_time_panel:
        time_groups = []
        if transition_time_errors is not None:
            time_groups.append(
                (
                    sum(method_positions) / len(method_positions),
                    "All transitions",
                    [error.value for error in transition_time_errors],
                    METHOD_COLORS["CONCERT"],
                )
            )
        if mach2_transition_time_errors is not None:
            time_groups.append(
                (
                    max(2.0, method_positions[-1]),
                    "MACH2 (edge midpoint)",
                    [error.value for error in mach2_transition_time_errors],
                    METHOD_COLORS["MACH2"],
                )
            )
        _plot_error_distribution_panel(
            axes[aggregate_axis_index],
            time_groups,
            title="Transition time",
            panel="c",
            x_limits=x_limits,
            font_family=font_family,
        )
        aggregate_axis_index += 1
    if transition_probability_errors is not None:
        if (
            diversitree_transition_probability_errors is None
            and beam_transition_probability_errors is None
        ):
            probability_x_limits = x_limits
            probability_groups = [
                (
                    sum(method_positions) / len(method_positions),
                    transition_probability_label,
                    transition_probability_errors,
                    METHOD_COLORS["CONCERT"],
                )
            ]
        else:
            probability_x_limits = x_limits
            probability_groups = [
                (
                    method_positions[plot_methods.index("CONCERT")],
                    "CONCERT",
                    transition_probability_errors,
                    METHOD_COLORS["CONCERT"],
                )
            ]
            if diversitree_transition_probability_errors is not None:
                probability_groups.append(
                    (
                    method_positions[plot_methods.index("MuSSE")],
                    "diversitree",
                    diversitree_transition_probability_errors,
                    METHOD_COLORS["MuSSE"],
                    )
                )
            if beam_transition_probability_errors is not None:
                probability_groups.append(
                    (
                        method_positions[plot_methods.index("BEAM")],
                        "BEAM",
                        beam_transition_probability_errors,
                        METHOD_COLORS["BEAM"],
                    )
                )
        _plot_error_distribution_panel(
            axes[aggregate_axis_index],
            probability_groups,
            title="Transition probability",
            panel="d" if has_time_panel else "c",
            x_limits=probability_x_limits,
            font_family=font_family,
        )

    if has_time_panel:
        shared_upper_limit = max(axis.get_ylim()[1] for axis in axes[:2])
        for axis in axes[:2]:
            axis.set_ylim(0, shared_upper_limit)
        axes[2].set_ylim(*TRANSITION_TIME_ERROR_YLIM)

    legend_handles = [
        Patch(
            facecolor=METHOD_COLORS[method],
            edgecolor=BOX_EDGE_COLOR,
            linewidth=BOX_EDGE_WIDTH,
            alpha=BOX_FILL_ALPHA,
            label=METHOD_DISPLAY_NAMES[method],
        )
        for method in (
            *plot_methods,
            *(("MACH2",) if mach2_transition_time_errors is not None else ()),
        )
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=8.5,
        columnspacing=1.2,
        handlelength=1.3,
        borderpad=0.35,
        facecolor="white",
        edgecolor="#555555",
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.995,
        bottom=0.2,
        top=0.86,
        wspace=0.45,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    pdf_path = output_path.with_suffix(".pdf")
    if pdf_path != output_path:
        figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)
    return output_path


def _topology_error_values(
    row: CombinedErrorPlotRow,
    metric: str,
    method: str,
) -> list[float]:
    if metric in {"net_growth", "scaled_birth"}:
        states = (MEAN_STATE,) if method == "BEAM" else STATES
        source_method = "MuSSE" if method == "diversitree" else method
        return [
            error.value
            for error in row.errors
            if error.metric == metric
            and error.method == source_method
            and error.state in states
        ]
    if metric == "transition_time":
        if method == "CONCERT":
            return [error.value for error in row.transition_time_errors]
        if method == "MACH2":
            return [error.value for error in row.mach2_transition_time_errors]
    if metric == "transition_rank_correlation":
        return [
            correlation.value
            for correlation in row.transition_rank_correlations
            if correlation.method == method and correlation.value is not None
        ]
    if metric == "transition_correlation":
        return [
            correlation.value
            for correlation in row.transition_correlations
            if correlation.method == method
        ]
    if metric == "transition_order_error":
        return [
            correlation.order_error_rate
            for correlation in row.transition_rank_correlations
            if correlation.method == method
        ]
    raise ValueError(f"Unsupported topology-comparison metric/method: {metric}/{method}")


def _draw_topology(ax: plt.Axes, topology: str) -> None:
    from matplotlib.patches import Ellipse, FancyArrowPatch

    line_color = "#303030"
    if topology == "star":
        nodes = {
            "A": (0.12, 0.5),
            "B": (0.78, 0.8),
            "C": (0.78, 0.5),
            "D": (0.78, 0.2),
        }
        edges = (("A", "B"), ("A", "C"), ("A", "D"))
        non_leaves = ("A",)
    else:
        nodes = {
            "A": (0.08, 0.5),
            "B": (0.36, 0.5),
            "C": (0.64, 0.5),
            "D": (0.92, 0.5),
        }
        edges = (("A", "B"), ("B", "C"), ("C", "D"))
        non_leaves = ("A", "B", "C")

    axis_position = ax.get_position()
    aspect = (
        axis_position.width
        * ax.figure.get_figwidth()
        / (axis_position.height * ax.figure.get_figheight())
    )
    radius_y = 0.075
    radius_x = radius_y / aspect
    for source, target in edges:
        start = nodes[source]
        end = nodes[target]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.hypot(dx * aspect, dy)
        direction = (dx * aspect / distance, dy / distance)
        start = (
            start[0] + direction[0] * radius_y / aspect,
            start[1] + direction[1] * radius_y,
        )
        end = (
            end[0] - direction[0] * radius_y / aspect,
            end[1] - direction[1] * radius_y,
        )
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=7,
                linewidth=0.8,
                color=line_color,
                zorder=1,
            )
        )
    for label, (x, y) in nodes.items():
        ax.add_patch(
            Ellipse(
                (x, y),
                2 * radius_x,
                2 * radius_y,
                transform=ax.transAxes,
                facecolor="#F1F1F1",
                edgecolor=line_color,
                linewidth=0.8,
                zorder=2,
            )
        )
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12.5,
            zorder=3,
        )
    for label in non_leaves:
        x, y = nodes[label]
        if topology == "star":
            start = (x - radius_x * 1.05, y - radius_y * 0.5)
            end = (x - radius_x * 1.05, y + radius_y * 0.5)
            loop_radius = -1.6
        else:
            start = (x - radius_x * 0.9, y - radius_y * 1.1)
            end = (x + radius_x * 0.9, y - radius_y * 1.1)
            loop_radius = 1.6
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                transform=ax.transAxes,
                connectionstyle=f"arc3,rad={loop_radius}",
                arrowstyle="-|>",
                mutation_scale=6,
                linewidth=0.75,
                color=line_color,
                zorder=4,
            )
        )
    ax.set_axis_off()


def plot_topology_comparison(
    rows: tuple[CombinedErrorPlotRow, CombinedErrorPlotRow],
    output_path: Path,
) -> Path:
    from matplotlib import font_manager
    from matplotlib.patches import Patch
    from matplotlib.text import Text

    line_color = "#303030"
    for candidate in ("Arial", "Nimbus Sans", "DejaVu Sans"):
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=candidate),
                fallback_to_default=False,
            )
        except ValueError:
            continue
        font_family = candidate
        break
    else:
        raise RuntimeError("No supported font is available for the topology figure")

    colors = {
        "CONCERT": "#4E9FCF",
        "diversitree": "#EDBB4F",
        "BEAM": "#4FBB9B",
        "MACH2": "#E5914F",
    }
    metrics = (
        ("net_growth", "Net\ngrowth", (0.0, 6.2), (0.0, 2.0, 4.0, 6.0),
         ("CONCERT", "diversitree", "BEAM")),
        ("scaled_birth", "Scaled\nbirth", (0.0, 6.2), (0.0, 2.0, 4.0, 6.0),
         ("CONCERT", "diversitree", "BEAM")),
        ("transition_time", "Transition\ntime", TRANSITION_TIME_ERROR_YLIM,
         (0.0, 0.05, 0.10, 0.15, 0.20), ("CONCERT", "MACH2")),
        ("transition_correlation", "Transition\ncorrelation",
         (-1.05, 1.05), (-1.0, -0.5, 0.0, 0.5, 1.0),
         ("CONCERT", "diversitree", "BEAM")),
    )
    topology_groups = (
        (rows[1], "linear", "Linear topology"),
        (rows[0], "star", "Star topology"),
    )
    figure = plt.figure(figsize=(19.0, 5.2))
    outer = figure.add_gridspec(1, 2, wspace=0.45)
    legend_colors = {
        "CONCERT": colors["CONCERT"],
        "diversitree": colors["diversitree"],
        "BEAM": colors["BEAM"],
        "MACH2": colors["MACH2"],
    }
    for topology_index, (row, topology, title) in enumerate(topology_groups):
        group_grid = outer[0, topology_index].subgridspec(
            2,
            4,
            height_ratios=(0.7, 2.0),
            width_ratios=(1.0, 1.0, 0.95, 1.15),
            wspace=1.1,
            hspace=0.32,
        )
        topology_ax = figure.add_subplot(group_grid[0, :])
        topology_ax.set_title(
            title,
            fontsize=18,
            fontweight="bold",
            pad=6,
        )
        topology_ax.text(
            0.0,
            1.14,
            ("(a)", "(b)")[topology_index],
            transform=topology_ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
            ha="left",
        )
        _draw_topology(topology_ax, topology)
        for metric_index, (
            metric,
            title,
            y_limits,
            y_ticks,
            methods,
        ) in enumerate(metrics):
            axis = figure.add_subplot(group_grid[1, metric_index])
            plotted_values = []
            for method_index, method in enumerate(methods):
                values = _topology_error_values(row, metric, method)
                if not values:
                    continue
                plotted_values.extend(values)
                boxplot = axis.boxplot(
                    [values],
                    positions=[method_index],
                    widths=0.88,
                    patch_artist=True,
                    showfliers=True,
                    medianprops={"color": "#111111", "linewidth": 1.1},
                    whiskerprops={"color": line_color, "linewidth": 0.8},
                    capprops={"color": line_color, "linewidth": 0.8},
                    flierprops={
                        "marker": "o",
                        "markersize": 1.5,
                        "markeredgewidth": 0.4,
                        "markerfacecolor": "none",
                    },
                )
                boxplot["boxes"][0].set_facecolor(colors[method])
                boxplot["boxes"][0].set_edgecolor(line_color)
                boxplot["boxes"][0].set_linewidth(0.8)
                for flier in boxplot["fliers"]:
                    flier.set_markeredgecolor(colors[method])

            axis.set_ylim(*y_limits)
            axis.set_yticks(y_ticks)
            axis.set_xlim(-0.5, len(methods) - 0.5)
            axis.set_xticks([])
            axis.tick_params(axis="x", length=0)
            axis.set_title(title, fontsize=16, pad=6)
            if not plotted_values:
                axis.text(
                    0.5,
                    0.5,
                    "No completed outputs",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                )
            axis.tick_params(axis="y", labelsize=12, length=3.5, pad=3)
            axis.grid(axis="y", linewidth=0.7, color="#B8B8B8")
            axis.set_axisbelow(True)
            ylabel = (
                "Pearson's r"
                if metric == "transition_correlation"
                else "Absolute error"
            )
            axis.set_ylabel(ylabel, fontsize=14, labelpad=6)
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                axis.spines[spine].set_color(line_color)
                axis.spines[spine].set_linewidth(0.8)

    handles = [
        Patch(
            facecolor=legend_colors[method],
            edgecolor=line_color,
            linewidth=0.7,
            label=(
                "MACH2"
                if method == "MACH2"
                else method
            ),
        )
        for method in ("CONCERT", "diversitree", "BEAM", "MACH2")
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        frameon=False,
        fontsize=14,
        columnspacing=1.3,
        handlelength=1.25,
    )
    figure.subplots_adjust(
        left=0.06,
        right=0.99,
        top=0.91,
        bottom=0.16,
    )
    for text in figure.findobj(match=Text):
        text.set_fontfamily(font_family)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    pdf_path = output_path.with_suffix(".pdf")
    if pdf_path != output_path:
        figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)
    return output_path


def plot_combined_error_boxplots(
    rows: tuple[CombinedErrorPlotRow, CombinedErrorPlotRow],
    output_path: Path,
) -> Path:
    from matplotlib.patches import Patch

    for row in rows:
        if not row.errors:
            raise ValueError(f"No parameter errors available for {row.label}")
        if not row.transition_time_errors:
            raise ValueError(f"No transition-time errors available for {row.label}")
        if not row.transition_probability_errors:
            raise ValueError(
                f"No transition-probability errors available for {row.label}"
            )
        if not row.diversitree_transition_probability_errors:
            raise ValueError(
                f"No diversitree probability errors available for {row.label}"
            )

    plot_methods = tuple(
        method
        for method in METHODS
        if any(error.method == method for row in rows for error in row.errors)
    )
    method_positions = {
        method: 1 + 0.5 * index for index, method in enumerate(METHODS)
    }
    x_limits = (0.75, 2.25)
    probability_x_limits = x_limits
    font_family = "Nimbus Sans"
    figure, axes = plt.subplots(
        1,
        8,
        figsize=(8.4, 3.0),
        sharey=False,
    )
    figure.patch.set_facecolor("white")

    for row_index, row in enumerate(rows):
        for column, (title, metric) in enumerate(
            (*METRICS, ("Transition time", ""), ("Transition probability", ""))
        ):
            axis = axes[row_index * 4 + column]
            if column < len(METRICS):
                groups = []
                for method in plot_methods:
                    states = STATES if method != "BEAM" else (MEAN_STATE,)
                    values = [
                        error.value
                        for error in row.errors
                        if error.metric == metric
                        and error.state in states
                        and error.method == method
                    ]
                    if values:
                        groups.append(
                            (
                                method_positions[method],
                                METHOD_DISPLAY_NAMES[method],
                                values,
                                METHOD_COLORS[method],
                            )
                        )
            elif column == 2:
                groups = [
                    (
                        method_positions["CONCERT"],
                        "All transitions",
                        [error.value for error in row.transition_time_errors],
                        METHOD_COLORS["CONCERT"],
                    )
                ]
                if row.mach2_transition_time_errors:
                    groups.append(
                        (
                            method_positions["BEAM"],
                            "MACH2 (edge midpoint)",
                            [
                                error.value
                                for error in row.mach2_transition_time_errors
                            ],
                            METHOD_COLORS["MACH2"],
                        )
                    )
            else:
                groups = [
                    (
                        method_positions["CONCERT"],
                        "CONCERT",
                        row.transition_probability_errors,
                        METHOD_COLORS["CONCERT"],
                    ),
                    (
                        method_positions["MuSSE"],
                        "diversitree",
                        row.diversitree_transition_probability_errors,
                        METHOD_COLORS["MuSSE"],
                    ),
                ]
                if row.beam_transition_probability_errors:
                    groups.append(
                        (
                            method_positions["BEAM"],
                            "BEAM",
                            row.beam_transition_probability_errors,
                            METHOD_COLORS["BEAM"],
                        )
                    )

            if not groups:
                raise ValueError(f"No {title.lower()} errors available for {row.label}")
            _plot_error_distribution_panel(
                axis,
                groups,
                title=title,
                panel="abcdefgh"[row_index * 4 + column],
                x_limits=probability_x_limits if column == 3 else x_limits,
                font_family=font_family,
            )
            if column != 0:
                axis.set_ylabel("")
            if column == 1:
                axis.tick_params(axis="y", labelleft=False)
            axis.set_title(title, fontsize=7.5, fontweight="normal", pad=5)

    parameter_axes = (axes[0], axes[1], axes[4], axes[5])
    shared_upper_limit = max(axis.get_ylim()[1] for axis in parameter_axes)
    for axis in parameter_axes:
        axis.set_ylim(0, shared_upper_limit)
    for axis in (axes[2], axes[6]):
        axis.set_ylim(*TRANSITION_TIME_ERROR_YLIM)
    probability_axes = (axes[3], axes[7])
    probability_upper_limit = max(axis.get_ylim()[1] for axis in probability_axes)
    for axis in probability_axes:
        axis.set_ylim(0, probability_upper_limit)

    figure.subplots_adjust(
        left=0.12,
        right=0.995,
        bottom=0.25,
        top=0.84,
        wspace=0.18,
    )
    for row_index, row in enumerate(rows):
        group_axes = axes[row_index * 4 : row_index * 4 + 4]
        left = group_axes[0].get_position().x0
        right = group_axes[-1].get_position().x1
        figure.text(
            (left + right) / 2,
            0.97,
            row.label,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            fontfamily=font_family,
        )

    legend_methods = list(plot_methods)
    if any(row.mach2_transition_time_errors for row in rows):
        legend_methods.append("MACH2")
    legend_handles = [
        Patch(
            facecolor=METHOD_COLORS[method],
            edgecolor=BOX_EDGE_COLOR,
            linewidth=BOX_EDGE_WIDTH,
            alpha=BOX_FILL_ALPHA,
            label=METHOD_DISPLAY_NAMES[method],
        )
        for method in legend_methods
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=8.5,
        columnspacing=1.2,
        handlelength=1.3,
        borderpad=0.35,
        facecolor="white",
        edgecolor="#555555",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    pdf_path = output_path.with_suffix(".pdf")
    if pdf_path != output_path:
        figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot net-growth and scaled-birth absolute errors for matched "
            "star or linear CONCERT and diversitree fits, with BEAM when available."
        )
    )
    parser.add_argument(
        "--num-transition-times",
        type=int,
        choices=(2, 3),
        default=2,
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Plot the linear_A_to_D simulation results instead of star results.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Combine star_3 and linear_A_to_D results in one figure.",
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.combined:
        if args.linear:
            parser.error("--combined cannot be used with --linear")
        star_data_dir = DEFAULT_DATA_ROOT / "star_3"
        star_row = load_selected_topology_comparison_row(
            star_data_dir,
            "star_3",
            "star",
            STAR3_TOP_TWENTY_BATCHES,
        )
        linear_data_dir = DEFAULT_DATA_ROOT / "linear_A_to_D"
        linear_row = load_selected_topology_comparison_row(
            linear_data_dir,
            "linear A→D",
            "linear",
            LINEAR_TOP_TWENTY_BATCHES,
        )
        output_path = args.output or (
            DEFAULT_DATA_ROOT
            / "figures"
            / "star_3_linear_error_boxplots.png"
        )
        plot_topology_comparison((star_row, linear_row), output_path)
        print(
            "Plotted "
            f"{len(linear_row.common_batch_keys)} common linear and "
            f"{len(star_row.common_batch_keys)} common star batches from the "
            "selected 20 per topology; tree-level methods are limited to "
            "trees 0–7. "
            f"Figure: {output_path}"
        )
        return

    if args.linear:
        data_dir = args.data_dir or DEFAULT_DATA_ROOT / "linear_A_to_D"
        output_path = args.output or (
            data_dir / "figures" / "concert_musse_error_boxplots.png"
        )
        errors = load_matched_linear_absolute_errors(data_dir)
        transition_errors = load_concert_transition_time_errors(data_dir)
        mach2_transition_errors = load_mach2_transition_time_errors(data_dir)
        (
            probability_errors,
            diversitree_probability_errors,
            beam_probability_errors,
            _,
            _,
        ) = load_linear_transition_probability_comparison_errors(data_dir)
        plot_error_boxplots(
            errors,
            output_path,
            transition_time_errors=transition_errors,
            mach2_transition_time_errors=(
                mach2_transition_errors or None
            ),
            transition_probability_errors=probability_errors,
            diversitree_transition_probability_errors=(
                diversitree_probability_errors
            ),
            beam_transition_probability_errors=beam_probability_errors,
            transition_probability_label="Linear edges pooled",
        )
    else:
        data_dir = args.data_dir or (
            DEFAULT_DATA_ROOT / f"star_{args.num_transition_times}"
        )
        output_path = args.output or (
            data_dir / "figures" / "concert_musse_beam_error_boxplots.png"
        )
        errors = load_matched_absolute_errors(data_dir, args.num_transition_times)
        transition_errors = load_star_transition_time_errors(
            data_dir,
            args.num_transition_times,
        )
        mach2_transition_errors = load_mach2_transition_time_errors(data_dir)
        if args.num_transition_times == 3:
            (
                probability_errors,
                diversitree_probability_errors,
                beam_probability_errors,
                _,
                _,
            ) = load_star_transition_probability_comparison_errors(
                data_dir,
                args.num_transition_times,
            )
        else:
            probability_errors = load_star_transition_probability_errors(
                data_dir,
                args.num_transition_times,
            )
            diversitree_probability_errors = None
            beam_probability_errors = None
        plot_error_boxplots(
            errors,
            output_path,
            transition_time_errors=transition_errors,
            mach2_transition_time_errors=(
                mach2_transition_errors or None
            ),
            transition_probability_errors=probability_errors,
            diversitree_transition_probability_errors=(
                diversitree_probability_errors
            ),
            beam_transition_probability_errors=beam_probability_errors,
            transition_probability_label="A→B–D pooled",
        )
    sample_counts = {
        method: len(
            {
                (error.setting, error.batch, error.tree)
                for error in errors
                if error.method == method
            }
        )
        for method in METHODS
    }
    print(
        "Plotted errors for "
        f"{sample_counts['CONCERT']} CONCERT batches, "
        f"{sample_counts['MuSSE']} diversitree batches, and "
        f"{sample_counts['BEAM']} BEAM trees to {output_path}"
    )


if __name__ == "__main__":
    main()
