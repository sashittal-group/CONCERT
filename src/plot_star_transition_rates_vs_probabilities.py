from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STATES = ("B", "C", "D")
METHODS = ("BEAM", "diversitree")
METHOD_COLORS = {"BEAM": "#009E73", "diversitree": "#E69F00"}
BEAM_RATE_COLUMNS = {
    state: f"tissueSubstModelLogger.relMigRate_A_{state}" for state in STATES
}
DIVERSITREE_RATE_COLUMNS = {"B": "q12", "C": "q13", "D": "q14"}
DEFAULT_DATA_DIR = Path("data/simulation/coord_diff/star_3")


@dataclass(frozen=True)
class RateObservation:
    method: str
    setting: str
    transition: str
    probability: float
    rate: float
    batch: str | None = None


@dataclass(frozen=True)
class CalibratedProbability:
    setting: str
    batch: str | None
    transition: str
    probability: float
    estimate: float
    slope: float
    clipped: bool


def _read_ground_truth(data_dir: Path) -> dict[str, dict[str, float]]:
    path = data_dir / "generated_settings.csv"
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        required = {"setting", *(f"p_A_to_{state}" for state in STATES)}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing transition probability columns in {path}")
        values = {}
        for row in reader:
            setting = row["setting"]
            if not setting or setting in values:
                raise ValueError(f"Invalid or duplicate setting in {path}: {setting!r}")
            probabilities = {
                state: float(row[f"p_A_to_{state}"]) for state in STATES
            }
            if any(
                not math.isfinite(probability) or not 0.0 <= probability <= 1.0
                for probability in probabilities.values()
            ):
                raise ValueError(f"Invalid transition probability for {setting}")
            values[setting] = probabilities
    return values


def _read_summary(path: Path) -> dict[str, str]:
    return {
        key: value
        for line in path.read_text().splitlines()
        if "=" in line
        for key, value in [line.split("=", maxsplit=1)]
    }


def _read_diversitree_rates(path: Path) -> dict[str, float]:
    summary = _read_summary(path)
    if summary.get("convergence_code") != "0":
        raise ValueError(f"diversitree fit did not converge: {path}")

    estimate_path = path.with_name("parameter_estimates.csv")
    with estimate_path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    rates = {}
    for state, parameter in DIVERSITREE_RATE_COLUMNS.items():
        matches = [row for row in rows if row.get("parameter") == parameter]
        if len(matches) != 1:
            raise ValueError(f"Expected one {parameter} estimate in {estimate_path}")
        rate = float(matches[0]["estimate"])
        if not math.isfinite(rate) or rate < 0:
            raise ValueError(f"Invalid {parameter} estimate in {estimate_path}")
        rates[state] = rate
    return rates


def _read_beam_rates(
    summary_path: Path,
    rate_columns: dict[str, str] | None = None,
) -> dict[str, float]:
    run_dir = summary_path.parent
    status_path = run_dir / "run_status.tsv"
    with status_path.open(newline="") as source:
        statuses = list(csv.DictReader(source, delimiter="\t"))
    if len(statuses) != 1 or statuses[0].get("status") != "completed":
        raise ValueError(f"BEAM run is not marked completed in {status_path}")

    with summary_path.open(newline="") as source:
        summary_rows = list(csv.DictReader(source, delimiter="\t"))
    try:
        post_burnin_counts = {
            int(row["samples_after_burnin"]) for row in summary_rows
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid BEAM summary sample counts in {summary_path}") from error
    if len(post_burnin_counts) != 1:
        raise ValueError(f"Inconsistent BEAM sample counts in {summary_path}")
    post_burnin_count = post_burnin_counts.pop()

    log_path = run_dir / "result.log"
    with log_path.open(newline="") as source:
        rows = list(
            csv.DictReader(
                (
                    line
                    for line in source
                    if line.strip() and not line.startswith("#")
                ),
                delimiter="\t",
            )
        )
    burnin_count = len(rows) - post_burnin_count
    if post_burnin_count <= 0 or burnin_count < 0:
        raise ValueError(f"Invalid BEAM burn-in count in {log_path}")
    rows = rows[burnin_count:]
    if not rows:
        raise ValueError(f"No post-burn-in BEAM samples in {log_path}")

    columns = BEAM_RATE_COLUMNS if rate_columns is None else rate_columns
    rates = {state: 0.0 for state in columns}
    for row in rows:
        try:
            clock_rate = float(row["tissueClockRate"])
            for state, column in columns.items():
                rates[state] += clock_rate * float(row[column])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid BEAM migration sample in {log_path}") from error
    for state in columns:
        rates[state] /= len(rows)
        if not math.isfinite(rates[state]) or rates[state] < 0:
            raise ValueError(f"Invalid BEAM transition {state} rate in {log_path}")
    return rates


def collect_rate_observations(data_dir: Path) -> list[RateObservation]:
    truth = _read_ground_truth(data_dir)
    observations = []
    for path in sorted(
        data_dir.glob("setting_*/batch_*/diversitree_musse/joint_musse_fit.txt")
    ):
        setting = path.parent.parent.parent.name
        if setting not in truth:
            warnings.warn(f"No ground truth for {setting}; skipping {path}")
            continue
        try:
            rates = _read_diversitree_rates(path)
        except (OSError, ValueError) as error:
            warnings.warn(f"Skipping diversitree output {path}: {error}")
            continue
        observations.extend(
            RateObservation(
                method="diversitree",
                setting=setting,
                transition=state,
                probability=truth[setting][state],
                rate=rates[state],
                batch=path.parent.parent.name,
            )
            for state in STATES
        )

    for path in sorted(
        data_dir.glob("setting_*/batch_*/tree_*/beam_snakemake/parameter_summary.tsv")
    ):
        setting = path.parent.parent.parent.parent.name
        if setting not in truth:
            warnings.warn(f"No ground truth for {setting}; skipping {path}")
            continue
        try:
            rates = _read_beam_rates(path)
        except (OSError, ValueError) as error:
            warnings.warn(f"Skipping BEAM output {path}: {error}")
            continue
        observations.extend(
            RateObservation(
                method="BEAM",
                setting=setting,
                transition=state,
                probability=truth[setting][state],
                rate=rates[state],
                batch=path.parent.parent.parent.name,
            )
            for state in STATES
        )

    for method in METHODS:
        if not any(observation.method == method for observation in observations):
            raise ValueError(f"No usable {method} transition-rate estimates in {data_dir}")
    return observations


def _proportionality_statistics(
    probabilities: list[float],
    rates: list[float],
) -> tuple[float | None, float | None, float | None]:
    if len(probabilities) != len(rates) or not probabilities:
        raise ValueError("Proportionality statistics require paired observations")

    sum_squares = sum(probability**2 for probability in probabilities)
    slope = (
        sum(probability * rate for probability, rate in zip(probabilities, rates))
        / sum_squares
        if sum_squares > 0
        else None
    )
    mean_probability = sum(probabilities) / len(probabilities)
    mean_rate = sum(rates) / len(rates)
    correlation_denominator = math.sqrt(
        sum((probability - mean_probability) ** 2 for probability in probabilities)
        * sum((rate - mean_rate) ** 2 for rate in rates)
    )
    pearson_r = (
        sum(
            (probability - mean_probability) * (rate - mean_rate)
            for probability, rate in zip(probabilities, rates)
        )
        / correlation_denominator
        if correlation_denominator > 0
        else None
    )
    total_squares = sum((rate - mean_rate) ** 2 for rate in rates)
    r_squared = (
        1.0
        - sum(
            (rate - slope * probability) ** 2
            for probability, rate in zip(probabilities, rates)
        )
        / total_squares
        if slope is not None and total_squares > 0
        else None
    )
    return slope, pearson_r, r_squared


def calibrate_rates_to_probabilities(
    observations: list[RateObservation],
    method: str,
) -> list[CalibratedProbability]:
    method_observations = [
        observation for observation in observations if observation.method == method
    ]
    if not method_observations:
        raise ValueError(f"No {method} observations available for calibration")

    calibrated = []
    for transition in STATES:
        transition_observations = [
            observation
            for observation in method_observations
            if observation.transition == transition
        ]
        settings = sorted(
            {observation.setting for observation in transition_observations}
        )
        for held_out_setting in settings:
            training = [
                observation
                for observation in transition_observations
                if observation.setting != held_out_setting
            ]
            slope, _, _ = _proportionality_statistics(
                [observation.probability for observation in training],
                [observation.rate for observation in training],
            )
            if slope is None or slope <= 0.0:
                raise ValueError(
                    f"Cannot fit a positive {method} slope for A->{transition}"
                )
            for observation in transition_observations:
                if observation.setting != held_out_setting:
                    continue
                raw_estimate = observation.rate / slope
                calibrated.append(
                    CalibratedProbability(
                        setting=observation.setting,
                        batch=observation.batch,
                        transition=transition,
                        probability=observation.probability,
                        estimate=min(1.0, max(0.0, raw_estimate)),
                        slope=slope,
                        clipped=not 0.0 <= raw_estimate <= 1.0,
                    )
                )
    if len(calibrated) != len(method_observations):
        raise ValueError("Could not calibrate every rate observation")
    return calibrated


def plot_rate_probability_relationships(
    observations: list[RateObservation],
    output_path: Path,
) -> Path:
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), sharex="col")
    for row_index, method in enumerate(METHODS):
        for column_index, state in enumerate(STATES):
            axis = axes[row_index, column_index]
            pairs = [
                observation
                for observation in observations
                if observation.method == method and observation.transition == state
            ]
            if not pairs:
                raise ValueError(f"No {method} observations for A->{state}")
            probabilities = [pair.probability for pair in pairs]
            rates = [pair.rate for pair in pairs]
            slope, pearson_r, r_squared = _proportionality_statistics(
                probabilities,
                rates,
            )

            axis.scatter(
                probabilities,
                rates,
                s=24,
                alpha=0.55,
                color=METHOD_COLORS[method],
                edgecolors="none",
            )
            if slope is not None:
                upper_probability = max(probabilities) * 1.05
                axis.plot(
                    [0, upper_probability],
                    [0, slope * upper_probability],
                    color="#444444",
                    linestyle="--",
                    linewidth=1.2,
                )
            score = lambda value: "undefined" if value is None else f"{value:.3f}"
            axis.text(
                0.04,
                0.96,
                f"slope = {score(slope)}\n"
                f"Pearson r = {score(pearson_r)}\n"
                f"R² (through origin) = {score(r_squared)}\n"
                f"n = {len(pairs)}",
                transform=axis.transAxes,
                va="top",
                ha="left",
                bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#cccccc"},
            )
            axis.set_title(f"{method}: A -> {state}")
            axis.set_xlabel(f"Configured transition probability (A -> {state})")
            axis.set_ylabel("Inferred transition rate (per time unit)")
            axis.grid(True, alpha=0.2)

    figure.suptitle(
        "star_3 transition rates vs. scheduled transition probabilities\n"
        "Dashed lines are proportional fits through the origin; "
        "BEAM rates include the tissue clock multiplier",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare star_3 BEAM and diversitree transition-rate estimates "
            "with configured A->B/C/D transition probabilities."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATA_DIR
        / "figures"
        / "transition_rates_vs_probabilities.png",
    )
    args = parser.parse_args()

    observations = collect_rate_observations(args.data_dir)
    output_path = plot_rate_probability_relationships(observations, args.output)
    print(
        f"Plotted {sum(item.method == 'BEAM' for item in observations) // 3} "
        "BEAM tree fits and "
        f"{sum(item.method == 'diversitree' for item in observations) // 3} "
        f"converged diversitree fits to {output_path}"
    )


if __name__ == "__main__":
    main()
