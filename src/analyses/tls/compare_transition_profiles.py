from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.analyses.common.profile_config import PipelineConfig, load_config
from src.analyses.common.profile_plot import plot_profile_data


DATASETS = ("TLS", "TLSC")
TABLE_SUFFIXES = {
    "profiles": "composite_parameter_profiles.csv",
    "ranges": "composite_parameter_value_ranges.csv",
    "metadata": "composite_parameter_metadata.csv",
}


def _read_profile_tables(
    profiles_dir: Path,
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = profiles_dir / "data" / f"{dataset}_transition_times_100"
    ranges_path = data_dir / TABLE_SUFFIXES["ranges"]
    metadata_path = data_dir / TABLE_SUFFIXES["metadata"]
    ranges = pd.read_csv(ranges_path)
    metadata = pd.read_csv(metadata_path)
    required_ranges = {
        "parameter",
        "likelihood_fraction",
        "lower_parameter_value",
        "higher_parameter_value",
    }
    required_metadata = {
        "parameter",
        "parameter_type",
        "parameter_index",
        "original_value",
    }
    if missing := required_ranges.difference(ranges.columns):
        raise ValueError(f"{ranges_path} is missing columns: {sorted(missing)}")
    if missing := required_metadata.difference(metadata.columns):
        raise ValueError(f"{metadata_path} is missing columns: {sorted(missing)}")
    return ranges, metadata


def plot_transition_profile_comparison(
    tls_profiles_dir: Path,
    tlsc_profiles_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
) -> list[Path]:
    dataset_ids = tuple(
        f"{dataset}_transition_times_100" for dataset in DATASETS
    )
    profile_dirs = (tls_profiles_dir, tlsc_profiles_dir)
    ranges_by_dataset = {}
    metadata_by_dataset = {}
    parameter_sets = []
    likelihood_fractions = []
    for dataset, profiles_dir in zip(dataset_ids, profile_dirs):
        label = dataset.removesuffix("_transition_times_100")
        ranges, metadata = _read_profile_tables(profiles_dir, label)
        ranges_by_dataset[dataset] = ranges
        metadata_by_dataset[dataset] = metadata
        parameter_sets.append(set(metadata["parameter"]))
        likelihood_fractions.append(set(ranges["likelihood_fraction"]))
    if parameter_sets[0] != parameter_sets[1]:
        raise ValueError("TLS and TLSC transition-time profiles have different parameters.")
    if likelihood_fractions[0] != likelihood_fractions[1]:
        raise ValueError(
            "TLS and TLSC transition-time profiles use different likelihood fractions."
        )

    plotting = dict(config.plotting)
    plotting.update(
        {
            "output_basename": "TLS_vs_TLSC_transition_times_100_composite_sensitivity",
            "parameter_output_basenames": {
                "birth": "TLS_vs_TLSC_transition_times_100_birth_rates",
                "death": "TLS_vs_TLSC_transition_times_100_death_rates",
                "transition": (
                    "TLS_vs_TLSC_transition_times_100_transition_probabilities"
                ),
                "sampling": "TLS_vs_TLSC_transition_times_100_sampling_rates",
            },
            "dataset_colors": {
                dataset: config.plotting["dataset_colors"][label]
                for dataset, label in zip(dataset_ids, DATASETS)
            },
            "dataset_labels": dict(zip(dataset_ids, DATASETS)),
            "formats": ["png"],
        }
    )
    plot_config = replace(
        config,
        datasets=dataset_ids,
        data_root=output_dir / "data",
        preprocessing_root=output_dir / "data",
        figure_root=output_dir,
        plotting=plotting,
    )
    figures = plot_profile_data(
        plot_config,
        ranges_by_dataset,
        metadata_by_dataset,
    )
    for figure in figures:
        print(f"Wrote {figure}")
    return figures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot TLS and TLSC 100-transition-time profiles side by side."
    )
    parser.add_argument("--tls-profiles", type=Path, required=True)
    parser.add_argument("--tlsc-profiles", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tls/fixed_sampling_0.1/profile_comparison"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config_fixed_sampling_0.1.yaml"),
    )
    arguments = parser.parse_args()
    plot_transition_profile_comparison(
        arguments.tls_profiles,
        arguments.tlsc_profiles,
        arguments.output_dir,
        load_config(arguments.config),
    )


if __name__ == "__main__":
    main()
