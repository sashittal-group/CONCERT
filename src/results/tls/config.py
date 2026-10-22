from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    data_root: Path
    preprocessing_root: Path
    figure_root: Path
    datasets: tuple[str, ...]
    state_names: tuple[str, ...]
    likelihood_fractions: tuple[float, ...]
    profile_grid_points: int
    rate_epsilon: float
    probability_epsilon: float
    max_birth_rate: float
    profile_filename: str
    ranges_filename: str
    metadata_filename: str
    plotting: dict[str, Any]

    def dataset_root(self, dataset: str) -> Path:
        if dataset not in self.datasets:
            raise ValueError(
                f"Unknown dataset {dataset!r}; expected one of {self.datasets}"
            )
        return self.data_root / dataset

    def dataset_output_dir(self, dataset: str) -> Path:
        self.dataset_root(dataset)
        return self.preprocessing_root / dataset


def load_config(path: Path) -> PipelineConfig:
    config_path = path.resolve()
    with config_path.open() as config_file:
        document = yaml.safe_load(config_file)
    if not isinstance(document, dict):
        raise ValueError(f"Configuration in {config_path} must be a mapping")

    settings = document.get("tls_composite")
    if not isinstance(settings, dict):
        raise ValueError(f"{config_path} must define a tls_composite mapping")

    preprocessing = settings.get("preprocessing")
    files = settings.get("files")
    plotting = settings.get("plotting")
    if not isinstance(preprocessing, dict) or not isinstance(files, dict):
        raise ValueError("tls_composite.preprocessing and tls_composite.files are required")
    if not isinstance(plotting, dict):
        raise ValueError("tls_composite.plotting must be a mapping")

    base_dir = config_path.parent

    def configured_path(name: str) -> Path:
        value = settings.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"tls_composite.{name} must be a non-empty path")
        configured = Path(value)
        if configured.is_absolute():
            return configured.resolve()
        return (base_dir / configured).resolve()

    raw_datasets = settings.get("datasets")
    raw_state_names = settings.get("state_names")
    raw_fractions = settings.get("likelihood_fractions")
    if not isinstance(raw_datasets, list) or not all(
        isinstance(value, str) and value for value in raw_datasets
    ):
        raise ValueError("tls_composite.datasets must be a list of names")
    if not isinstance(raw_state_names, list) or not all(
        isinstance(value, str) and value for value in raw_state_names
    ):
        raise ValueError("tls_composite.state_names must be a list of names")
    if not isinstance(raw_fractions, list):
        raise ValueError("tls_composite.likelihood_fractions must be a list")

    datasets = tuple(raw_datasets)
    state_names = tuple(raw_state_names)
    fractions = tuple(float(value) for value in raw_fractions)
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("tls_composite.datasets must contain unique dataset names")
    if len(state_names) != 3 or len(set(state_names)) != len(state_names):
        raise ValueError("tls_composite.state_names must contain three unique states")
    if (
        not fractions
        or len(set(fractions)) != len(fractions)
        or any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in fractions)
    ):
        raise ValueError(
            "tls_composite.likelihood_fractions must be unique values in (0, 1)"
        )

    grid_points = int(preprocessing.get("profile_grid_points", 0))
    rate_epsilon = float(preprocessing.get("rate_epsilon", 0.0))
    probability_epsilon = float(preprocessing.get("probability_epsilon", 0.0))
    max_birth_rate = float(preprocessing.get("max_birth_rate", 0.0))
    if grid_points < 2:
        raise ValueError("profile_grid_points must be at least 2")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (rate_epsilon, probability_epsilon, max_birth_rate)
    ):
        raise ValueError("Rate/probability epsilons and max_birth_rate must be positive")

    for key in ("profiles", "ranges", "metadata"):
        if not isinstance(files.get(key), str) or not files[key]:
            raise ValueError(f"tls_composite.files.{key} must be a filename")
    required_plot_options = {
        "output_basename",
        "parameter_output_basenames",
        "formats",
        "figsize",
        "parameter_figsize",
        "group_spacing",
        "dataset_offset",
        "x_limits",
        "single_group_x_limits",
        "font_size",
        "font_family",
        "math_fontset",
        "boxplot_likelihood_fraction",
        "whisker_likelihood_fraction",
        "box_width",
        "dataset_colors",
    }
    missing_plot_options = required_plot_options.difference(plotting)
    if missing_plot_options:
        raise ValueError(
            f"tls_composite.plotting is missing: {sorted(missing_plot_options)}"
        )
    if not isinstance(plotting["formats"], list) or not plotting["formats"]:
        raise ValueError("tls_composite.plotting.formats must be a non-empty list")
    required_parameter_plots = {"birth", "death", "transition", "sampling"}
    if (
        not isinstance(plotting["parameter_output_basenames"], dict)
        or required_parameter_plots.difference(
            plotting["parameter_output_basenames"]
        )
    ):
        raise ValueError(
            "parameter_output_basenames must define birth, death, transition, and sampling"
        )
    boxplot_fraction = float(plotting["boxplot_likelihood_fraction"])
    whisker_fraction = float(plotting["whisker_likelihood_fraction"])
    if boxplot_fraction not in fractions or whisker_fraction not in fractions:
        raise ValueError(
            "Boxplot and whisker likelihood fractions must be in likelihood_fractions"
        )
    if whisker_fraction >= boxplot_fraction:
        raise ValueError(
            "whisker_likelihood_fraction must be lower than boxplot_likelihood_fraction"
        )
    if not isinstance(plotting["dataset_colors"], dict) or any(
        dataset not in plotting["dataset_colors"] for dataset in datasets
    ):
        raise ValueError("tls_composite.plotting.dataset_colors must cover all datasets")

    return PipelineConfig(
        data_root=configured_path("data_root"),
        preprocessing_root=configured_path("preprocessing_root"),
        figure_root=configured_path("figure_root"),
        datasets=datasets,
        state_names=state_names,
        likelihood_fractions=fractions,
        profile_grid_points=grid_points,
        rate_epsilon=rate_epsilon,
        probability_epsilon=probability_epsilon,
        max_birth_rate=max_birth_rate,
        profile_filename=files["profiles"],
        ranges_filename=files["ranges"],
        metadata_filename=files["metadata"],
        plotting=plotting,
    )
