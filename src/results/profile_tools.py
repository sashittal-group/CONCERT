from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParameterProfile:
    name: str
    parameter_type: str
    index: int
    original_value: float
    values: np.ndarray


def parameter_grid(
    original_value: float,
    lower_bound: float,
    upper_bound: float,
    grid_points: int,
    parameter_name: str,
) -> np.ndarray:
    if (
        not lower_bound < upper_bound
        or not lower_bound <= original_value <= upper_bound
    ):
        raise ValueError(
            f"Invalid feasible grid for {parameter_name}: "
            f"[{lower_bound}, {upper_bound}] does not contain {original_value}"
        )
    return np.unique(
        np.concatenate(
            (np.linspace(lower_bound, upper_bound, grid_points), [original_value])
        )
    )


def threshold_bounds(
    values: np.ndarray,
    drops: np.ndarray,
    target_drop: float,
) -> tuple[tuple[float, float, bool], tuple[float, float, bool]]:
    best = int(np.argmin(drops))

    def interpolate(
        branch_values: np.ndarray,
        branch_drops: np.ndarray,
    ) -> tuple[float, float, bool]:
        maximum_drop = float(np.max(branch_drops))
        domain_limit = float(branch_values[-1])
        if maximum_drop < target_drop:
            return domain_limit, maximum_drop, True
        for index in range(len(branch_values) - 1):
            left_drop, right_drop = branch_drops[index:index + 2]
            if min(left_drop, right_drop) <= target_drop <= max(left_drop, right_drop):
                if left_drop == right_drop:
                    value = float(branch_values[index])
                else:
                    fraction = (target_drop - left_drop) / (right_drop - left_drop)
                    value = float(
                        branch_values[index]
                        + fraction * (branch_values[index + 1] - branch_values[index])
                    )
                is_limit = bool(
                    np.isclose(value, domain_limit, rtol=1e-12, atol=0.0)
                )
                return value, maximum_drop, is_limit
        return domain_limit, maximum_drop, True

    lower = interpolate(values[:best + 1][::-1], drops[:best + 1][::-1])
    higher = interpolate(values[best:], drops[best:])
    return lower, higher


def profile_rows(
    dataset: str,
    profile: ParameterProfile,
    log_likelihoods: np.ndarray,
    baseline: float,
    likelihood_fractions: tuple[float, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    drops = baseline - log_likelihoods
    best_index = int(np.argmin(drops))
    metadata = {
        "dataset": dataset,
        "parameter": profile.name,
        "parameter_type": profile.parameter_type,
        "parameter_index": profile.index,
        "original_value": profile.original_value,
    }
    profile_rows = [
        {
            **metadata,
            "parameter_value": float(value),
            "summed_log_likelihood": float(log_likelihood),
        }
        for value, log_likelihood in zip(profile.values, log_likelihoods)
    ]
    range_rows = []
    for alpha in likelihood_fractions:
        lower, higher = threshold_bounds(profile.values, drops, -np.log(alpha))
        range_rows.append(
            {
                **metadata,
                "likelihood_fraction": alpha,
                "profile_best_value": float(profile.values[best_index]),
                "lower_parameter_value": lower[0],
                "higher_parameter_value": higher[0],
                "lower_maximum_drop_in_domain": lower[1],
                "higher_maximum_drop_in_domain": higher[1],
                "lower_bound_is_domain_limit": lower[2],
                "higher_bound_is_domain_limit": higher[2],
            }
        )
    return profile_rows, range_rows
