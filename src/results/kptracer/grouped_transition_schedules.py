from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import re


@dataclass
class GroupedTransitionSchedule:
    schedule_id: str
    transition_times: tuple[float, ...]
    source_solutions: set[tuple[int, int]] = field(default_factory=set)


def schedule_id_for_times(transition_times: tuple[float, ...]) -> str:
    if not transition_times or any(
        not math.isfinite(time) or not 0.0 < time < 1.0
        for time in transition_times
    ):
        raise ValueError("Transition times must be finite values in (0, 1)")
    if any(
        current <= previous
        for previous, current in zip(transition_times, transition_times[1:])
    ):
        raise ValueError("Transition times must be strictly increasing")

    canonical_times = ",".join(f"{time:.12g}" for time in transition_times)
    digest = hashlib.sha256(canonical_times.encode("ascii")).hexdigest()[:16]
    return f"n{len(transition_times)}-{digest}"


def load_grouped_transition_schedules(
    table_paths: list[str | Path],
) -> dict[str, GroupedTransitionSchedule]:
    grouped_rows: dict[tuple[int, int], dict[int, float]] = {}
    for table_path in table_paths:
        path = Path(table_path)
        if not path.is_file():
            raise FileNotFoundError(f"Grouped schedule table does not exist: {path}")
        with path.open(newline="") as table_file:
            reader = csv.DictReader(table_file, delimiter="\t")
            required_columns = {
                "solution_index",
                "sharptni_gamma",
                "group_index_by_time",
                "grouped_time_carta",
            }
            if not reader.fieldnames or not required_columns.issubset(
                reader.fieldnames
            ):
                raise ValueError(
                    f"Grouped schedule table is missing required columns: {path}"
                )
            for row in reader:
                try:
                    gamma = int(row["sharptni_gamma"])
                    solution_index = int(row["solution_index"])
                    group_index = int(row["group_index_by_time"])
                    transition_time = float(row["grouped_time_carta"])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid grouped schedule row in {path}: {row}"
                    ) from error
                solution_key = (gamma, solution_index)
                groups = grouped_rows.setdefault(solution_key, {})
                if group_index in groups:
                    raise ValueError(
                        f"Duplicate group {group_index} in solution "
                        f"{solution_key} from {path}"
                    )
                groups[group_index] = transition_time

    schedules: dict[str, GroupedTransitionSchedule] = {}
    for (gamma, solution_index), group_times in sorted(grouped_rows.items()):
        group_indices = sorted(group_times)
        if group_indices != list(range(1, len(group_indices) + 1)):
            raise ValueError(
                f"Non-contiguous group indices for gamma={gamma}, "
                f"solution={solution_index}"
            )
        times = tuple(
            sorted(group_times[index] for index in group_indices)
        )
        schedule_id = schedule_id_for_times(times)
        schedule = schedules.get(schedule_id)
        if schedule is None:
            schedule = GroupedTransitionSchedule(schedule_id, times)
            schedules[schedule_id] = schedule
        elif schedule.transition_times != times:
            raise ValueError(f"Schedule ID collision for {schedule_id}")
        schedule.source_solutions.add((gamma, solution_index))

    if not schedules:
        raise ValueError("No grouped transition schedules were found")
    return dict(sorted(schedules.items()))


def read_fit_log_likelihood(path: str | Path) -> float:
    match = re.search(r"^LL=(\S+)$", Path(path).read_text(), re.MULTILINE)
    if match is None:
        raise ValueError(f"No final log likelihood found in {path}")
    try:
        likelihood = float(match.group(1))
    except ValueError as error:
        raise ValueError(f"Invalid final log likelihood in {path}") from error
    if not math.isfinite(likelihood):
        raise ValueError(f"Final log likelihood in {path} is not finite")
    return likelihood
