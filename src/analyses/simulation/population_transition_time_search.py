from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import heapq
from itertools import product
import json
import math
from pathlib import Path
import random
import time

import torch

from src.concert.infer_parameters import read_free_edges, read_transition_times
from src.concert.infer_joint_transition_times import (
    _build_datastructures,
    _load_trees,
    _project_transition_times_to_grid,
)
from src.concert.optimize_tree_parameters import (
    LikelihoodModel,
    PhylodynamicParameters,
)


@dataclass
class ScheduleCandidate:
    schedule_id: int
    times: tuple[float, ...]
    operator: str
    stage: int = 1
    grid_resolution: int = 1000
    parent_ids: tuple[int, ...] = ()
    score: float = -math.inf
    parameters: dict | None = None
    optimizer_state: dict | None = None
    cached_datastructures: list | None = None
    optimization_steps: int = 0
    fit_blocks: int = 0
    steps_without_improvement: int = 0
    eligible_after: int = 0
    fit_status: str = "pending"
    freeze_reason: str | None = None
    fit_error: str | None = None


def _parse_initial_output(
    path: Path,
    num_states: int,
) -> tuple[tuple[float, ...], dict[str, list]]:
    lines = path.read_text().splitlines()

    def parse_vector(name: str) -> list[float]:
        matches = [
            line for line in lines if line.startswith(f"{name}:")
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one {name} vector in {path}")
        values = [
            float(value)
            for value in matches[0].split(":", maxsplit=1)[1].split()
        ]
        if len(values) != num_states or not all(math.isfinite(x) for x in values):
            raise ValueError(f"Expected {num_states} finite {name} values in {path}")
        return values

    transition_probabilities = [
        [0.0] * num_states for _ in range(num_states)
    ]
    transition_times = []
    section = None
    for line in lines:
        stripped = line.strip()
        if stripped == "transitions:":
            section = "transitions"
            continue
        if stripped == "transition_times:":
            section = "times"
            continue
        if section == "transitions":
            source, separator, remainder = stripped.partition("->")
            if separator:
                target, _, raw_probability = remainder.partition(" ")
                i, j = int(source), int(target)
                probability = float(raw_probability)
                if not 0 <= i < num_states or not 0 <= j < num_states:
                    raise ValueError(f"Invalid transition edge in {path}: {stripped}")
                if not math.isfinite(probability) or probability < 0:
                    raise ValueError(f"Invalid transition probability in {path}")
                transition_probabilities[i][j] = probability
        elif section == "times" and stripped:
            index, separator, raw_time = stripped.partition(" ")
            if separator:
                transition_times.append(float(raw_time))

    if not transition_times or any(
        not math.isfinite(value) for value in transition_times
    ):
        raise ValueError(f"Missing finite transition times in {path}")
    if any(right <= left for left, right in zip(transition_times, transition_times[1:])):
        raise ValueError(f"Transition times are not strictly increasing in {path}")

    return tuple(transition_times), {
        "initial_birth_rates": parse_vector("birth"),
        "initial_death_rates": parse_vector("death"),
        "initial_sampling_probabilities": parse_vector("sampling"),
        "initial_transition_probabilities": transition_probabilities,
    }


def _fit_candidate(
    candidate: ScheduleCandidate,
    trees,
    num_states: int,
    free_edges: list[tuple[int, int]],
    device: torch.device,
    initial_parameters: dict | None,
    max_steps: int,
    patience: int,
    max_total_steps: int = 300,
    improvement_tolerance: float = 1e-4,
    verbose: bool = False,
) -> None:
    if candidate.fit_status in {"invalid", "frozen"}:
        raise ValueError(
            f"Cannot fit schedule {candidate.schedule_id} in "
            f"{candidate.fit_status} state"
        )
    remaining_steps = max_total_steps - candidate.optimization_steps
    if remaining_steps <= 0:
        candidate.fit_status = "frozen"
        candidate.freeze_reason = "step_cap"
        candidate.cached_datastructures = None
        candidate.optimizer_state = None
        return

    params = PhylodynamicParameters(
        num_states=num_states,
        free_edges=free_edges,
        given_transition_times=list(candidate.times),
        device=device,
        **deepcopy(candidate.parameters or initial_parameters or {}),
    ).to(device)
    if candidate.cached_datastructures is None:
        candidate.cached_datastructures = _build_datastructures(
            trees, list(candidate.times), num_states, verbose=verbose
        )
    model = LikelihoodModel(
        params=params,
        trees=trees,
        datastructures=candidate.cached_datastructures,
        normalize_root_likelihood=True,
    )
    trainable_parameters = [
        parameter for parameter in params.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters for schedule fit")

    optimizer = torch.optim.Adam(trainable_parameters, lr=0.01)
    if candidate.optimizer_state is not None:
        optimizer.load_state_dict(deepcopy(candidate.optimizer_state))

    has_prior_fit = (
        candidate.parameters is not None and math.isfinite(candidate.score)
    )
    best_score = candidate.score if has_prior_fit else -math.inf
    best_state = (
        {
            name: value.detach().clone()
            for name, value in params.state_dict().items()
        }
        if has_prior_fit
        else None
    )
    best_optimizer_state = deepcopy(candidate.optimizer_state)
    steps_without_improvement = candidate.steps_without_improvement
    candidate.fit_error = None
    candidate.freeze_reason = None
    candidate.fit_blocks += 1
    steps_this_block = min(max_steps, remaining_steps)

    for _ in range(steps_this_block):
        step = candidate.optimization_steps + 1
        optimizer.zero_grad(set_to_none=True)
        likelihood = model()
        if not torch.isfinite(likelihood):
            candidate.optimization_steps = step
            candidate.fit_error = (
                f"Non-finite likelihood at step {step} for schedule "
                f"{candidate.schedule_id}: {candidate.times}"
            )
            if best_state is None:
                candidate.score = -math.inf
                candidate.fit_status = "invalid"
                candidate.cached_datastructures = None
                candidate.optimizer_state = None
                return
            candidate.fit_status = "frozen"
            candidate.freeze_reason = "nonfinite"
            break

        score = float(likelihood.detach().item())
        improvement = score - best_score
        if score > best_score:
            best_score = score
            best_state = {
                name: value.detach().clone()
                for name, value in params.state_dict().items()
            }
            best_optimizer_state = deepcopy(optimizer.state_dict())
            if improvement >= improvement_tolerance:
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
        else:
            steps_without_improvement += 1

        candidate.optimization_steps = step
        candidate.steps_without_improvement = steps_without_improvement
        if steps_without_improvement >= patience:
            candidate.fit_status = "frozen"
            candidate.freeze_reason = "plateau"
            break

        (-likelihood).backward()
        optimizer.step()

    if best_state is None:
        candidate.fit_status = "invalid"
        candidate.score = -math.inf
        candidate.cached_datastructures = None
        candidate.optimizer_state = None
        return

    params.load_state_dict(best_state)
    candidate.score = best_score
    if candidate.fit_status not in {"frozen", "invalid"}:
        if candidate.optimization_steps >= max_total_steps:
            candidate.fit_status = "frozen"
            candidate.freeze_reason = "step_cap"
        else:
            candidate.fit_status = "active"
    candidate.parameters = {
        "initial_birth_rates": params.birth_rates.detach().cpu().tolist(),
        "initial_death_rates": params.death_rates.detach().cpu().tolist(),
        "initial_sampling_probabilities": (
            params.sampling_probabilities.detach().cpu().tolist()
        ),
        "initial_transition_probabilities": (
            params.transition_probabilities.detach().cpu().tolist()
        ),
    }
    if candidate.fit_status == "active":
        candidate.optimizer_state = best_optimizer_state
    else:
        candidate.optimizer_state = None
        candidate.cached_datastructures = None


def _schedule_key(
    times: tuple[float, ...],
    max_time: float,
) -> tuple[int, ...]:
    return tuple(round(value * 1000 / max_time) for value in times)


def _push_candidate(
    queue: list[tuple[float, int]],
    candidate: ScheduleCandidate,
) -> None:
    heapq.heappush(queue, (-candidate.score, candidate.schedule_id))


def _pop_eligible_candidate(
    queue: list[tuple[float, int]],
    candidates: dict[int, ScheduleCandidate],
    completed_schedules: int,
    ignore_cooldown: bool = False,
) -> ScheduleCandidate | None:
    deferred = []
    selected = None
    while queue:
        priority = heapq.heappop(queue)
        candidate = candidates[priority[1]]
        if (
            (ignore_cooldown or candidate.eligible_after <= completed_schedules)
            and candidate.parameters is not None
            and math.isfinite(candidate.score)
        ):
            selected = candidate
            break
        deferred.append(priority)
    for priority in deferred:
        heapq.heappush(queue, priority)
    return selected


def _queue_record(
    candidate: ScheduleCandidate,
    completed_schedules: int,
) -> dict:
    if candidate.fit_status == "pending":
        queue_state = "pending"
    elif candidate.parameters is None or not math.isfinite(candidate.score):
        queue_state = "invalid"
    elif candidate.eligible_after > completed_schedules:
        queue_state = "waiting"
    else:
        queue_state = "ready"
    return {
        "schedule_id": candidate.schedule_id,
        "operator": candidate.operator,
        "parent_ids": list(candidate.parent_ids),
        "transition_times": list(candidate.times),
        "stage": candidate.stage,
        "grid_resolution": candidate.grid_resolution,
        "log_likelihood": (
            candidate.score if math.isfinite(candidate.score) else None
        ),
        "fit_status": candidate.fit_status,
        "freeze_reason": candidate.freeze_reason,
        "optimization_steps": candidate.optimization_steps,
        "fit_blocks": candidate.fit_blocks,
        "steps_without_improvement": candidate.steps_without_improvement,
        "queue_state": queue_state,
        "queue_pool": candidate.fit_status,
        "eligible_after_fit_count": candidate.eligible_after,
        "waits_for_fits": (
            max(0, candidate.eligible_after - completed_schedules)
            if queue_state == "waiting"
            else 0
        ),
    }


def _queue_snapshot(
    queue: list[tuple[float, int]],
    candidates: dict[int, ScheduleCandidate],
    completed_schedules: int,
) -> list[dict]:
    ordered_ids = sorted(
        (schedule_id for _, schedule_id in queue),
        key=lambda schedule_id: (
            -candidates[schedule_id].score,
            schedule_id,
        ),
    )
    return [
        {
            "queue_rank": rank,
            **_queue_record(candidates[schedule_id], completed_schedules),
        }
        for rank, schedule_id in enumerate(ordered_ids, start=1)
    ]


def _project_schedule(
    values: list[float],
    max_time: float,
    min_gap: float,
    grid_resolution: int = 1000,
) -> tuple[float, ...]:
    return tuple(
        _project_transition_times_to_grid(
            values, max_time, min_gap, grid_size=grid_resolution
        )
    )


def _random_schedule(
    rng: random.Random,
    num_times: int,
    max_time: float,
    min_gap: float,
    grid_resolution: int = 1000,
) -> tuple[float, ...]:
    if not 1 <= num_times < grid_resolution:
        raise ValueError(
            f"Schedule size must be between 1 and {grid_resolution - 1}"
        )
    return _project_schedule(
        [
            max_time * index / grid_resolution
            for index in rng.sample(range(1, grid_resolution), num_times)
        ],
        max_time,
        min_gap,
        grid_resolution,
    )


def _mutate_schedule(
    parent: tuple[float, ...],
    rng: random.Random,
    max_time: float,
    min_gap: float,
    scale: float,
    grid_resolution: int = 1000,
) -> tuple[float, ...]:
    return _project_schedule(
        [time + rng.gauss(0.0, scale) for time in parent],
        max_time,
        min_gap,
        grid_resolution,
    )


def _crossover_schedule(
    left: tuple[float, ...],
    right: tuple[float, ...],
    rng: random.Random,
    max_time: float,
    min_gap: float,
    grid_resolution: int = 1000,
) -> tuple[float, ...]:
    values = [
        rng.choice((left_time, right_time))
        for left_time, right_time in zip(left, right)
    ]
    return _project_schedule(values, max_time, min_gap, grid_resolution)


def _unique_crossover_schedule(
    left: tuple[float, ...],
    right: tuple[float, ...],
    rng: random.Random,
    max_time: float,
    min_gap: float,
    seen_schedules: set[tuple[int, ...]],
    grid_resolution: int = 1000,
) -> tuple[float, ...] | None:
    if len(left) <= 12:
        unseen = {}
        for choices in product((False, True), repeat=len(left)):
            times = _project_schedule(
                [
                    right_time if choose_right else left_time
                    for left_time, right_time, choose_right in zip(
                        left, right, choices
                    )
                ],
                max_time,
                min_gap,
                grid_resolution,
            )
            key = _schedule_key(times, max_time)
            if key not in seen_schedules:
                unseen[key] = times
        return rng.choice(list(unseen.values())) if unseen else None

    for _ in range(1000):
        times = _crossover_schedule(
            left, right, rng, max_time, min_gap, grid_resolution
        )
        if _schedule_key(times, max_time) not in seen_schedules:
            return times
    return None


def _pop_best_parent(
    active_queue: list[tuple[float, int]],
    frozen_queue: list[tuple[float, int]],
    candidates: dict[int, ScheduleCandidate],
    completed_schedules: int,
) -> tuple[ScheduleCandidate | None, str | None, bool]:
    active = _pop_eligible_candidate(active_queue, candidates, completed_schedules)
    frozen = _pop_eligible_candidate(frozen_queue, candidates, completed_schedules)
    if active is not None or frozen is not None:
        cooldown_overridden = False
    else:
        active = _pop_eligible_candidate(
            active_queue, candidates, completed_schedules, ignore_cooldown=True
        )
        frozen = _pop_eligible_candidate(
            frozen_queue, candidates, completed_schedules, ignore_cooldown=True
        )
        cooldown_overridden = active is not None or frozen is not None
    if active is None and frozen is None:
        return None, None, False
    if active is None:
        return frozen, "frozen", cooldown_overridden
    if frozen is None:
        return active, "active", cooldown_overridden
    if active.score >= frozen.score:
        _push_candidate(frozen_queue, frozen)
        return active, "active", cooldown_overridden
    _push_candidate(active_queue, active)
    return frozen, "frozen", cooldown_overridden


def population_search(
    trees,
    num_states: int,
    free_edges: list[tuple[int, int]],
    initial_times: tuple[float, ...],
    initial_parameters: dict,
    *,
    grid_resolutions: tuple[int, ...] = (25, 100, 250, 1000),
    stage_budgets: tuple[int, ...] = (100, 200, 300, 400),
    parent_cooldown: int = 5,
    crossover_probability: float = 0.5,
    max_time: float = 1.0,
    min_gap: float = 1e-4,
    max_steps: int = 100,
    max_total_steps: int = 300,
    patience: int = 30,
    plateau_tolerance: float = 1e-4,
    random_seed: int = 20261028,
    device: torch.device | None = None,
    verbose: bool = False,
    queue_log_path: Path | None = None,
) -> dict:
    if not grid_resolutions or len(grid_resolutions) != len(stage_budgets):
        raise ValueError("grid_resolutions and stage_budgets must have equal nonzero lengths")
    if any(not isinstance(resolution, int) or resolution < 2 for resolution in grid_resolutions):
        raise ValueError("Grid resolutions must be integers of at least 2")
    if any(1000 % resolution != 0 for resolution in grid_resolutions):
        raise ValueError("Each grid resolution must divide 1000 for cross-stage deduplication")
    if any(right <= left for left, right in zip(grid_resolutions, grid_resolutions[1:])):
        raise ValueError("Grid resolutions must be strictly increasing")
    if any(not isinstance(budget, int) or budget < 1 for budget in stage_budgets):
        raise ValueError("Every stage budget must be a positive integer")
    if parent_cooldown < 0:
        raise ValueError("parent_cooldown must be nonnegative")
    if not 0.0 <= crossover_probability <= 1.0:
        raise ValueError("crossover_probability must be between 0 and 1")
    if (
        not math.isfinite(max_time)
        or max_time <= 0
        or not math.isfinite(min_gap)
        or min_gap <= 0
    ):
        raise ValueError("max_time and min_gap must be positive and finite")
    if max_steps < 1 or max_total_steps < max_steps or patience < 1:
        raise ValueError("Step limits and plateau patience must be positive")
    if not math.isfinite(plateau_tolerance) or plateau_tolerance <= 0:
        raise ValueError("plateau_tolerance must be positive and finite")
    if device is None:
        device = torch.device("cpu")
    if queue_log_path is not None:
        queue_log_path.parent.mkdir(parents=True, exist_ok=True)
        queue_log_path.write_text("")

    rng = random.Random(random_seed)
    num_times = len(initial_times)
    total_schedules = sum(stage_budgets)
    first_resolution = grid_resolutions[0]
    incumbent = _project_schedule(
        list(initial_times), max_time, min_gap, first_resolution
    )
    candidates: dict[int, ScheduleCandidate] = {}
    seen_schedules: set[tuple[int, ...]] = set()
    active_queue: list[tuple[float, int]] = []
    frozen_queue: list[tuple[float, int]] = []
    frozen_archive: dict[int, ScheduleCandidate] = {}
    fit_history: list[dict] = []
    waiting_schedule_ids: set[int] = set()
    completed_schedules = 0
    queue_log_sequence = 0
    global_best: ScheduleCandidate | None = None

    def queue_size() -> int:
        return len(active_queue) + len(frozen_queue)

    def log_queue_event(event_type: str, **details: object) -> None:
        nonlocal queue_log_sequence
        if queue_log_path is None:
            return
        queue_log_sequence += 1
        event = {
            "event_number": queue_log_sequence,
            "event": event_type,
            **details,
        }
        with queue_log_path.open("a") as log_file:
            log_file.write(json.dumps(event, separators=(",", ":")) + "\n")

    def enqueue_candidate(candidate: ScheduleCandidate) -> None:
        if candidate.fit_status == "active":
            _push_candidate(active_queue, candidate)
        elif candidate.fit_status == "frozen":
            frozen_archive[candidate.schedule_id] = candidate
            _push_candidate(frozen_queue, candidate)
        elif candidate.fit_status != "invalid":
            raise RuntimeError(
                f"Schedule {candidate.schedule_id} has unqueueable status "
                f"{candidate.fit_status}"
            )
        log_queue_event(
            "QUEUE_ADD" if candidate.fit_status != "invalid" else "SCHEDULE_ARCHIVED",
            completed_schedules=completed_schedules,
            queue_size=queue_size(),
            schedule=_queue_record(candidate, completed_schedules),
        )

    def queue_snapshot_items() -> list[dict]:
        queued = active_queue + frozen_queue
        items = _queue_snapshot(queued, candidates, completed_schedules)
        invalid = sorted(
            (
                candidate
                for candidate in candidates.values()
                if candidate.fit_status == "invalid"
            ),
            key=lambda candidate: candidate.schedule_id,
        )
        for candidate in invalid:
            items.append(
                {
                    "queue_rank": len(items) + 1,
                    **_queue_record(candidate, completed_schedules),
                }
            )
        return items

    def refresh_waiting_schedules() -> None:
        for schedule_id in sorted(tuple(waiting_schedule_ids)):
            candidate = candidates[schedule_id]
            if candidate.eligible_after <= completed_schedules:
                waiting_schedule_ids.remove(schedule_id)
                log_queue_event(
                    "QUEUE_READY",
                    completed_schedules=completed_schedules,
                    queue_size=queue_size(),
                    schedule=_queue_record(candidate, completed_schedules),
                )

    log_queue_event(
        "SEARCH_STARTED",
        grid_resolutions=list(grid_resolutions),
        stage_budgets=list(stage_budgets),
        total_schedules=total_schedules,
        parent_cooldown=parent_cooldown,
        crossover_probability=crossover_probability,
        max_steps_per_block=max_steps,
        max_total_steps=max_total_steps,
        plateau_patience=patience,
        plateau_tolerance=plateau_tolerance,
        random_seed=random_seed,
    )

    def register_candidate(
        times: tuple[float, ...],
        operator: str,
        stage: int,
        grid_resolution: int,
        parent_ids: tuple[int, ...] = (),
    ) -> ScheduleCandidate:
        key = _schedule_key(times, max_time)
        if key in seen_schedules:
            raise RuntimeError(f"Attempted to register duplicate schedule: {times}")
        candidate = ScheduleCandidate(
            schedule_id=len(candidates) + 1,
            times=times,
            operator=operator,
            stage=stage,
            grid_resolution=grid_resolution,
            parent_ids=parent_ids,
        )
        seen_schedules.add(key)
        candidates[candidate.schedule_id] = candidate
        log_queue_event(
            "SCHEDULE_PROPOSED",
            stage=stage,
            grid_resolution=grid_resolution,
            completed_schedules=completed_schedules,
            schedule=_queue_record(candidate, completed_schedules),
        )
        return candidate

    def fit_and_log(
        candidate: ScheduleCandidate,
        parameters: dict | None,
        *,
        search_stage: int,
        is_refit: bool,
    ) -> None:
        nonlocal completed_schedules, global_best
        parent_text = ",".join(map(str, candidate.parent_ids)) or "-"
        print(
            f"FIT_START schedule={candidate.schedule_id}/{total_schedules} "
            f"stage={search_stage} grid={candidate.grid_resolution} "
            f"operator={candidate.operator} parents={parent_text} "
            f"refit={str(is_refit).lower()} times={candidate.times}",
            flush=True,
        )
        started = time.perf_counter()
        previous_status = candidate.fit_status
        previous_steps = candidate.optimization_steps
        _fit_candidate(
            candidate,
            trees,
            num_states,
            free_edges,
            device,
            parameters,
            max_steps,
            patience,
            max_total_steps=max_total_steps,
            improvement_tolerance=plateau_tolerance,
            verbose=verbose,
        )
        elapsed = time.perf_counter() - started
        if not is_refit:
            completed_schedules += 1
        if candidate.fit_error is not None:
            print(
                f"FIT_WARNING schedule={candidate.schedule_id} "
                f"status={candidate.fit_status} error={candidate.fit_error}",
                flush=True,
            )
        block = {
            "schedule_id": candidate.schedule_id,
            "search_stage": search_stage,
            "schedule_stage": candidate.stage,
            "grid_resolution": candidate.grid_resolution,
            "refit": is_refit,
            "fit_block": candidate.fit_blocks,
            "steps_this_block": candidate.optimization_steps - previous_steps,
            "optimization_steps": candidate.optimization_steps,
            "steps_without_improvement": candidate.steps_without_improvement,
            "log_likelihood": candidate.score if math.isfinite(candidate.score) else None,
            "fit_status": candidate.fit_status,
            "freeze_reason": candidate.freeze_reason,
            "fit_error": candidate.fit_error,
            "fit_seconds": elapsed,
        }
        fit_history.append(block)
        print(
            f"FIT_DONE schedule={candidate.schedule_id}/{total_schedules} "
            f"status={candidate.fit_status} "
            f"log_likelihood={candidate.score:.8f} "
            f"steps={candidate.optimization_steps} "
            f"elapsed_seconds={elapsed:.1f}",
            flush=True,
        )
        log_queue_event(
            "FIT_RESULT",
            completed_schedules=completed_schedules,
            fit_seconds=elapsed,
            refit=is_refit,
            previous_status=previous_status,
            schedule=_queue_record(candidate, completed_schedules),
        )
        if candidate.fit_status == "frozen" and previous_status != "frozen":
            log_queue_event(
                "SCHEDULE_FROZEN",
                completed_schedules=completed_schedules,
                reason=candidate.freeze_reason,
                schedule=_queue_record(candidate, completed_schedules),
            )
        if math.isfinite(candidate.score) and (
            global_best is None or candidate.score > global_best.score
        ):
            global_best = candidate

    def select_parent(search_stage: int) -> ScheduleCandidate:
        parent, pool, cooldown_overridden = _pop_best_parent(
            active_queue,
            frozen_queue,
            candidates,
            completed_schedules,
        )
        if parent is None or pool is None:
            log_queue_event(
                "QUEUE_BLOCKED",
                stage=search_stage,
                reason="no_parent_available",
                completed_schedules=completed_schedules,
                queue_size=queue_size(),
                items=queue_snapshot_items(),
            )
            raise RuntimeError("No valid schedule is available to produce offspring")
        if cooldown_overridden:
            log_queue_event(
                "PARENT_COOLDOWN_OVERRIDDEN",
                stage=search_stage,
                completed_schedules=completed_schedules,
                reason="no_parent_ready_after_cooldown",
                schedule=_queue_record(parent, completed_schedules),
            )
        log_queue_event(
            "PARENT_SELECTED",
            stage=search_stage,
            queue_size=queue_size(),
            completed_schedules=completed_schedules,
            pool=pool,
            schedule=_queue_record(parent, completed_schedules),
        )
        return parent

    def fit_active_parent(candidate: ScheduleCandidate, search_stage: int) -> None:
        if candidate.fit_status == "active":
            fit_and_log(
                candidate,
                candidate.parameters,
                search_stage=search_stage,
                is_refit=True,
            )

    for stage_index, (grid_resolution, budget) in enumerate(
        zip(grid_resolutions, stage_budgets),
        start=1,
    ):
        print(
            f"STAGE_START stage={stage_index}/{len(stage_budgets)} "
            f"grid={grid_resolution} new_schedules={budget}",
            flush=True,
        )
        log_queue_event(
            "STAGE_STARTED",
            stage=stage_index,
            grid_resolution=grid_resolution,
            new_schedule_budget=budget,
            completed_schedules=completed_schedules,
        )

        if stage_index == 1:
            initial_candidates = [
                register_candidate(
                    incumbent,
                    operator="initial_incumbent",
                    stage=stage_index,
                    grid_resolution=grid_resolution,
                )
            ]
            attempts = 0
            while len(initial_candidates) < budget:
                attempts += 1
                if attempts > budget * 1000:
                    raise RuntimeError(
                        "Could not generate enough unique initial schedules "
                        "on the stage grid"
                    )
                times = _random_schedule(
                    rng, num_times, max_time, min_gap, grid_resolution
                )
                if _schedule_key(times, max_time) in seen_schedules:
                    continue
                initial_candidates.append(
                    register_candidate(
                        times,
                        operator="initial_random",
                        stage=stage_index,
                        grid_resolution=grid_resolution,
                    )
                )
            for candidate in initial_candidates:
                fit_and_log(
                    candidate,
                    initial_parameters,
                    search_stage=stage_index,
                    is_refit=False,
                )
                enqueue_candidate(candidate)
                refresh_waiting_schedules()
        else:
            attempts = 0
            new_schedules = 0
            while new_schedules < budget:
                completed_before = completed_schedules
                parent = select_parent(stage_index)
                operator = (
                    "crossover"
                    if rng.random() < crossover_probability
                    else "mutation"
                )
                second_parent = None
                fallback = ""
                if operator == "crossover":
                    second_parent = select_parent(stage_index)
                    candidate_times = _unique_crossover_schedule(
                        parent.times,
                        second_parent.times,
                        rng,
                        max_time,
                        min_gap,
                        seen_schedules,
                        grid_resolution,
                    )
                    if candidate_times is None:
                        log_queue_event(
                            "CROSSOVER_EXHAUSTED",
                            stage=stage_index,
                            completed_schedules=completed_schedules,
                            first_parent=_queue_record(parent, completed_schedules),
                            second_parent=_queue_record(second_parent, completed_schedules),
                        )
                        enqueue_candidate(second_parent)
                        log_queue_event(
                            "PARENT_REQUEUED",
                            stage=stage_index,
                            reason="crossover_space_exhausted",
                            completed_schedules=completed_schedules,
                            queue_size=queue_size(),
                            schedule=_queue_record(second_parent, completed_schedules),
                        )
                        second_parent = None
                        operator = "mutation"
                        fallback = "crossover_space_exhausted"
                    else:
                        times = candidate_times
                parents = [parent] + ([second_parent] if second_parent is not None else [])
                duplicate_attempts = 0
                if operator == "mutation":
                    times = None
                    while times is None:
                        progress = completed_before / total_schedules
                        mutation_scale = max(
                            max_time / grid_resolution,
                            max_time * (0.15 - 0.135 * progress),
                        )
                        proposed = _mutate_schedule(
                            parent.times,
                            rng,
                            max_time,
                            min_gap,
                            mutation_scale,
                            grid_resolution,
                        )
                        if _schedule_key(proposed, max_time) in seen_schedules:
                            duplicate_attempts += 1
                            if duplicate_attempts >= 1000:
                                raise RuntimeError(
                                    f"Could not generate a unique mutated schedule "
                                    f"from parent {parent.schedule_id} after 1000 attempts"
                                )
                        else:
                            times = proposed
                if duplicate_attempts:
                    log_queue_event(
                        "MUTATION_DUPLICATES_SKIPPED",
                        stage=stage_index,
                        parent_id=parent.schedule_id,
                        attempts=duplicate_attempts,
                    )

                for selected_parent in parents:
                    fit_active_parent(selected_parent, stage_index)
                warm_start_parent = max(
                    parents,
                    key=lambda candidate: (candidate.score, -candidate.schedule_id),
                )
                if warm_start_parent.parameters is None:
                    raise RuntimeError(
                        f"Schedule {warm_start_parent.schedule_id} has no fitted parameters"
                    )
                child = register_candidate(
                    times,
                    operator=operator,
                    stage=stage_index,
                    grid_resolution=grid_resolution,
                    parent_ids=tuple(item.schedule_id for item in parents),
                )
                child.eligible_after = completed_before + 1
                for selected_parent in parents:
                    selected_parent.eligible_after = completed_before + parent_cooldown
                print(
                    f"OFFSPRING schedule={child.schedule_id}/{total_schedules} "
                    f"stage={stage_index} grid={grid_resolution} "
                    f"operator={operator} "
                    f"parents={','.join(str(item.schedule_id) for item in parents)}"
                    f"{f' fallback={fallback}' if fallback else ''}",
                    flush=True,
                )
                fit_and_log(
                    child,
                    warm_start_parent.parameters,
                    search_stage=stage_index,
                    is_refit=False,
                )
                enqueue_candidate(child)
                for selected_parent in parents:
                    enqueue_candidate(selected_parent)
                    parent_record = _queue_record(
                        selected_parent, completed_schedules
                    )
                    if parent_record["queue_state"] == "waiting":
                        waiting_schedule_ids.add(selected_parent.schedule_id)
                    log_queue_event(
                        "PARENT_REQUEUED",
                        stage=stage_index,
                        completed_schedules=completed_schedules,
                        queue_size=queue_size(),
                        schedule=parent_record,
                    )
                refresh_waiting_schedules()
                new_schedules += 1
                attempts += 1
                if attempts > budget * 2000:
                    raise RuntimeError(
                        f"Could not generate {budget} unique schedules "
                        f"at grid resolution {grid_resolution}"
                    )
                if completed_schedules % 10 == 0 or new_schedules == budget:
                    print(
                        f"PROGRESS stage={stage_index} "
                        f"stage_fitted={new_schedules}/{budget} "
                        f"total_fitted={completed_schedules}/{total_schedules} "
                        f"active={len(active_queue)} frozen={len(frozen_queue)} "
                        f"best_schedule={global_best.schedule_id if global_best else 'none'} "
                        f"best_log_likelihood="
                        f"{global_best.score if global_best else float('-inf'):.8f}",
                        flush=True,
                    )

        if global_best is None:
            log_queue_event(
                "QUEUE_BLOCKED",
                stage=stage_index,
                reason="all_schedules_invalid",
                completed_schedules=completed_schedules,
                queue_size=queue_size(),
                items=queue_snapshot_items(),
            )
            raise RuntimeError("All fitted schedules produced non-finite likelihoods")
        log_queue_event(
            "QUEUE_SNAPSHOT",
            reason=f"stage_{stage_index}_completed",
            stage=stage_index,
            grid_resolution=grid_resolution,
            completed_schedules=completed_schedules,
            queue_size=queue_size(),
            active_count=len(active_queue),
            frozen_count=len(frozen_queue),
            invalid_count=sum(
                candidate.fit_status == "invalid" for candidate in candidates.values()
            ),
            items=queue_snapshot_items(),
        )
        print(
            f"STAGE_DONE stage={stage_index}/{len(stage_budgets)} "
            f"fitted={completed_schedules}/{total_schedules} "
            f"active={len(active_queue)} frozen={len(frozen_queue)} "
            f"best_schedule={global_best.schedule_id} "
            f"best_log_likelihood={global_best.score:.8f} "
            f"best_times={global_best.times}",
            flush=True,
        )

    if completed_schedules != total_schedules:
        raise RuntimeError(
            f"Expected {total_schedules} unique fitted schedules; "
            f"completed {completed_schedules}"
        )
    if global_best is None:
        raise RuntimeError("Population search completed without a finite schedule")

    log_queue_event(
        "QUEUE_SNAPSHOT",
        reason="search_completed",
        completed_schedules=completed_schedules,
        queue_size=queue_size(),
        active_count=len(active_queue),
        frozen_count=len(frozen_queue),
        invalid_count=sum(
            candidate.fit_status == "invalid" for candidate in candidates.values()
        ),
        items=queue_snapshot_items(),
    )
    log_queue_event(
        "SEARCH_COMPLETED",
        completed_schedules=completed_schedules,
        best_schedule_id=global_best.schedule_id,
        best_log_likelihood=global_best.score,
        best_transition_times=list(global_best.times),
        active_count=len(active_queue),
        frozen_count=len(frozen_queue),
        invalid_count=sum(
            candidate.fit_status == "invalid" for candidate in candidates.values()
        ),
    )
    return {
        "initial_log_likelihood": (
            candidates[1].score if math.isfinite(candidates[1].score) else None
        ),
        "initial_transition_times": list(incumbent),
        "best_log_likelihood": global_best.score,
        "best_transition_times": list(global_best.times),
        "best_parameters": global_best.parameters,
        "unique_schedules_fitted": completed_schedules,
        "active_schedule_ids": sorted(
            candidate.schedule_id
            for candidate in candidates.values()
            if candidate.fit_status == "active"
        ),
        "frozen_schedule_ids": sorted(frozen_archive),
        "invalid_schedule_ids": sorted(
            candidate.schedule_id
            for candidate in candidates.values()
            if candidate.fit_status == "invalid"
        ),
        "queue_log_path": str(queue_log_path) if queue_log_path is not None else None,
        "search_config": {
            "grid_resolutions": list(grid_resolutions),
            "stage_budgets": list(stage_budgets),
            "total_schedules": total_schedules,
            "parent_cooldown": parent_cooldown,
            "crossover_probability": crossover_probability,
            "max_steps_per_block": max_steps,
            "max_total_steps_per_schedule": max_total_steps,
            "plateau_patience": patience,
            "plateau_tolerance": plateau_tolerance,
            "random_seed": random_seed,
        },
        "schedule_history": [
            {
                "schedule_id": candidate.schedule_id,
                "operator": candidate.operator,
                "stage": candidate.stage,
                "grid_resolution": candidate.grid_resolution,
                "parent_ids": list(candidate.parent_ids),
                "transition_times": list(candidate.times),
                "log_likelihood": candidate.score if math.isfinite(candidate.score) else None,
                "optimization_steps": candidate.optimization_steps,
                "fit_blocks": candidate.fit_blocks,
                "steps_without_improvement": candidate.steps_without_improvement,
                "fit_status": candidate.fit_status,
                "freeze_reason": candidate.freeze_reason,
                "fit_error": candidate.fit_error,
            }
            for candidate in candidates.values()
        ],
        "fit_history": fit_history,
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimental population search for joint transition times."
    )
    parser.add_argument("--tree-directory", type=Path, required=True)
    parser.add_argument("--num-trees", type=int, required=True)
    parser.add_argument("--initial-output", type=Path, required=True)
    parser.add_argument(
        "--transition-edges-file",
        type=Path,
        default=Path("src/analyses/simulation/config/transition_edges_star.txt"),
    )
    parser.add_argument("--truth-times-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--queue-log", type=Path, default=None)
    parser.add_argument(
        "--grid-resolutions",
        nargs="+",
        type=int,
        default=[25, 100, 250, 1000],
    )
    parser.add_argument(
        "--stage-budgets",
        nargs="+",
        type=int,
        default=[100, 200, 300, 400],
    )
    parser.add_argument("--parent-cooldown", type=int, default=5)
    parser.add_argument("--crossover-probability", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-total-steps", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--plateau-tolerance", type=float, default=1e-4)
    parser.add_argument("--random-seed", type=int, default=20261028)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed cut-tree likelihood data structures.",
    )
    args = parser.parse_args()

    if not args.tree_directory.is_dir():
        raise FileNotFoundError(args.tree_directory)
    if not args.transition_edges_file.is_file():
        raise FileNotFoundError(args.transition_edges_file)
    if not args.initial_output.is_file():
        raise FileNotFoundError(args.initial_output)

    initial_times, initial_parameters = _parse_initial_output(
        args.initial_output,
        num_states=4,
    )
    trees = _load_trees(str(args.tree_directory), args.num_trees)
    if len(trees) != args.num_trees:
        raise ValueError(
            f"Expected {args.num_trees} trees in {args.tree_directory}; "
            f"found {len(trees)}"
        )
    free_edges = read_free_edges(str(args.transition_edges_file))
    torch.set_num_threads(1)
    queue_log_path = args.queue_log
    if queue_log_path is None and args.output is not None:
        queue_log_path = args.output.with_name(
            f"{args.output.stem}.queue.jsonl"
        )

    result = population_search(
        trees=trees,
        num_states=4,
        free_edges=free_edges,
        initial_times=initial_times,
        initial_parameters=initial_parameters,
        grid_resolutions=tuple(args.grid_resolutions),
        stage_budgets=tuple(args.stage_budgets),
        parent_cooldown=args.parent_cooldown,
        crossover_probability=args.crossover_probability,
        max_steps=args.max_steps,
        max_total_steps=args.max_total_steps,
        patience=args.patience,
        plateau_tolerance=args.plateau_tolerance,
        random_seed=args.random_seed,
        device=torch.device("cpu"),
        verbose=args.verbose,
        queue_log_path=queue_log_path,
    )
    if args.truth_times_file is not None:
        truth_times = read_transition_times(str(args.truth_times_file))
        estimated_times = result["best_transition_times"]
        if len(truth_times) != len(estimated_times):
            raise ValueError(
                f"Expected {len(estimated_times)} truth times in "
                f"{args.truth_times_file}; found {len(truth_times)}"
            )
        absolute_errors = [
            abs(estimate - truth)
            for estimate, truth in zip(estimated_times, truth_times)
        ]
        result["truth_times_for_evaluation_only"] = truth_times
        result["absolute_time_errors"] = absolute_errors
        result["mean_absolute_time_error"] = (
            sum(absolute_errors) / len(absolute_errors)
        )
    rendered = json.dumps(result, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(f"Wrote population-search results to {args.output}")


if __name__ == "__main__":
    main()
