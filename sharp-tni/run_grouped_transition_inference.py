#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.results.kptracer.grouped_transition_schedules import (
    load_grouped_transition_schedules,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit CARTA parameters for one grouped transition schedule."
    )
    parser.add_argument("--schedule-id", required=True)
    parser.add_argument("--schedule-tables", type=Path, nargs="+", required=True)
    parser.add_argument("--transition-edges-file", type=Path, required=True)
    parser.add_argument("--tree-directory", type=Path, required=True)
    parser.add_argument("--initial-parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--sampling-probability", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--patience", type=int, default=500)
    args = parser.parse_args()

    schedules = load_grouped_transition_schedules(args.schedule_tables)
    try:
        schedule = schedules[args.schedule_id]
    except KeyError as error:
        raise ValueError(
            f"Schedule {args.schedule_id!r} was not found in "
            f"{args.schedule_tables}"
        ) from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="carta-grouped-schedule-") as temp_dir:
        temp_root = Path(temp_dir)
        cache_dir = temp_root / "inductor-cache"
        tmp_dir = temp_root / "tmp"
        cache_dir.mkdir()
        tmp_dir.mkdir()
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
        os.environ["TMPDIR"] = str(tmp_dir)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "1"

        transition_times_path = temp_root / "transition_times.txt"
        transition_times_path.write_text(
            "".join(f"{time:.17g}\n" for time in schedule.transition_times)
        )

        import torch._inductor.config as inductor_config

        inductor_config.cpp_cache_precompile_headers = False
        sys.argv = [
            "src.main2",
            str(args.tree_directory),
            "--num-states",
            "5",
            "--num-trees",
            "1",
            "--transition-edges-file",
            str(args.transition_edges_file),
            "--transition-times-file",
            str(transition_times_path),
            "--initial-parameters",
            str(args.initial_parameters),
            "--fixed-shared-sampling-probability",
            str(args.sampling_probability),
            "--max-steps",
            str(args.max_steps),
            "--patience",
            str(args.patience),
            "--output",
            str(args.output),
            "--log",
            str(args.log),
            "--device",
            "cpu",
            "--optimizer",
            "adam",
            "--normalize-root-likelihood",
        ]
        runpy.run_module("src.main2", run_name="__main__")


if __name__ == "__main__":
    main()
