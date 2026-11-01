from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.run_beam_fixed_tree import STATUS_COLUMNS, SUMMARY_COLUMNS


def combine_summaries(
    input_root: Path,
    parameter_output: Path,
    status_output: Path,
) -> None:
    run_directories = sorted(
        input_root.glob("setting_*/batch_*/tree_*/beam_snakemake")
    )
    if not run_directories:
        raise FileNotFoundError(f"No BEAM run directories found under {input_root}")

    parameter_output.parent.mkdir(parents=True, exist_ok=True)
    status_output.parent.mkdir(parents=True, exist_ok=True)
    with (
        parameter_output.open("w", newline="") as parameter_file,
        status_output.open("w", newline="") as status_file,
    ):
        parameter_writer = csv.DictWriter(
            parameter_file,
            fieldnames=SUMMARY_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        status_writer = csv.DictWriter(
            status_file,
            fieldnames=STATUS_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        parameter_writer.writeheader()
        status_writer.writeheader()

        for run_directory in run_directories:
            status_path = run_directory / "run_status.tsv"
            if not status_path.is_file():
                raise FileNotFoundError(f"Missing BEAM run status: {status_path}")
            with status_path.open(newline="") as input_file:
                for row in csv.DictReader(input_file, delimiter="\t"):
                    status_writer.writerow(row)

            summary_path = run_directory / "parameter_summary.tsv"
            if not summary_path.is_file():
                continue
            with summary_path.open(newline="") as input_file:
                reader = csv.DictReader(input_file, delimiter="\t")
                if tuple(reader.fieldnames or ()) != SUMMARY_COLUMNS:
                    raise ValueError(f"Unexpected summary columns in {summary_path}")
                parameter_writer.writerows(reader)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine per-tree BEAM parameter and status summaries."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--parameter-output", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    args = parser.parse_args()
    combine_summaries(
        args.input_root,
        args.parameter_output,
        args.status_output,
    )


if __name__ == "__main__":
    main()
