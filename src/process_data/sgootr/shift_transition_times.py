from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT / "data" / "sgootr" / "processed" / "upgma_min_subtracted"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "sgootr"
    / "processed"
    / "upgma_min_subtracted_eps_1e-4"
)
DEFAULT_EPSILON = 1e-4


def shift_transition_time_files(
    input_csv: Path,
    input_text: Path,
    output_dir: Path,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[Path, Path]:
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and nonnegative.")

    csv_output = output_dir / "transition_times.csv"
    text_output = output_dir / "transition_times.txt"
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "transition_time" not in reader.fieldnames:
            raise ValueError(
                f"Transition CSV must have a 'transition_time' column: {input_csv}"
            )
        rows = list(reader)
        for row in rows:
            transition_time = _parse_time(row["transition_time"], input_csv)
            shifted_time = transition_time - epsilon
            if shifted_time < 0.0:
                raise ValueError(
                    f"Subtracting epsilon would make a transition time negative "
                    f"in {input_csv}: {transition_time} - {epsilon}."
                )
            row["transition_time"] = f"{shifted_time:.15g}"

        with csv_output.open("w", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    text_lines = input_text.read_text().splitlines()
    shifted_lines = []
    for line_number, line in enumerate(text_lines, start=1):
        if not line.strip():
            shifted_lines.append(line)
            continue
        transition_time = _parse_time(line, input_text, line_number)
        shifted_time = transition_time - epsilon
        if shifted_time < 0.0:
            raise ValueError(
                f"Subtracting epsilon would make a transition time negative "
                f"in {input_text}:{line_number}: {transition_time} - {epsilon}."
            )
        shifted_lines.append(f"{shifted_time:.15g}")

    text_output.write_text(
        "\n".join(shifted_lines) + ("\n" if text_lines else "")
    )
    return csv_output, text_output


def _parse_time(
    value: str | None,
    path: Path,
    line_number: int | None = None,
) -> float:
    location = f"{path}:{line_number}" if line_number is not None else str(path)
    try:
        transition_time = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid transition time in {location}: {value!r}.") from exc
    if not math.isfinite(transition_time):
        raise ValueError(f"Non-finite transition time in {location}.")
    return transition_time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subtract an epsilon from per-edge and global transition times."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_DIR / "transition_times.csv",
    )
    parser.add_argument(
        "--input-text",
        type=Path,
        default=DEFAULT_INPUT_DIR / "transition_times.txt",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    outputs = shift_transition_time_files(
        arguments.input_csv,
        arguments.input_text,
        arguments.output_dir,
        arguments.epsilon,
    )
    print(f"Wrote shifted per-edge transition times to {outputs[0]}.")
    print(f"Wrote shifted global transition times to {outputs[1]}.")


if __name__ == "__main__":
    main()
