from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import sys


PILOT_TREE_COUNT = 4
CHAIN_LENGTH = 10_000
BEAST_EXECUTABLE = os.environ.get(
    "BEAST_EXECUTABLE",
    "/home/saminrk/.local/opt/beast2-2.7.7/bin/beast",
)


def _pilot_candidates(data_root: Path) -> list[Path]:
    candidates = []
    for nodes_path in data_root.glob("setting_*/batch_*/tree_[0-7]/nodes.csv"):
        tree_dir = nodes_path.parent
        barcode_path = tree_dir / "barcodes_100_sites.tsv"
        if (
            not barcode_path.is_file()
            or not (tree_dir / "edges.csv").is_file()
            or (tree_dir / "beam_snakemake").exists()
        ):
            continue
        with nodes_path.open(newline="") as source:
            leaves = [
                row
                for row in csv.DictReader(source)
                if row["is_leaf"].strip().lower() in {"1", "true"}
            ]
        if len(leaves) >= 3 and len(
            {row["state_id"] for row in leaves}
        ) >= 2:
            candidates.append((nodes_path.stat().st_size, nodes_path))
    return [
        nodes_path
        for _, nodes_path in sorted(
            candidates,
            key=lambda item: (item[0], str(item[1])),
        )[:PILOT_TREE_COUNT]
    ]


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "simulation" / "coord_diff" / "linear_A_to_D"
    beast_path = Path(BEAST_EXECUTABLE)
    if not beast_path.is_file():
        print(
            f"BEAST executable not found: {BEAST_EXECUTABLE}. "
            "Set BEAST_EXECUTABLE to its path.",
            file=sys.stderr,
        )
        return 1

    candidates = _pilot_candidates(data_root)
    if len(candidates) < PILOT_TREE_COUNT:
        print(
            f"Found only {len(candidates)} eligible linear trees; "
            f"need {PILOT_TREE_COUNT}. Existing barcodes are required.",
            file=sys.stderr,
        )
        return 1

    succeeded = 0
    failed = 0
    for index, nodes_path in enumerate(candidates, start=1):
        tree_dir = nodes_path.parent
        relative_tree = tree_dir.relative_to(data_root)
        output_dir = tree_dir / "beam_snakemake"
        command = [
            sys.executable,
            "-m",
            "src.run_beam_fixed_tree",
            "--nodes",
            str(nodes_path),
            "--edges",
            str(tree_dir / "edges.csv"),
            "--barcodes",
            str(tree_dir / "barcodes_100_sites.tsv"),
            "--template",
            str(project_root / "src" / "beam_fixed_tree_template.xml"),
            "--output-dir",
            str(output_dir),
            "--beast-log",
            str(tree_dir / "beam_snakemake.log"),
            "--beast-executable",
            str(beast_path),
            "--chain-length",
            str(CHAIN_LENGTH),
            "--burnin",
            "0.1",
            "--seed",
            str(42_000 + index),
            "--tree-id",
            "/".join(relative_tree.parts),
        ]
        try:
            subprocess.run(command, cwd=project_root, check=True)
        except subprocess.CalledProcessError as error:
            failed += 1
            print(
                f"FAILED linear_A_to_D/{relative_tree}: exit {error.returncode}",
                file=sys.stderr,
                flush=True,
            )
        else:
            succeeded += 1
            print(f"COMPLETED linear_A_to_D/{relative_tree}", flush=True)

    print(f"Linear BEAM pilot: {succeeded} completed, {failed} failed", flush=True)
    return 0 if succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
