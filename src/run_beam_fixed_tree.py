from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from src.generate_beam_barcodes import TreeData, read_tree

STATE_NAMES = {0: "A", 1: "B", 2: "C", 3: "D"}
SUMMARY_COLUMNS = (
    "tree_id",
    "parameter",
    "mean",
    "median",
    "HPD95_lower",
    "HPD95_upper",
    "samples_after_burnin",
)
STATUS_COLUMNS = (
    "tree_id",
    "status",
    "tip_count",
    "observed_tissues",
    "message",
)
NON_PARAMETER_COLUMNS = {
    "Sample",
    "posterior",
    "likelihood",
    "prior",
    "treeLikelihood",
    "tissueLikelihood",
}


def _quote_newick_label(label: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        return label
    return "'" + label.replace("'", "''") + "'"


def render_fixed_tree(tree: TreeData) -> tuple[str, float]:
    if len(tree.tips) < 2:
        raise ValueError("BEAM fixed-tree inference requires at least two tips")

    origin_time = tree.node_ages[tree.root_id] + 1e-4
    tree_root = tree.root_id
    while len(tree.children[tree_root]) == 1:
        tree_root = tree.children[tree_root][0][0]

    def render_subtree(node_id: str) -> str:
        children = tree.children[node_id]
        if not children:
            return _quote_newick_label(node_id)
        descendants = [
            f"{render_subtree(child_id)}:{branch_length:.17g}"
            for child_id, branch_length in children
        ]
        return f"({','.join(descendants)})"

    return f"{render_subtree(tree_root)};", origin_time


def read_barcode_taxa(barcode_path: Path) -> tuple[str, ...]:
    with barcode_path.open(newline="") as input_file:
        reader = csv.reader(input_file, delimiter="\t")
        header = next(reader, None)
        if not header or header[0] != "":
            raise ValueError(f"{barcode_path} must have an empty first header field")
        taxa = tuple(row[0] for row in reader if row)
    if len(taxa) != len(set(taxa)):
        raise ValueError(f"{barcode_path} contains duplicate taxon IDs")
    return taxa


def write_tissue_labels(
    nodes_path: Path,
    taxa: tuple[str, ...],
    output_path: Path,
) -> tuple[str, ...]:
    tip_states: dict[str, int] = {}
    with nodes_path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            if (row["is_leaf"] or "").strip().lower() not in {"1", "true"}:
                continue
            tip_id = (row["node_id"] or "").strip()
            try:
                state_id = int(row["state_id"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Tip {tip_id!r} in {nodes_path} has an invalid state_id"
                ) from error
            if state_id not in STATE_NAMES:
                raise ValueError(
                    f"Tip {tip_id!r} in {nodes_path} has unsupported state_id "
                    f"{state_id}; expected 0 through 3"
                )
            tip_states[tip_id] = state_id

    if set(taxa) != set(tip_states):
        missing_labels = sorted(set(taxa) - set(tip_states))
        extra_labels = sorted(set(tip_states) - set(taxa))
        raise ValueError(
            "Barcode taxa and tree tips differ; "
            f"missing tissue labels={missing_labels[:5]}, "
            f"extra tree tips={extra_labels[:5]}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerows(
            (taxon, STATE_NAMES[tip_states[taxon]]) for taxon in taxa
        )
    return tuple(sorted({STATE_NAMES[tip_states[taxon]] for taxon in taxa}))


def write_beam_xml(
    template_path: Path,
    output_path: Path,
    origin_time: float,
    chain_length: int,
) -> None:
    if chain_length < 1:
        raise ValueError("chain_length must be positive")
    template = template_path.read_text()
    replacements = {
        "@@ORIGIN_TIME@@": f"{origin_time:.17g}",
        "@@CHAIN_LENGTH@@": str(chain_length),
    }
    for token, value in replacements.items():
        if token not in template:
            raise ValueError(f"BEAM XML template is missing placeholder {token}")
        template = template.replace(token, value)
    if "@@" in template:
        raise ValueError("BEAM XML template contains an unresolved placeholder")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template)


def _hpd_interval(values: np.ndarray, mass: float = 0.95) -> tuple[float, float]:
    ordered = np.sort(values)
    count = max(1, int(np.ceil(mass * len(ordered))))
    if count >= len(ordered):
        return float(ordered[0]), float(ordered[-1])
    widths = ordered[count - 1 :] - ordered[: len(ordered) - count + 1]
    start = int(np.argmin(widths))
    return float(ordered[start]), float(ordered[start + count - 1])


def summarize_result_log(
    log_path: Path,
    summary_path: Path,
    tree_id: str,
    burnin: float,
) -> None:
    if not 0 <= burnin < 1:
        raise ValueError("burnin must be in [0, 1)")
    with log_path.open() as input_file:
        rows = list(
            csv.DictReader(
                (line for line in input_file if line.strip() and not line.startswith("#")),
                delimiter="\t",
            )
        )
    if not rows:
        raise ValueError(f"{log_path} contains no parameter samples")
    rows = rows[int(len(rows) * burnin) :]
    parameter_names = [
        name
        for name in rows[0]
        if name not in NON_PARAMETER_COLUMNS
    ]
    output_path = summary_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=SUMMARY_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for parameter in parameter_names:
            values = np.asarray(
                [float(row[parameter]) for row in rows],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{log_path} contains non-finite samples for {parameter}"
                )
            lower, upper = _hpd_interval(values)
            writer.writerow(
                {
                    "tree_id": tree_id,
                    "parameter": parameter,
                    "mean": f"{float(np.mean(values)):.12g}",
                    "median": f"{float(np.median(values)):.12g}",
                    "HPD95_lower": f"{lower:.12g}",
                    "HPD95_upper": f"{upper:.12g}",
                    "samples_after_burnin": len(values),
                }
            )


def write_status(
    status_path: Path,
    tree_id: str,
    status: str,
    tip_count: int,
    observed_tissues: tuple[str, ...],
    message: str,
) -> None:
    with status_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=STATUS_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "tree_id": tree_id,
                "status": status,
                "tip_count": tip_count,
                "observed_tissues": ",".join(observed_tissues),
                "message": message,
            }
        )


def run_inference(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    beast_log = args.beast_log.resolve()
    beast_log.parent.mkdir(parents=True, exist_ok=True)
    tree = read_tree(args.nodes, args.edges)
    status_path = output_dir / "run_status.tsv"
    if len(tree.tips) < 2:
        beast_log.write_text("Skipped: BEAM fixed-tree inference requires at least two tips.\n")
        write_status(
            status_path,
            args.tree_id,
            "skipped_single_tip",
            len(tree.tips),
            (),
            "BEAM fixed-tree inference requires at least two tips.",
        )
        return

    taxa = read_barcode_taxa(args.barcodes)
    if set(taxa) != set(tree.tips):
        raise ValueError(
            f"Barcode taxa in {args.barcodes} do not match tips in {args.nodes}"
        )
    observed_tissues = write_tissue_labels(
        args.nodes,
        taxa,
        output_dir / "tissues.csv",
    )
    starting_tree, origin_time = render_fixed_tree(tree)
    (output_dir / "startingTree.nwk").write_text(starting_tree + "\n")
    write_beam_xml(
        args.template,
        output_dir / "input.xml",
        origin_time,
        args.chain_length,
    )

    if not Path(args.beast_executable).is_file() and not shutil.which(
        args.beast_executable
    ):
        raise FileNotFoundError(
            f"BEAST executable {args.beast_executable!r} was not found. "
            "Set BEAST_EXECUTABLE or pass --config "
            "star3_beam_beast_executable=/path/to/beast."
        )
    with beast_log.open("w") as log_file:
        subprocess.run(
            [
                args.beast_executable,
                "-seed",
                str(args.seed),
                "-overwrite",
                "input.xml",
            ],
            cwd=output_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )

    result_log = output_dir / "result.log"
    result_trees = output_dir / "result.trees"
    if not result_log.is_file() or not result_trees.is_file():
        raise FileNotFoundError(
            "BEAST exited successfully but did not create result.log and result.trees"
        )
    summarize_result_log(
        result_log,
        output_dir / "parameter_summary.tsv",
        args.tree_id,
        args.burnin,
    )
    write_status(
        status_path,
        args.tree_id,
        "completed",
        len(tree.tips),
        observed_tissues,
        "",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BEAM on one fixed star_3 tree and summarize its parameters."
    )
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--barcodes", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--beast-log", type=Path, required=True)
    parser.add_argument("--beast-executable", required=True)
    parser.add_argument("--tree-id", required=True)
    parser.add_argument("--chain-length", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--burnin", type=float, default=0.1)
    run_inference(parser.parse_args())


if __name__ == "__main__":
    main()
