from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

from Bio import Phylo
from Bio.Phylo.BaseTree import Tree

from src.process_data.sgootr.executables import resolve_executable
from src.process_data.sgootr.tree import load_tree


def date_tree_with_lsd2(
    input_tree_path: Path,
    output_tree_path: Path,
    project_root: Path,
    *,
    executable: str | None = None,
    sequence_length: int = 1000,
    minimum_branch_length: float = 0.001,
    rounding_factor: int = 100,
    ultrametric_tolerance: float = 1e-5,
) -> Tree:
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if not math.isfinite(minimum_branch_length) or minimum_branch_length <= 0:
        raise ValueError("minimum_branch_length must be finite and positive")
    if rounding_factor < 1:
        raise ValueError("rounding_factor must be positive")

    input_tree = load_tree(input_tree_path)
    executable_path = resolve_executable("lsd2", executable, project_root)
    output_tree_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sgootr-lsd2-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        output_prefix = temporary_path / "dated"
        command = [
            str(executable_path),
            "-i",
            str(input_tree_path.resolve()),
            "-o",
            str(output_prefix),
            "-s",
            str(sequence_length),
            "-r",
            "a",
            "-l",
            "-1",
            "-u",
            str(minimum_branch_length),
            "-U",
            str(minimum_branch_length),
            "-R",
            str(rounding_factor),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        log_path = output_tree_path.parent / "lsd2.log"
        log_path.write_text(
            f"$ {' '.join(command)}\n\n"
            f"Exit code: {result.returncode}\n\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"LSD2 failed with exit code {result.returncode}; see {log_path}."
            )

        dated_nexus_path = Path(f"{output_prefix}.date.nexus")
        if not dated_nexus_path.is_file():
            raise FileNotFoundError(
                f"LSD2 did not create the expected tree: {dated_nexus_path}"
            )
        dated_tree = Phylo.read(dated_nexus_path, "nexus")

    input_tips = {tip.name for tip in input_tree.get_terminals()}
    dated_tips = {tip.name for tip in dated_tree.get_terminals()}
    if input_tips != dated_tips:
        raise ValueError("LSD2 output tip labels do not match the input tree.")

    dated_tree.root.branch_length = None
    depths = dated_tree.depths()
    tips = [depths[tip] for tip in dated_tree.get_terminals()]
    tip_depth_range = max(tips) - min(tips)
    if tip_depth_range > ultrametric_tolerance:
        raise ValueError(
            f"LSD2 output is not ultrametric: root-to-tip range "
            f"{tip_depth_range:.3g} exceeds {ultrametric_tolerance:g}."
        )

    output_tree_path.parent.mkdir(parents=True, exist_ok=True)
    Phylo.write(
        dated_tree,
        output_tree_path,
        "newick",
        format_branch_length="%1.10f",
    )
    return dated_tree
