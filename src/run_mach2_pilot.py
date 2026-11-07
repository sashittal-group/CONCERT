from __future__ import annotations

from pathlib import Path
import sys

from src.run_mach2_simulated_tree import run_mach2_simulated_tree


def _is_complete(output_dir: Path) -> bool:
    log_path = output_dir / "mach2.log"
    if not log_path.is_file() or "Exit code: 0" not in log_path.read_text():
        return False
    refined_trees = list(
        (output_dir / "mach2_output").glob("*-T-0.refined.tree")
    )
    if len(refined_trees) != 1:
        return False
    prefix = str(refined_trees[0])[: -len(".refined.tree")]
    return all(
        Path(f"{prefix}{suffix}").is_file()
        for suffix in (".nodes", ".location.labeling")
    )


PILOT_TREE_DIRS = (
    "star_3/setting_85/batch_9/tree_6",
    "star_3/setting_25/batch_5/tree_2",
    "star_3/setting_48/batch_8/tree_2",
    "linear_A_to_D/setting_75/batch_5/tree_6",
    "linear_A_to_D/setting_92/batch_9/tree_6",
    "linear_A_to_D/setting_45/batch_3/tree_0",
)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    succeeded = 0
    failed = 0
    for tree_dir in PILOT_TREE_DIRS:
        tree_root = project_root / "data" / "simulation" / "coord_diff" / tree_dir
        output_dir = tree_root / "mach2"
        if _is_complete(output_dir):
            succeeded += 1
            print(f"SKIPPED completed {tree_dir}", flush=True)
            continue
        try:
            output_dir = run_mach2_simulated_tree(
                tree_root / "nodes.csv",
                tree_root / "edges.csv",
                tree_root / "mach2",
                threads=1,
            )
        except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
            failed += 1
            print(f"FAILED {tree_dir}: {error}", file=sys.stderr, flush=True)
        else:
            succeeded += 1
            print(f"COMPLETED {tree_dir}: {output_dir}", flush=True)

    print(f"MACH2 pilot: {succeeded} completed, {failed} failed", flush=True)
    return 0 if succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
