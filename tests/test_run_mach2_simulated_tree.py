from __future__ import annotations

import csv
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.run_mach2_simulated_tree import (
    _read_tree_inputs,
    run_mach2_simulated_tree,
)


class RunMach2SimulatedTreeTests(unittest.TestCase):
    def test_reads_simulated_tree_as_one_based_mach2_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nodes_path = root / "nodes.csv"
            edges_path = root / "edges.csv"
            with nodes_path.open("w", newline="") as source:
                writer = csv.writer(source)
                writer.writerow(("node_id", "T", "is_leaf", "state_id"))
                writer.writerows(
                    ((0, 0.0, 0, -1), (1, 1.0, 1, 1), (2, 1.0, 1, 3))
                )
            with edges_path.open("w", newline="") as source:
                writer = csv.writer(source)
                writer.writerow(("parent", "child", "branch_length"))
                writer.writerows(((0, 1, 1.0), (0, 2, 1.0)))

            edges, labels = _read_tree_inputs(nodes_path, edges_path)

        self.assertEqual(edges, [(1, 2), (1, 3)])
        self.assertEqual(labels, [(2, 1), (3, 3)])

    def test_runs_mach2_and_keeps_first_solution_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nodes_path = root / "nodes.csv"
            edges_path = root / "edges.csv"
            nodes_path.write_text(
                "node_id,T,is_leaf,state_id\n"
                "0,0,0,-1\n"
                "1,1,1,1\n"
                "2,1,1,3\n"
            )
            edges_path.write_text(
                "parent,child,branch_length\n0,1,1\n0,2,1\n"
            )
            output_dir = root / "mach2"

            def fake_run(command, **kwargs):
                output_path = Path(command[-1])
                (output_path / "0-T-0.refined.tree").write_text("1\t2\t0.5\n")
                (output_path / "0-T-0.nodes").write_text("1\t1\n")
                (output_path / "0-T-0.location.labeling").write_text("1\t0\n")
                return subprocess.CompletedProcess(command, 0, "done", "")

            with (
                patch(
                    "src.run_mach2_simulated_tree.resolve_executable",
                    return_value=Path("/mock/mach2"),
                ),
                patch(
                    "src.run_mach2_simulated_tree.subprocess.run",
                    side_effect=fake_run,
                ) as run,
            ):
                result = run_mach2_simulated_tree(
                    nodes_path,
                    edges_path,
                    output_dir,
                    timelimit=1800,
                )

                self.assertEqual(result, output_dir)
                self.assertTrue((output_dir / "mach2.log").is_file())
                self.assertEqual(
                    (output_dir / "tree.tsv").read_text(),
                    "1\t2\n1\t3\n",
                )
                self.assertEqual(
                    (output_dir / "labeling.tsv").read_text(),
                    "2\t1\n3\t3\n",
                )
                command = run.call_args.args[0]
                self.assertEqual(
                    command[-2:],
                    ["-o", str(output_dir / "mach2_output")],
                )
                self.assertEqual(command[3:5], ["-t", "1"])
                self.assertEqual(command[5:7], ["--timelimit", "1800"])

    def test_rejects_nonpositive_timelimit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(ValueError, "timelimit"):
                run_mach2_simulated_tree(
                    root / "nodes.csv",
                    root / "edges.csv",
                    root / "mach2",
                    timelimit=0,
                )


if __name__ == "__main__":
    unittest.main()
