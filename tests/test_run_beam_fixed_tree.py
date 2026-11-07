from __future__ import annotations

from argparse import Namespace
import csv
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from src.combine_beam_summaries import combine_summaries
from src.generate_beam_barcodes import read_tree
from src.run_beam_fixed_tree import (
    render_fixed_tree,
    run_inference,
    summarize_result_log,
    write_beam_xml,
    write_tissue_labels,
)


class RunBeamFixedTreeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.nodes_path = self.root / "nodes.csv"
        self.edges_path = self.root / "edges.csv"
        self.nodes_path.write_text(
            "node_id,T,is_leaf,state_id\n"
            "root,2,0,-1\n"
            "branch,1.5,0,-1\n"
            "left,0,1,0\n"
            "right,0,1,3\n"
        )
        self.edges_path.write_text(
            "parent,child,branch_length\n"
            "root,branch,0.5\n"
            "branch,left,1.5\n"
            "branch,right,1.5\n"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_newick_suppresses_unary_root_and_preserves_its_stem(self) -> None:
        tree = read_tree(self.nodes_path, self.edges_path)

        newick, origin_time = render_fixed_tree(tree)

        self.assertEqual(newick, "(left:1.5,right:1.5);")
        self.assertAlmostEqual(origin_time, 2.0001)

    def test_tissue_labels_follow_barcode_taxon_order(self) -> None:
        output_path = self.root / "tissues.csv"

        observed = write_tissue_labels(
            self.nodes_path,
            ("right", "left"),
            output_path,
        )

        self.assertEqual(observed, ("A", "D"))
        with output_path.open(newline="") as input_file:
            self.assertEqual(
                list(csv.reader(input_file)),
                [["right", "D"], ["left", "A"]],
            )

    def test_xml_template_receives_tree_specific_values(self) -> None:
        template = Path(__file__).parents[1] / "src/beam_fixed_tree_template.xml"
        output = self.root / "input.xml"

        write_beam_xml(template, output, origin_time=2.0001, chain_length=500)

        content = output.read_text()
        self.assertIn('chainLength="500"', content)
        self.assertNotIn("@@", content)
        root = ElementTree.parse(output).getroot()
        origin_time = next(
            element for element in root.iter() if element.get("id") == "originTime"
        )
        self.assertAlmostEqual(float(origin_time.text or ""), 2.0001)

    def test_summary_uses_burnin_and_reports_all_parameter_columns(self) -> None:
        log_path = self.root / "result.log"
        log_path.write_text(
            "# BEAST log\n"
            "Sample\tposterior\tlikelihood\tclockRate\n"
            "0\t-20\t-10\t1\n"
            "10\t-19\t-9\t2\n"
            "20\t-18\t-8\t3\n"
            "30\t-17\t-7\t4\n"
            "40\t-16\t-6\t5\n"
        )
        summary_path = self.root / "summary.tsv"

        summarize_result_log(log_path, summary_path, "setting_1/tree_0", 0.2)

        with summary_path.open(newline="") as input_file:
            rows = list(csv.DictReader(input_file, delimiter="\t"))
        self.assertEqual([row["parameter"] for row in rows], ["clockRate"])
        self.assertEqual(rows[0]["samples_after_burnin"], "4")
        self.assertEqual(rows[0]["median"], "3.5")

    def test_runner_and_batch_summary_with_mock_beast(self) -> None:
        barcode_path = self.root / "barcodes.tsv"
        barcode_path.write_text(
            "\tbarcode_1\nleft\t0\nright\t1\n"
        )
        fake_beast = self.root / "fake-beast"
        fake_beast.write_text(
            "#!/usr/bin/env python3\n"
            "from pathlib import Path\n"
            "Path('result.log').write_text("
            "'Sample\\tposterior\\tlikelihood\\tprior\\tclockRate\\n"
            "0\\t-3\\t-2\\t-1\\t1\\n10\\t-2\\t-1\\t-1\\t2\\n')\n"
            "Path('result.trees').write_text('#NEXUS\\n')\n"
        )
        fake_beast.chmod(0o755)
        output_dir = self.root / "setting_1/batch_0/tree_0/beam_snakemake"
        template = Path(__file__).parents[1] / "src/beam_fixed_tree_template.xml"

        run_inference(
            Namespace(
                nodes=self.nodes_path,
                edges=self.edges_path,
                barcodes=barcode_path,
                template=template,
                output_dir=output_dir,
                beast_log=self.root / "beast.log",
                beast_executable=str(fake_beast),
                tree_id="setting_1/batch_0/tree_0",
                chain_length=100,
                seed=17,
                burnin=0.0,
            )
        )
        parameter_output = self.root / "combined_parameters.tsv"
        status_output = self.root / "combined_status.tsv"
        combine_summaries(self.root, parameter_output, status_output)

        self.assertIn("completed", status_output.read_text())
        self.assertIn("clockRate", parameter_output.read_text())

    def test_missing_beast_executable_fails_with_configuration_hint(self) -> None:
        barcode_path = self.root / "barcodes.tsv"
        barcode_path.write_text("\tbarcode_1\nleft\t0\nright\t1\n")
        template = Path(__file__).parents[1] / "src/beam_fixed_tree_template.xml"

        with self.assertRaisesRegex(
            FileNotFoundError,
            "Set BEAST_EXECUTABLE or pass --config",
        ):
            run_inference(
                Namespace(
                    nodes=self.nodes_path,
                    edges=self.edges_path,
                    barcodes=barcode_path,
                    template=template,
                    output_dir=self.root / "missing/beam_snakemake",
                    beast_log=self.root / "missing.beast.log",
                    beast_executable="missing-beast-executable",
                    tree_id="setting_1/batch_0/tree_0",
                    chain_length=100,
                    seed=17,
                    burnin=0.0,
                )
            )

    def test_single_tip_tree_is_recorded_as_skipped(self) -> None:
        nodes_path = self.root / "single_node.csv"
        edges_path = self.root / "single_edges.csv"
        nodes_path.write_text("node_id,T,is_leaf,state_id\nonly,0,1,2\n")
        edges_path.write_text("parent,child,branch_length\n")
        output_dir = self.root / "single/beam_snakemake"

        run_inference(
            Namespace(
                nodes=nodes_path,
                edges=edges_path,
                barcodes=self.root / "not-needed.tsv",
                template=self.root / "not-needed.xml",
                output_dir=output_dir,
                beast_log=self.root / "single.log",
                beast_executable="missing-beast-executable",
                tree_id="setting_1/batch_0/tree_1",
                chain_length=100,
                seed=18,
                burnin=0.1,
            )
        )

        self.assertIn("skipped_single_tip", (output_dir / "run_status.tsv").read_text())


if __name__ == "__main__":
    unittest.main()
