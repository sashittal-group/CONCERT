configfile: "src/analyses/tls/config.yaml"

from pathlib import Path
import csv
import os
import shlex
import sys
import zlib
import yaml
from src.analyses.simulation.prepare_beam_barcodes import read_tree
from src.analyses.simulation.inference_completed_batches import (
    LINEAR_INFERENCE_COMPLETED_BATCHES,
    STAR3_INFERENCE_COMPLETED_BATCHES,
)

python_executable = sys.executable

crc11_input_dir = (
    "data/sgootr/all_trees/combined_inference/"
    "S10_CRC11_transition_chain_input"
)
crc11_nodes_file = f"{crc11_input_dir}/nodes_2.csv"
crc11_edges_file = f"{crc11_input_dir}/edges_2.csv"
crc11_transition_edges_file = f"{crc11_input_dir}/transition_edges.txt"
crc11_search_data_dir = (
    "data/sgootr/all_trees/combined_inference/"
    "S10_CRC11_state_specific_3transition_shared_source_search_snakemake"
)
crc11_search_summary_dir = (
    "outputs/sgootr/state_specific_schedule_search/"
    "S10_CRC11_state_specific_3transition_shared_source_search_snakemake"
)
crc11_fit_root = (
    "data/sgootr/all_trees/combined_inference/"
    "S10_CRC11_state_specific_fit_snakemake"
)
crc11_fit_file = f"{crc11_fit_root}/schedule_3857/inference.txt"
crc11_fit_schedule_file = f"{crc11_fit_root}/best_schedule.txt"
crc11_selected_fit_dir = "outputs/sgootr/CRC11/selected_fit_snakemake"
crc11_transition_count_data_dir = (
    "data/sgootr/all_trees/combined_inference/"
    "S10_CRC11_ptln_transition_count_analysis_snakemake"
)
crc11_transition_count_summary_dir = (
    "outputs/sgootr/transition_count_analysis/"
    "S10_CRC11_ptln_transition_count_analysis_snakemake"
)
crc11_figure_dir = "outputs/sgootr/CRC11/manuscript_figure_snakemake"

tls_fixed_sampling_config_path = Path(
    "src/analyses/tls/config_fixed_sampling_0.1.yaml"
)
with tls_fixed_sampling_config_path.open() as fixed_sampling_config_file:
    tls_fixed_sampling_config = yaml.safe_load(fixed_sampling_config_file)[
        "tls_composite"
    ]
tls_fixed_sampling_paths = {
    name: (
        Path(value)
        if Path(value).is_absolute()
        else tls_fixed_sampling_config_path.parent / value
    ).resolve()
    for name, value in (
        ("data_root", tls_fixed_sampling_config["data_root"]),
        ("inference_root", tls_fixed_sampling_config["inference_root"]),
        ("preprocessing_root", tls_fixed_sampling_config["preprocessing_root"]),
        ("figure_root", tls_fixed_sampling_config["figure_root"]),
    )
}
tls_fixed_sampling_files = tls_fixed_sampling_config["files"]
tls_fixed_sampling_plotting = tls_fixed_sampling_config["plotting"]
tls_fixed_sampling_datasets = tls_fixed_sampling_config["datasets"]
tls_fixed_sampling_inference_outputs = [
    str(
        tls_fixed_sampling_paths["inference_root"]
        / dataset
        / "fixed_sampling_0.1"
        / "inference.out"
    )
    for dataset in tls_fixed_sampling_datasets
]
tls_fixed_sampling_profile_outputs = [
    str(tls_fixed_sampling_paths["preprocessing_root"] / dataset / filename)
    for dataset in tls_fixed_sampling_datasets
    for filename in tls_fixed_sampling_files.values()
]
tls_fixed_sampling_figure_basenames = [
    tls_fixed_sampling_plotting["output_basename"],
    *tls_fixed_sampling_plotting["parameter_output_basenames"].values(),
]
tls_fixed_sampling_plot_outputs = [
    str(tls_fixed_sampling_paths["figure_root"] / f"{basename}.{extension}")
    for basename in tls_fixed_sampling_figure_basenames
    for extension in tls_fixed_sampling_plotting["formats"]
]


def tls_tree_files(dataset, filename):
    tree_root = (
        tls_fixed_sampling_paths["data_root"]
        / dataset
        / "subtrees_kept_states"
    )
    node_files = sorted(tree_root.glob("tree_*/subtree_*/nodes.csv"))
    if not node_files:
        raise FileNotFoundError(f"No exported subtree nodes found under {tree_root}")
    edge_files = [node_file.with_name("edges.csv") for node_file in node_files]
    missing = [edge_file for edge_file in edge_files if not edge_file.is_file()]
    if missing:
        raise ValueError(f"Missing matching subtree edge files: {missing}")
    return [
        str(node_file if filename == "nodes.csv" else edge_file)
        for node_file, edge_file in zip(node_files, edge_files)
    ]


num_batches = 10
trees_per_batch = 10
star_trees_per_batch = 128
seed_stride = 100
batch_indices = range(num_batches)
star_tree_indices = range(star_trees_per_batch)
setting_root = "data/simulation/coord_diff"
star_transition_roots = {3: f"{setting_root}/star_3"}
star_transition_setting_ids = {
    count: glob_wildcards(f"{root}/setting_{{setting}}/setting.txt").setting
    for count, root in star_transition_roots.items()
}
linear_transition_root = f"{setting_root}/linear_A_to_D"
linear_transition_num_settings = 100
linear_transition_setting_ids = range(
    1,
    linear_transition_num_settings + 1,
)
first_eight_tree_indices = range(8)
linear_transition_first8_tree_indices = first_eight_tree_indices
mach2_simulation_roots = {
    "star_3": star_transition_roots[3],
    "linear_A_to_D": linear_transition_root,
}
mach2_timelimit = 1800
mach2_executable = config.get("mach2_executable") or os.environ.get(
    "MACH2_EXECUTABLE"
)
mach2_executable_arg = (
    f"--mach2-executable {shlex.quote(str(mach2_executable))}"
    if mach2_executable
    else ""
)
star_transition_edges_file = "src/analyses/simulation/config/transition_edges_star.txt"


def star_source_seed(wildcards):
    setting = int(wildcards.setting)
    batch = int(wildcards.batch)
    tree = int(wildcards.tree)
    if tree < trees_per_batch:
        return (
            setting * num_batches * trees_per_batch
            + batch * trees_per_batch
            + tree
        ) * seed_stride
    return (
        100_000_000
        + setting * num_batches * star_trees_per_batch * seed_stride
        + (batch * star_trees_per_batch + tree) * seed_stride
    )


def star_transition_tree_edge_inputs(num_transition_times):
    root = star_transition_roots[num_transition_times]
    setting_ids = glob_wildcards(f"{root}/setting_{{setting}}/setting.txt").setting
    if not setting_ids:
        raise FileNotFoundError(
            f"No settings found under {root}; run "
            "`python -m src.concert --analysis generate-star-transition-settings` first"
        )
    return expand(
        f"{root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/edges.csv",
        setting=setting_ids,
        batch=batch_indices,
        tree=star_tree_indices,
    )
tls_root = "data/troupe/TLS"
tls_tree_root = f"{tls_root}/subtrees_kept_states"
tls_nodes_files = sorted(Path(tls_tree_root).rglob("nodes.csv"))
tls_tree_pairs = [
    (nodes_file, nodes_file.with_name("edges.csv"))
    for nodes_file in tls_nodes_files
    if nodes_file.with_name("edges.csv").is_file()
]
if len(tls_tree_pairs) != len(tls_nodes_files):
    raise ValueError(f"Some TLS node files have no matching edges file in {tls_tree_root}")
tls_transition_edges_file = "data/troupe/transition-edges.txt"
tls_transition_times_file = "data/troupe/transition-times.txt"
tls_fixed_sampling_transition_counts = (5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200)
tls_fixed_sampling_transition_root = f"{tls_root}/fixed_sampling_0.1"
tls_fixed_sampling_transition_outputs = [
    f"{tls_fixed_sampling_transition_root}/inference_transition_times_{count}.out"
    for count in tls_fixed_sampling_transition_counts
]
tlsc_root = "data/troupe/TLSC"
tlsc_tree_root = f"{tlsc_root}/subtrees_kept_states"
tlsc_fixed_sampling_transition_root = f"{tlsc_root}/fixed_sampling_0.1"
tlsc_fixed_sampling_transition_outputs = [
    f"{tlsc_fixed_sampling_transition_root}/inference_transition_times_{count}.out"
    for count in tls_fixed_sampling_transition_counts
]
tlsc_nodes_files = sorted(Path(tlsc_tree_root).rglob("nodes.csv"))
tlsc_tree_pairs = [
    (nodes_file, nodes_file.with_name("edges.csv"))
    for nodes_file in tlsc_nodes_files
    if nodes_file.with_name("edges.csv").is_file()
]
if len(tlsc_tree_pairs) != len(tlsc_nodes_files):
    raise ValueError(f"Some TLSC node files have no matching edges file in {tlsc_tree_root}")
tlsc_transition_times_file = "data/troupe/transition-times.txt"


def star_diversitree_tree_files(num_transition_times, wildcards, filename):
    batch_dir = (
        Path(star_transition_roots[num_transition_times])
        / f"setting_{wildcards.setting}"
        / f"batch_{wildcards.batch}"
    )
    tree_dirs = sorted(
        path
        for path in batch_dir.glob("tree_*")
        if path.is_dir()
        and path.name.removeprefix("tree_").isdigit()
        and (
            (path / "nodes.csv").is_file()
            or (path / "edges.csv").is_file()
        )
    )
    if not tree_dirs:
        raise FileNotFoundError(f"No simulated trees found under {batch_dir}")
    files = [tree_dir / filename for tree_dir in tree_dirs]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing diversitree input files: {missing}")
    return [str(path) for path in files]


def star3_diversitree_tree_files(wildcards, filename):
    return star_diversitree_tree_files(3, wildcards, filename)


def star3_beam_origin_time(wildcards, input):
    tree = read_tree(Path(input.nodes), Path(input.edges))
    return tree.node_ages[tree.root_id] + 1e-4


def linear_beam_origin_time(wildcards, input):
    tree = read_tree(Path(input.nodes), Path(input.edges))
    return tree.node_ages[tree.root_id] + 1e-4


def star3_beam_barcode_outputs(wildcards):
    node_files = sorted(
        Path(star_transition_roots[3]).glob(
            "setting_*/batch_*/tree_*/nodes.csv"
        )
    )
    return [
        str(nodes_file.parent / "barcodes_100_sites.tsv")
        for nodes_file in node_files
        if nodes_file.with_name("edges.csv").is_file()
    ]


def star3_diversitree_output_files(wildcards):
    batch_pairs = sorted(
        {
            (
                inference_file.parent.parent.name,
                inference_file.parent.name,
            )
            for inference_file in Path(star_transition_roots[3]).glob(
                "setting_*/batch_*/inference_progressive_joint_first_8_trees.out"
            )
        }
    )
    return [
        (
            f"{star_transition_roots[3]}/{setting}/{batch}/"
            "diversitree_musse/joint_musse_fit.txt"
        )
        for setting, batch in batch_pairs
    ]


def star3_beam_seed(wildcards):
    tree_id = (
        wildcards.setting,
        wildcards.batch,
        wildcards.tree,
    )
    if tree_id == ("74", "0", "0"):
        return 7
    path = "/".join(tree_id).encode()
    return 7 + zlib.crc32(path)


def star3_beam_mcmc_seed(wildcards):
    tree_id = f"{wildcards.setting}/{wildcards.batch}/{wildcards.tree}"
    return 1 + zlib.crc32(tree_id.encode()) % 2_147_483_646


def linear_beam_seed(wildcards):
    tree_id = f"{wildcards.setting}/{wildcards.batch}/{wildcards.tree}"
    return 7 + zlib.crc32(tree_id.encode())


def linear_beam_mcmc_seed(wildcards):
    tree_id = f"{wildcards.setting}/{wildcards.batch}/{wildcards.tree}"
    return 1 + zlib.crc32(tree_id.encode()) % 2_147_483_646


def star3_beam_run_directories(wildcards):
    node_files = sorted(
        Path(star_transition_roots[3]).glob(
            "setting_*/batch_*/tree_*/nodes.csv"
        )
    )
    return [
        str(nodes_file.parent / "beam_snakemake")
        for nodes_file in node_files
        if nodes_file.with_name("edges.csv").is_file()
    ]


star3_beam_chain_length = int(config.get("star3_beam_chain_length", 1_000_000))
star3_beam_burnin = float(config.get("star3_beam_burnin", 0.1))
star3_beam_executable = config.get(
    "star3_beam_beast_executable",
    os.environ.get("BEAST_EXECUTABLE", "beast"),
)
linear_beam_chain_length = int(
    config.get("linear_beam_chain_length", star3_beam_chain_length)
)
linear_beam_burnin = float(config.get("linear_beam_burnin", star3_beam_burnin))
linear_beam_executable = config.get(
    "linear_beam_beast_executable",
    star3_beam_executable,
)
star3_beam_parameter_summary = (
    f"{star_transition_roots[3]}/beam_snakemake_parameter_summary.tsv"
)
star3_beam_run_status = f"{star_transition_roots[3]}/beam_snakemake_run_status.tsv"


rule all:
    input:
        tls_fixed_sampling_plot_outputs,



# TLS and TLSC results
rule tls_fixed_sampling_01:
    input:
        inference=tls_fixed_sampling_inference_outputs,
        profiles=tls_fixed_sampling_profile_outputs,
        plots=tls_fixed_sampling_plot_outputs,


rule tls_fixed_sampling_01_inference:
    input:
        nodes=lambda wildcards: tls_tree_files(
            wildcards.dataset,
            "nodes.csv",
        ),
        edges=lambda wildcards: tls_tree_files(
            wildcards.dataset,
            "edges.csv",
        ),
        transition_edges=tls_transition_edges_file,
        transition_times=tls_transition_times_file,
    output:
        out=str(
            tls_fixed_sampling_paths["inference_root"]
            / "{dataset}"
            / "fixed_sampling_0.1"
            / "inference.out"
        ),
    params:
        tree_directory=lambda wildcards: (
            tls_fixed_sampling_paths["data_root"]
            / wildcards.dataset
            / "subtrees_kept_states"
        ),
        num_trees=lambda wildcards: len(
            tls_tree_files(wildcards.dataset, "nodes.csv")
        ),
        num_states=len(tls_fixed_sampling_config["state_names"]),
        sampling_probability=tls_fixed_sampling_config[
            "fixed_sampling_probability"
        ],
    log:
        optimization=str(
            tls_fixed_sampling_paths["inference_root"]
            / "{dataset}"
            / "fixed_sampling_0.1"
            / "inference.log"
        ),
        stdout=str(
            tls_fixed_sampling_paths["inference_root"]
            / "{dataset}"
            / "fixed_sampling_0.1"
            / "stdout.log"
        ),
        stderr=str(
            tls_fixed_sampling_paths["inference_root"]
            / "{dataset}"
            / "fixed_sampling_0.1"
            / "stderr.log"
        ),
    benchmark:
        str(
            tls_fixed_sampling_paths["inference_root"]
            / "{dataset}"
            / "fixed_sampling_0.1"
            / "inference.benchmark"
        ),
    wildcard_constraints:
        dataset="TLS|TLSC",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export DYLD_FALLBACK_LIBRARY_PATH=\"/opt/homebrew/opt/llvm/lib/c++\" && "
        "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && "
        "python -m src.concert --given-transition-times {params.tree_directory:q} "
        "--num-states {params.num_states} --num-trees {params.num_trees} "
        "--transition-edges-file {input.transition_edges:q} "
        "--transition-times-file {input.transition_times:q} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer adam --normalize-root-likelihood "
        "--fixed-shared-sampling-probability {params.sampling_probability} "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule tls_fixed_sampling_01_preprocess:
    input:
        settings=str(tls_fixed_sampling_config_path),
        code=[
            "src/analyses/common/profile_preprocess.py",
            "src/analyses/common/profile_tools.py",
            "src/analyses/common/profile_data.py",
            "src/analyses/common/profile_likelihood.py",
            "src/analyses/common/profile_config.py",
            "src/concert/optimize_transition_parameters.py",
            "src/concert/closed_form_transitions.py",
            "src/concert/closed_form_tree_likelihood.py",
            "src/concert/tree_utils.py",
            "src/concert/tree_csv.py",
        ],
        inference=str(
            tls_fixed_sampling_paths["inference_root"]
            / "{dataset}"
            / "fixed_sampling_0.1"
            / "inference.out"
        ),
        nodes=lambda wildcards: tls_tree_files(
            wildcards.dataset,
            "nodes.csv",
        ),
        edges=lambda wildcards: tls_tree_files(
            wildcards.dataset,
            "edges.csv",
        ),
    output:
        profiles=str(
            tls_fixed_sampling_paths["preprocessing_root"]
            / "{dataset}"
            / tls_fixed_sampling_files["profiles"]
        ),
        ranges=str(
            tls_fixed_sampling_paths["preprocessing_root"]
            / "{dataset}"
            / tls_fixed_sampling_files["ranges"]
        ),
        metadata=str(
            tls_fixed_sampling_paths["preprocessing_root"]
            / "{dataset}"
            / tls_fixed_sampling_files["metadata"]
        ),
    wildcard_constraints:
        dataset="TLS|TLSC",
    shell:
        "python -m src.concert --analysis profile-preprocess "
        "--config {input.settings:q} --dataset {wildcards.dataset:q}"


rule tls_fixed_sampling_01_plot:
    input:
        settings=str(tls_fixed_sampling_config_path),
        code=[
            "src/analyses/common/profile_plot.py",
            "src/analyses/common/profile_config.py",
        ],
        ranges=expand(
            str(
                tls_fixed_sampling_paths["preprocessing_root"]
                / "{dataset}"
                / tls_fixed_sampling_files["ranges"]
            ),
            dataset=tls_fixed_sampling_datasets,
        ),
        metadata=expand(
            str(
                tls_fixed_sampling_paths["preprocessing_root"]
                / "{dataset}"
                / tls_fixed_sampling_files["metadata"]
            ),
            dataset=tls_fixed_sampling_datasets,
        ),
    output:
        tls_fixed_sampling_plot_outputs,
    shell:
        "python -m src.concert --analysis profile-plot --config {input.settings:q}"


rule tls_transition_count_fixed_sampling_01:
    input:
        tls_fixed_sampling_transition_outputs,


rule infer_tls_transition_count_fixed_sampling_01:
    input:
        nodes=[str(nodes_file) for nodes_file, _ in tls_tree_pairs],
        edges=[str(edges_file) for _, edges_file in tls_tree_pairs],
        transition_edges=tls_transition_edges_file,
        transition_times=lambda wildcards: (
            f"data/troupe/transition-times-{wildcards.transition_count}.txt"
        ),
        initial_parameters=(
            f"{tls_fixed_sampling_transition_root}/"
            "inference_transition_times_{transition_count}.tmp.out"
        ),
    output:
        out=(
            f"{tls_fixed_sampling_transition_root}/"
            "inference_transition_times_{transition_count}.out"
        ),
    params:
        tree_directory=tls_tree_root,
        num_trees=len(tls_tree_pairs),
        num_states=3,
        python=python_executable,
    log:
        optimization=(
            f"{tls_fixed_sampling_transition_root}/"
            "inference_transition_times_{transition_count}.log"
        ),
        stdout=(
            f"{tls_fixed_sampling_transition_root}/"
            "inference_transition_times_{transition_count}.stdout.log"
        ),
        stderr=(
            f"{tls_fixed_sampling_transition_root}/"
            "inference_transition_times_{transition_count}.stderr.log"
        ),
    benchmark:
        f"{tls_fixed_sampling_transition_root}/inference_transition_times_{{transition_count}}.benchmark",
    wildcard_constraints:
        transition_count="5|10|15|20|25|30|40|50|75|100|200",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export DYLD_FALLBACK_LIBRARY_PATH=\"/opt/homebrew/opt/llvm/lib/c++\" && "
        "{params.python:q} -m src.concert --continue {params.tree_directory:q} "
        "--num-states {params.num_states} --num-trees {params.num_trees} "
        "--transition-edges-file {input.transition_edges:q} "
        "--transition-times-file {input.transition_times:q} "
        "--initial-parameters {input.initial_parameters:q} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer adam --normalize-root-likelihood "
        "--fixed-shared-sampling-probability 0.1 "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule tlsc_transition_count_fixed_sampling_01:
    input:
        tlsc_fixed_sampling_transition_outputs,


rule infer_tlsc_transition_count_fixed_sampling_01:
    input:
        nodes=[str(nodes_file) for nodes_file, _ in tlsc_tree_pairs],
        edges=[str(edges_file) for _, edges_file in tlsc_tree_pairs],
        transition_edges=tls_transition_edges_file,
        transition_times=lambda wildcards: (
            f"data/troupe/transition-times-{wildcards.transition_count}.txt"
        ),
        initial_parameters=(
            f"{tlsc_fixed_sampling_transition_root}/"
            "inference_transition_times_{transition_count}.tmp.out"
        ),
    output:
        out=(
            f"{tlsc_fixed_sampling_transition_root}/"
            "inference_transition_times_{transition_count}.out"
        ),
    params:
        tree_directory=tlsc_tree_root,
        num_trees=len(tlsc_tree_pairs),
        num_states=3,
        python=python_executable,
    log:
        optimization=(
            f"{tlsc_fixed_sampling_transition_root}/"
            "inference_transition_times_{transition_count}.log"
        ),
        stdout=(
            f"{tlsc_fixed_sampling_transition_root}/"
            "inference_transition_times_{transition_count}.stdout.log"
        ),
        stderr=(
            f"{tlsc_fixed_sampling_transition_root}/"
            "inference_transition_times_{transition_count}.stderr.log"
        ),
    benchmark:
        f"{tlsc_fixed_sampling_transition_root}/inference_transition_times_{{transition_count}}.benchmark",
    wildcard_constraints:
        transition_count="5|10|15|20|25|30|40|50|75|100|200",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export DYLD_FALLBACK_LIBRARY_PATH=\"/opt/homebrew/opt/llvm/lib/c++\" && "
        "{params.python:q} -m src.concert --continue {params.tree_directory:q} "
        "--num-states {params.num_states} --num-trees {params.num_trees} "
        "--transition-edges-file {input.transition_edges:q} "
        "--transition-times-file {input.transition_times:q} "
        "--initial-parameters {input.initial_parameters:q} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer adam --normalize-root-likelihood "
        "--fixed-shared-sampling-probability 0.1 "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule infer_tlsc:
    input:
        nodes=[str(nodes_file) for nodes_file, _ in tlsc_tree_pairs],
        edges=[str(edges_file) for _, edges_file in tlsc_tree_pairs],
        transition_edges=tls_transition_edges_file,
        transition_times=tlsc_transition_times_file,
    output:
        out=f"{tlsc_root}/inference.out",
    params:
        tree_directory=tlsc_tree_root,
        num_trees=len(tlsc_tree_pairs),
        num_states=3,
    log:
        optimization=f"{tlsc_root}/inference.log",
        stdout=f"{tlsc_root}/inference.stdout.log",
        stderr=f"{tlsc_root}/inference.stderr.log",
    benchmark:
        f"{tlsc_root}/inference.benchmark",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export DYLD_FALLBACK_LIBRARY_PATH=\"/opt/homebrew/opt/llvm/lib/c++\" && "
        "python -m src.concert --given-transition-times {params.tree_directory:q} "
        "--num-states {params.num_states} --num-trees {params.num_trees} "
        "--transition-edges-file {input.transition_edges:q} "
        "--transition-times-file {input.transition_times:q} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer adam --normalize-root-likelihood "
        "--shared-sampling-rate "
        "> {log.stdout:q} 2> {log.stderr:q}"


# Simulation and method-comparison results
rule simulate_star_3:
    input:
        setting_file=f"{star_transition_roots[3]}/setting_{{setting}}/setting.txt",
        code="src/analyses/simulation/simulate_coordinate_tree.py",
    output:
        nodes=f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/nodes.csv",
        edges=f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/edges.csv",
        T_fig=f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/T.png",
        T_extant_fig=f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/T_extant.png",
        transition_times=f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/transition_times.csv",
        transitions_with_edges=f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/transition_times_with_edges.csv",
    params:
        source_seed=star_source_seed,
        output_dir=lambda wildcards: (
            f"{star_transition_roots[3]}/setting_{wildcards.setting}/"
            f"batch_{wildcards.batch}/tree_{wildcards.tree}"
        ),
        python=sys.executable,
    log:
        std=f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/stdout.log",
        err=f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/stderr.log",
    benchmark:
        f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/benchmark.txt",
    shell:
        "{params.python:q} -m src.concert --analysis simulate-coordinate-tree {input.setting_file:q} "
        "{params.source_seed} {params.output_dir:q} "
        "> {log.std:q} 2> {log.err:q}"


rule simulate_star_trees_3:
    input:
        lambda wildcards: star_transition_tree_edge_inputs(3)


rule generate_linear_transition_settings:
    input:
        code="src/analyses/simulation/generate_linear_transition_settings.py",
    output:
        settings=expand(
            f"{linear_transition_root}/setting_{{setting}}/setting.txt",
            setting=linear_transition_setting_ids,
        ),
        manifest=f"{linear_transition_root}/generated_settings.csv",
    params:
        output_dir=linear_transition_root,
        num_settings=linear_transition_num_settings,
        seed=3,
        python=sys.executable,
    shell:
        "{params.python:q} -m src.concert --analysis generate-linear-transition-settings "
        "--output-dir {params.output_dir:q} "
        "--num-settings {params.num_settings} --seed {params.seed}"


rule simulate_linear_transition_3:
    input:
        setting_file=(
            f"{linear_transition_root}/setting_{{setting}}/setting.txt"
        ),
        code="src/analyses/simulation/simulate_coordinate_tree.py",
    output:
        nodes=f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/nodes.csv",
        edges=f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/edges.csv",
        T_fig=f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/T.png",
        T_extant_fig=f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/T_extant.png",
        transition_times=f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/transition_times.csv",
        transitions_with_edges=f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/transition_times_with_edges.csv",
    params:
        source_seed=star_source_seed,
        output_dir=lambda wildcards: (
            f"{linear_transition_root}/setting_{wildcards.setting}/"
            f"batch_{wildcards.batch}/tree_{wildcards.tree}"
        ),
        python=sys.executable,
    log:
        std=f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/stdout.log",
        err=f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/stderr.log",
    benchmark:
        f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/benchmark.txt",
    shell:
        "{params.python:q} -m src.concert --analysis simulate-coordinate-tree {input.setting_file:q} "
        "{params.source_seed} {params.output_dir:q} "
        "> {log.std:q} 2> {log.err:q}"


rule simulate_linear_transition_3_all:
    input:
        expand(
            f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
            "tree_{{tree}}/edges.csv",
            setting=linear_transition_setting_ids,
            batch=batch_indices,
            tree=star_tree_indices,
        )


rule run_mach2_simulated_tree:
    input:
        nodes=lambda wildcards: (
            f"{mach2_simulation_roots[wildcards.dataset]}/"
            f"setting_{wildcards.setting}/batch_{wildcards.batch}/"
            f"tree_{wildcards.tree}/nodes.csv"
        ),
        edges=lambda wildcards: (
            f"{mach2_simulation_roots[wildcards.dataset]}/"
            f"setting_{wildcards.setting}/batch_{wildcards.batch}/"
            f"tree_{wildcards.tree}/edges.csv"
        ),
        code=[
            "src/analyses/simulation/run_mach2_simulated_tree.py",
            "src/analyses/common/executables.py",
        ],
    output:
        mach2_dir=directory(
            f"{setting_root}/{{dataset}}/setting_{{setting}}/"
            "batch_{batch}/tree_{tree}/mach2_full"
        ),
    params:
        python=sys.executable,
        mach2_executable_arg=mach2_executable_arg,
        timelimit=mach2_timelimit,
    threads: 1
    log:
        f"{setting_root}/{{dataset}}/setting_{{setting}}/"
        "batch_{batch}/tree_{tree}/mach2_full.snakemake.log",
    benchmark:
        f"{setting_root}/{{dataset}}/setting_{{setting}}/"
        "batch_{batch}/tree_{tree}/mach2_full.benchmark.txt",
    wildcard_constraints:
        dataset="star_3|linear_A_to_D",
    shell:
        "{params.python:q} -m src.concert --analysis run-mach2-simulated-tree "
        "--nodes {input.nodes:q} --edges {input.edges:q} "
        "--output-dir {output.mach2_dir:q} --threads {threads} "
        "--timelimit {params.timelimit} "
        "{params.mach2_executable_arg} > {log:q} 2>&1"





star3_inference_completed_concert = [
    f"{star_transition_roots[3]}/setting_{setting}/batch_{batch}/"
    "inference_progressive_joint_first_8_trees.out"
    for setting, batch in STAR3_INFERENCE_COMPLETED_BATCHES
]
linear_inference_completed_concert = [
    f"{linear_transition_root}/setting_{setting}/batch_{batch}/"
    "inference_joint_coordinate_first_8_trees.out"
    for setting, batch in LINEAR_INFERENCE_COMPLETED_BATCHES
]
star3_inference_completed_diversitree = [
    f"{star_transition_roots[3]}/setting_{setting}/batch_{batch}/"
    "diversitree_musse_first8/joint_musse_fit.txt"
    for setting, batch in STAR3_INFERENCE_COMPLETED_BATCHES
]
linear_inference_completed_diversitree = [
    f"{linear_transition_root}/setting_{setting}/batch_{batch}/"
    "diversitree_musse_first8/joint_musse_fit.txt"
    for setting, batch in LINEAR_INFERENCE_COMPLETED_BATCHES
]
star3_inference_completed_beam = [
    f"{star_transition_roots[3]}/setting_{setting}/batch_{batch}/"
    f"tree_{tree}/beam_snakemake"
    for setting, batch in STAR3_INFERENCE_COMPLETED_BATCHES
    for tree in first_eight_tree_indices
]
linear_inference_completed_beam = [
    f"{linear_transition_root}/setting_{setting}/batch_{batch}/"
    f"tree_{tree}/beam_snakemake"
    for setting, batch in LINEAR_INFERENCE_COMPLETED_BATCHES
    for tree in linear_transition_first8_tree_indices
]
star3_inference_completed_mach2 = [
    f"{star_transition_roots[3]}/setting_{setting}/batch_{batch}/"
    f"tree_{tree}/mach2_full"
    for setting, batch in STAR3_INFERENCE_COMPLETED_BATCHES
    for tree in first_eight_tree_indices
]
linear_inference_completed_mach2 = [
    f"{linear_transition_root}/setting_{setting}/batch_{batch}/"
    f"tree_{tree}/mach2_full"
    for setting, batch in LINEAR_INFERENCE_COMPLETED_BATCHES
    for tree in linear_transition_first8_tree_indices
]


rule coord_diff_inference_completed_concert:
    input:
        star3=star3_inference_completed_concert,
        linear=linear_inference_completed_concert


rule coord_diff_inference_completed_diversitree:
    input:
        star3=star3_inference_completed_diversitree,
        linear=linear_inference_completed_diversitree


rule coord_diff_inference_completed_beam:
    input:
        star3=star3_inference_completed_beam,
        linear=linear_inference_completed_beam


rule coord_diff_inference_completed_mach2:
    input:
        star3=star3_inference_completed_mach2,
        linear=linear_inference_completed_mach2


rule coord_diff_inference_completed_batches_methods:
    input:
        concert=star3_inference_completed_concert + linear_inference_completed_concert,
        diversitree=(
            star3_inference_completed_diversitree
            + linear_inference_completed_diversitree
        ),
        beam=star3_inference_completed_beam + linear_inference_completed_beam,
        mach2=star3_inference_completed_mach2 + linear_inference_completed_mach2


rule run_diversitree_star_3_batch:
    input:
        nodes=lambda wildcards: ancient(
            star3_diversitree_tree_files(wildcards, "nodes.csv")
        ),
        edges=lambda wildcards: ancient(
            star3_diversitree_tree_files(wildcards, "edges.csv")
        ),
        code="src/analyses/simulation/run_diversitree_musse_batch.R",
    output:
        fit=(
            f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/"
            "diversitree_musse/joint_musse_fit.txt"
        ),
    params:
        batch_dir=lambda wildcards: (
            f"{star_transition_roots[3]}/setting_{wildcards.setting}/"
            f"batch_{wildcards.batch}"
        ),
        output_dir=lambda wildcards: (
            f"{star_transition_roots[3]}/setting_{wildcards.setting}/"
            f"batch_{wildcards.batch}/diversitree_musse"
        ),
        maxit=500,
    threads: 1
    log:
        f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/"
        "diversitree_musse.snakemake.log",
    shell:
        "module load R/4.5.2-gfbf-2025b GSL/2.8-GCC-14.3.0 && "
        "Rscript {input.code:q} {params.batch_dir:q} {params.output_dir:q} "
        "{params.maxit} > {log:q} 2>&1"


rule run_diversitree_star_3_first8_batch:
    input:
        nodes=lambda wildcards: expand(
            f"{star_transition_roots[3]}/setting_{wildcards.setting}/"
            f"batch_{wildcards.batch}/tree_{{tree}}/nodes.csv",
            tree=first_eight_tree_indices,
        ),
        edges=lambda wildcards: expand(
            f"{star_transition_roots[3]}/setting_{wildcards.setting}/"
            f"batch_{wildcards.batch}/tree_{{tree}}/edges.csv",
            tree=first_eight_tree_indices,
        ),
        code="src/analyses/simulation/run_diversitree_musse_batch.R",
    output:
        fit=(
            f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/"
            "diversitree_musse_first8/joint_musse_fit.txt"
        ),
    params:
        batch_dir=lambda wildcards: (
            f"{star_transition_roots[3]}/setting_{wildcards.setting}/"
            f"batch_{wildcards.batch}"
        ),
        output_dir=lambda wildcards: (
            f"{star_transition_roots[3]}/setting_{wildcards.setting}/"
            f"batch_{wildcards.batch}/diversitree_musse_first8"
        ),
        maxit=500,
        transition_model="star",
        num_trees=8,
    threads: 1
    log:
        f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/"
        "diversitree_musse_first8.snakemake.log",
    shell:
        "module load R/4.5.2-gfbf-2025b GSL/2.8-GCC-14.3.0 && "
        "Rscript {input.code:q} {params.batch_dir:q} {params.output_dir:q} "
        "{params.maxit} {params.transition_model:q} {params.num_trees} "
        "> {log:q} 2>&1"


rule run_diversitree_linear_transition_batch:
    input:
        nodes=lambda wildcards: expand(
            f"{linear_transition_root}/setting_{wildcards.setting}/"
            f"batch_{wildcards.batch}/tree_{{tree}}/nodes.csv",
            tree=star_tree_indices,
        ),
        edges=lambda wildcards: expand(
            f"{linear_transition_root}/setting_{wildcards.setting}/"
            f"batch_{wildcards.batch}/tree_{{tree}}/edges.csv",
            tree=star_tree_indices,
        ),
        code="src/analyses/simulation/run_diversitree_musse_batch.R",
    output:
        fit=(
            f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
            "diversitree_musse/joint_musse_fit.txt"
        ),
    params:
        batch_dir=lambda wildcards: (
            f"{linear_transition_root}/setting_{wildcards.setting}/"
            f"batch_{wildcards.batch}"
        ),
        output_dir=lambda wildcards: (
            f"{linear_transition_root}/setting_{wildcards.setting}/"
            f"batch_{wildcards.batch}/diversitree_musse"
        ),
        maxit=500,
        transition_model="linear",
    threads: 1
    log:
        f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
        "diversitree_musse.snakemake.log",
    shell:
        "module load R/4.5.2-gfbf-2025b GSL/2.8-GCC-14.3.0 && "
        "Rscript {input.code:q} {params.batch_dir:q} {params.output_dir:q} "
        "{params.maxit} {params.transition_model:q} > {log:q} 2>&1"


rule run_diversitree_linear_transition_first8_batch:
    input:
        nodes=lambda wildcards: expand(
            f"{linear_transition_root}/setting_{wildcards.setting}/"
            f"batch_{wildcards.batch}/tree_{{tree}}/nodes.csv",
            tree=linear_transition_first8_tree_indices,
        ),
        edges=lambda wildcards: expand(
            f"{linear_transition_root}/setting_{wildcards.setting}/"
            f"batch_{wildcards.batch}/tree_{{tree}}/edges.csv",
            tree=linear_transition_first8_tree_indices,
        ),
        code="src/analyses/simulation/run_diversitree_musse_batch.R",
    output:
        fit=(
            f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
            "diversitree_musse_first8/joint_musse_fit.txt"
        ),
    params:
        batch_dir=lambda wildcards: (
            f"{linear_transition_root}/setting_{wildcards.setting}/"
            f"batch_{wildcards.batch}"
        ),
        output_dir=lambda wildcards: (
            f"{linear_transition_root}/setting_{wildcards.setting}/"
            f"batch_{wildcards.batch}/diversitree_musse_first8"
        ),
        maxit=500,
        transition_model="linear",
        num_trees=8,
    threads: 1
    log:
        f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
        "diversitree_musse_first8.snakemake.log",
    shell:
        "module load R/4.5.2-gfbf-2025b GSL/2.8-GCC-14.3.0 && "
        "Rscript {input.code:q} {params.batch_dir:q} {params.output_dir:q} "
        "{params.maxit} {params.transition_model:q} {params.num_trees} "
        "> {log:q} 2>&1"


rule diversitree_star_3_matched_batches:
    input:
        star3_diversitree_output_files


rule diversitree_linear_transition_3_all:
    input:
        expand(
            f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
            "diversitree_musse/joint_musse_fit.txt",
            setting=linear_transition_setting_ids,
            batch=batch_indices,
        )


rule diversitree_linear_transition_first8_all:
    input:
        expand(
            f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
            "diversitree_musse_first8/joint_musse_fit.txt",
            setting=linear_transition_setting_ids,
            batch=batch_indices,
        )


rule generate_star_3_beam_barcodes:
    input:
        nodes=(
            f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/"
            "tree_{tree}/nodes.csv"
        ),
        edges=(
            f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/"
            "tree_{tree}/edges.csv"
        ),
        code="src/analyses/simulation/prepare_beam_barcodes.py",
    output:
        f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/"
        "tree_{tree}/barcodes_100_sites.tsv",
    params:
        python=sys.executable,
        num_sites=100,
        mutation_rate=0.1,
        num_edit_states=4,
        silencing_rate=0.1,
        origin_time=star3_beam_origin_time,
        seed=star3_beam_seed,
    threads: 1
    log:
        f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/"
        "tree_{tree}/barcodes_100_sites.snakemake.log",
    shell:
        "{params.python:q} -m src.concert --analysis prepare-beam-barcodes "
        "--nodes {input.nodes:q} --edges {input.edges:q} "
        "--output {output:q} --num-sites {params.num_sites} "
        "--mutation-rate {params.mutation_rate} "
        "--num-edit-states {params.num_edit_states} "
        "--silencing-rate {params.silencing_rate} "
        "--origin-time {params.origin_time} --seed {params.seed} "
        "> {log:q} 2>&1"


rule generate_star_3_beam_barcodes_all:
    input:
        star3_beam_barcode_outputs


rule run_star_3_beam_fixed_tree:
    input:
        nodes=(
            f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/"
            "tree_{tree}/nodes.csv"
        ),
        edges=(
            f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/"
            "tree_{tree}/edges.csv"
        ),
        barcodes=(
            f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/"
            "tree_{tree}/barcodes_100_sites.tsv"
        ),
        template="src/analyses/simulation/templates/beam_fixed_tree_template.xml",
        code=[
            "src/analyses/simulation/run_beam_fixed_tree.py",
            "src/analyses/simulation/prepare_beam_barcodes.py",
        ],
    output:
        run_dir=directory(
            f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/"
            "tree_{tree}/beam_snakemake"
        ),
    params:
        python=sys.executable,
        beast_executable=star3_beam_executable,
        beast_log=lambda wildcards: (
            f"{star_transition_roots[3]}/setting_{wildcards.setting}/"
            f"batch_{wildcards.batch}/tree_{wildcards.tree}/"
            "beam_snakemake.beast.log"
        ),
        chain_length=star3_beam_chain_length,
        burnin=star3_beam_burnin,
        seed=star3_beam_mcmc_seed,
        tree_id=lambda wildcards: (
            f"setting_{wildcards.setting}/batch_{wildcards.batch}/"
            f"tree_{wildcards.tree}"
        ),
    threads: 2
    log:
        f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/"
        "tree_{tree}/beam_snakemake.runner.log",
    shell:
        "{params.python:q} -m src.concert --analysis run-beam-fixed-tree "
        "--nodes {input.nodes:q} --edges {input.edges:q} "
        "--barcodes {input.barcodes:q} --template {input.template:q} "
        "--output-dir {output.run_dir:q} --beast-log {params.beast_log:q} "
        "--beast-executable {params.beast_executable:q} "
        "--chain-length {params.chain_length} --burnin {params.burnin} "
        "--seed {params.seed} --tree-id {params.tree_id:q} "
        "> {log:q} 2>&1"


rule summarize_star_3_beam_all:
    input:
        runs=star3_beam_run_directories,
        code="src/analyses/simulation/combine_beam_summaries.py",
    output:
        parameters=star3_beam_parameter_summary,
        status=star3_beam_run_status,
    params:
        python=sys.executable,
        root=star_transition_roots[3],
    shell:
        "{params.python:q} -m src.concert --analysis combine-beam-summaries "
        "--input-root {params.root:q} "
        "--parameter-output {output.parameters:q} "
        "--status-output {output.status:q}"


rule infer_star_3_beam_all:
    input:
        parameters=star3_beam_parameter_summary,
        status=star3_beam_run_status


rule generate_linear_transition_beam_barcodes_first8:
    input:
        nodes=(
            f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
            "tree_{tree}/nodes.csv"
        ),
        edges=(
            f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
            "tree_{tree}/edges.csv"
        ),
        code="src/analyses/simulation/prepare_beam_barcodes.py",
    output:
        f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
        "tree_{tree}/barcodes_100_sites.tsv",
    params:
        python=sys.executable,
        num_sites=100,
        mutation_rate=0.1,
        num_edit_states=4,
        silencing_rate=0.1,
        origin_time=linear_beam_origin_time,
        seed=linear_beam_seed,
    threads: 1
    log:
        f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
        "tree_{tree}/barcodes_100_sites.snakemake.log",
    shell:
        "{params.python:q} -m src.concert --analysis prepare-beam-barcodes "
        "--nodes {input.nodes:q} --edges {input.edges:q} "
        "--output {output:q} --num-sites {params.num_sites} "
        "--mutation-rate {params.mutation_rate} "
        "--num-edit-states {params.num_edit_states} "
        "--silencing-rate {params.silencing_rate} "
        "--origin-time {params.origin_time} --seed {params.seed} "
        "> {log:q} 2>&1"


rule generate_linear_transition_beam_barcodes_first8_all:
    input:
        expand(
            f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
            "tree_{{tree}}/barcodes_100_sites.tsv",
            setting=linear_transition_setting_ids,
            batch=batch_indices,
            tree=linear_transition_first8_tree_indices,
        )


rule run_linear_transition_beam_fixed_tree_first8:
    input:
        nodes=(
            f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
            "tree_{tree}/nodes.csv"
        ),
        edges=(
            f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
            "tree_{tree}/edges.csv"
        ),
        barcodes=(
            f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
            "tree_{tree}/barcodes_100_sites.tsv"
        ),
        template="src/analyses/simulation/templates/beam_fixed_tree_template.xml",
        code=[
            "src/analyses/simulation/run_beam_fixed_tree.py",
            "src/analyses/simulation/prepare_beam_barcodes.py",
        ],
    output:
        run_dir=directory(
            f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
            "tree_{tree}/beam_snakemake"
        ),
    params:
        python=sys.executable,
        beast_executable=linear_beam_executable,
        beast_log=lambda wildcards: (
            f"{linear_transition_root}/setting_{wildcards.setting}/"
            f"batch_{wildcards.batch}/tree_{wildcards.tree}/"
            "beam_snakemake.beast.log"
        ),
        chain_length=linear_beam_chain_length,
        burnin=linear_beam_burnin,
        seed=linear_beam_mcmc_seed,
        tree_id=lambda wildcards: (
            f"setting_{wildcards.setting}/batch_{wildcards.batch}/"
            f"tree_{wildcards.tree}"
        ),
    threads: 2
    log:
        f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
        "tree_{tree}/beam_snakemake.runner.log",
    shell:
        "{params.python:q} -m src.concert --analysis run-beam-fixed-tree "
        "--nodes {input.nodes:q} --edges {input.edges:q} "
        "--barcodes {input.barcodes:q} --template {input.template:q} "
        "--output-dir {output.run_dir:q} --beast-log {params.beast_log:q} "
        "--beast-executable {params.beast_executable:q} "
        "--chain-length {params.chain_length} --burnin {params.burnin} "
        "--seed {params.seed} --tree-id {params.tree_id:q} "
        "> {log:q} 2>&1"


rule infer_linear_transition_beam_first8_all:
    input:
        expand(
            f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
            "tree_{tree}/beam_snakemake",
            setting=linear_transition_setting_ids,
            batch=batch_indices,
            tree=linear_transition_first8_tree_indices,
        )


rule infer_star_3_joint_progressive:
    input:
        nodes=lambda wildcards: expand(
            f"{star_transition_roots[3]}/setting_{wildcards.setting}/"
            "batch_{batch}/tree_{tree}/nodes.csv",
            batch=[wildcards.batch],
            tree=range(8),
        ),
        edges=lambda wildcards: expand(
            f"{star_transition_roots[3]}/setting_{wildcards.setting}/"
            "batch_{batch}/tree_{tree}/edges.csv",
            batch=[wildcards.batch],
            tree=range(8),
        ),
        transition_edges=star_transition_edges_file,
        code=[
            "src/concert/progressive_transition_schedule.py",
            "src/concert/infer_joint_transition_times.py",
            "src/concert/infer_parameters.py",
            "src/concert/tree_csv.py",
            "src/concert/optimize_tree_parameters.py",
            "src/concert/closed_form_tree_likelihood.py",
            "src/concert/tree_utils.py",
            "src/concert/tree_types.py",
        ],
    output:
        out=(
            f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/"
            "inference_progressive_joint_first_8_trees.out"
        ),
    params:
        tree_directory=lambda wildcards: (
            f"{star_transition_roots[3]}/setting_{wildcards.setting}/"
            f"batch_{wildcards.batch}"
        ),
        random_seed=lambda wildcards: (
            int(wildcards.setting.split("_")[-1]) * 1000
            + int(wildcards.batch)
        ),
        python=sys.executable,
    log:
        optimization=(
            f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/"
            "inference_progressive_joint_first_8_trees.log"
        ),
        stdout=(
            f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/"
            "inference_progressive_joint_first_8_trees.stdout.log"
        ),
        stderr=(
            f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/"
            "inference_progressive_joint_first_8_trees.stderr.log"
        ),
    benchmark:
        f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/inference_progressive_joint_first_8_trees.benchmark",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && "
        "{params.python:q} -m src.concert --progressive-schedule "
        "--tree-directory {params.tree_directory:q} "
        "--num-states 4 --num-trees 8 "
        "--transition-edges-file {input.transition_edges:q} "
        "--num-transition-times 3 --max-time 1 --grid-size 1000 "
        "--random-seed {params.random_seed} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer adam --normalize-root-likelihood "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule infer_star_3_joint_progressive_all:
    input:
        expand(
            f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/"
            "inference_progressive_joint_first_8_trees.out",
            setting=star_transition_setting_ids[3],
            batch=batch_indices,
        )


rule infer_linear_transition_3_joint_coordinate_first8:
    input:
        nodes=lambda wildcards: expand(
            f"{linear_transition_root}/setting_{wildcards.setting}/"
            "batch_{batch}/tree_{tree}/nodes.csv",
            batch=[wildcards.batch],
            tree=range(8),
        ),
        edges=lambda wildcards: expand(
            f"{linear_transition_root}/setting_{wildcards.setting}/"
            "batch_{batch}/tree_{tree}/edges.csv",
            batch=[wildcards.batch],
            tree=range(8),
        ),
        transition_edges="src/analyses/simulation/config/transition_edges_linear.txt",
        code=[
            "src/analyses/crc11/joint_state_specific_transition_times.py",
            "src/concert/optimize_state_specific_parameters.py",
            "src/concert/closed_form_state_specific_transitions.py",
            "src/concert/infer_joint_transition_times.py",
            "src/concert/infer_parameters.py",
            "src/concert/optimize_tree_parameters.py",
            "src/concert/tree_csv.py",
            "src/concert/tree_utils.py",
            "src/concert/tree_types.py",
        ],
    output:
        out=(
            f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_joint_coordinate_first_8_trees.out"
        ),
    params:
        tree_directory=lambda wildcards: (
            f"{linear_transition_root}/setting_{wildcards.setting}/"
            f"batch_{wildcards.batch}"
        ),
        random_seed=lambda wildcards: (
            int(wildcards.setting.split("_")[-1]) * 1000
            + int(wildcards.batch)
        ),
        python=sys.executable,
    log:
        optimization=(
            f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_joint_coordinate_first_8_trees.log"
        ),
        stdout=(
            f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_joint_coordinate_first_8_trees.stdout.log"
        ),
        stderr=(
            f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_joint_coordinate_first_8_trees.stderr.log"
        ),
    benchmark:
        f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/inference_joint_coordinate_first_8_trees.benchmark",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && "
        "{params.python:q} -m src.concert --analysis joint-state-specific-transition-times "
        "--tree-directory {params.tree_directory:q} "
        "--num-states 4 --num-trees 8 "
        "--transition-edges-file {input.transition_edges:q} "
        "--max-time 1 --grid-size 1000 "
        "--random-seed {params.random_seed} --lbfgs-max-iter 20 "
        "--search-rounds 6 "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule infer_linear_transition_3_joint_coordinate_all:
    input:
        expand(
            f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_joint_coordinate_first_8_trees.out",
            setting=linear_transition_setting_ids,
            batch=batch_indices,
        )


rule plot_linear_transition_3_identifiable_parameters:
    input:
        truth=f"{linear_transition_root}/generated_settings.csv",
        joint_coordinate=expand(
            f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_joint_coordinate_first_8_trees.out",
            setting=linear_transition_setting_ids,
            batch=batch_indices,
        ),
        diversitree=expand(
            f"{linear_transition_root}/setting_{{setting}}/batch_{{batch}}/"
            "diversitree_musse/joint_musse_fit.txt",
            setting=linear_transition_setting_ids,
            batch=batch_indices,
        ),
        code="src/analyses/simulation/plot_linear_transition_inference_accuracy.py",
    output:
        plot=(
            "outputs/simulation/linear_A_to_D/figures/"
            "identifiable_parameters_ground_truth_vs_inferred.png"
        ),
    params:
        data_dir=linear_transition_root,
        python=sys.executable,
    shell:
        "{params.python:q} -m src.concert --analysis plot-linear-transition-inference-accuracy "
        "--data-dir {params.data_dir:q} --output {output.plot:q}"


# Bian et al. (SGOOTR/CRC11) results
rule sgootr_crc11_schedule_search:
    input:
        nodes=crc11_nodes_file,
        edges=crc11_edges_file,
        trees=f"{crc11_input_dir}/trees.txt",
        transition_edges=crc11_transition_edges_file,
        transition_times=f"{crc11_input_dir}/transition_times.txt",
        code=[
            "src/analyses/crc11/state_specific_schedule_search.py",
            "src/analyses/crc11/schedule_search.py",
            "src/concert/__main__.py",
            "src/concert/infer_state_specific_transitions.py",
            "src/concert/optimize_state_specific_parameters.py",
            "src/concert/closed_form_state_specific_transitions.py",
            "src/concert/tree_csv.py",
            "src/concert/tree_utils.py",
        ],
    output:
        results=directory(crc11_search_data_dir),
        summary=directory(crc11_search_summary_dir),
    params:
        python=sys.executable,
        input_dir=crc11_input_dir,
        output_dir=crc11_search_data_dir,
        summary_dir=crc11_search_summary_dir,
        workers=4,
        seed=42,
    log:
        "logs/sgootr/CRC11/state_specific_schedule_search.log",
    shell:
        "{params.python:q} -m src.analyses.crc11.state_specific_schedule_search "
        "--input-dir {params.input_dir:q} "
        "--output-dir {params.output_dir:q} "
        "--summary-dir {params.summary_dir:q} "
        "--workers {params.workers} --seed {params.seed} "
        "> {log:q} 2>&1"


rule sgootr_crc11_fit_schedule_3857:
    input:
        nodes=crc11_nodes_file,
        edges=crc11_edges_file,
        trees=f"{crc11_input_dir}/trees.txt",
        transition_edges=crc11_transition_edges_file,
        code=[
            "src/concert/__main__.py",
            "src/concert/infer_state_specific_transitions.py",
            "src/concert/optimize_state_specific_parameters.py",
            "src/concert/closed_form_state_specific_transitions.py",
            "src/concert/tree_csv.py",
            "src/concert/tree_utils.py",
        ],
    output:
        inference=crc11_fit_file,
        best_schedule=crc11_fit_schedule_file,
    params:
        python=sys.executable,
        input_dir=crc11_input_dir,
        fit_dir=f"{crc11_fit_root}/schedule_3857",
    log:
        optimizer=f"{crc11_fit_root}/schedule_3857/optimizer.log",
        stdout="logs/sgootr/CRC11/state_specific_fit_schedule_3857.log",
    shell:
        "mkdir -p {params.fit_dir:q} && "
        "{params.python:q} -m src.concert --state-specific-transitions "
        "{params.input_dir:q} --nc-to-pt-time 0.356 "
        "--pt-to-ln-time 0.131 --second-pt-to-ln-time 0.282 "
        "--output {output.inference:q} --log {log.optimizer:q} "
        "--device cpu --lbfgs-max-iter 20 > {log.stdout:q} 2>&1 && "
        "printf '%s\\n' 0.131 0.282 0.356 > {output.best_schedule:q}"


rule sgootr_crc11_selected_fit_plots:
    input:
        inference=crc11_fit_file,
        nodes=crc11_nodes_file,
        edges=crc11_edges_file,
        trees=f"{crc11_input_dir}/trees.txt",
        transition_edges=crc11_transition_edges_file,
        code=[
            "src/analyses/crc11/plot_state_specific_transitions.py",
            "src/analyses/crc11/preprocess.py",
            "src/analyses/crc11/profile_combined_inference.py",
            "src/analyses/common/tree_plotting.py",
            "src/analyses/common/profile_tools.py",
            "src/analyses/common/profile_config.py",
            "src/analyses/common/profile_plot.py",
            "src/analyses/common/state_coloring.py",
            "src/analyses/common/state_coloring_plot.py",
        ],
    output:
        directory(crc11_selected_fit_dir),
    params:
        python=sys.executable,
        input_dir=crc11_input_dir,
        output_dir=crc11_selected_fit_dir,
    log:
        "logs/sgootr/CRC11/selected_fit_plots.log",
    shell:
        "{params.python:q} -m src.analyses.crc11.plot_state_specific_transitions "
        "{input.inference:q} --input-dir {params.input_dir:q} "
        "--output-dir {params.output_dir:q} > {log:q} 2>&1"


rule sgootr_crc11_transition_count_analysis:
    input:
        inference=crc11_fit_file,
        best_schedule=crc11_fit_schedule_file,
        nodes=crc11_nodes_file,
        edges=crc11_edges_file,
        trees=f"{crc11_input_dir}/trees.txt",
        transition_edges=crc11_transition_edges_file,
        transition_times=f"{crc11_input_dir}/transition_times.txt",
        code=[
            "src/analyses/crc11/transition_count_analysis.py",
            "src/analyses/crc11/schedule_search.py",
            "src/concert/optimize_state_specific_parameters.py",
            "src/concert/closed_form_state_specific_transitions.py",
            "src/concert/tree_csv.py",
            "src/concert/tree_utils.py",
        ],
    output:
        results=directory(crc11_transition_count_data_dir),
        summary=directory(crc11_transition_count_summary_dir),
    params:
        python=sys.executable,
        input_dir=crc11_input_dir,
        output_dir=crc11_transition_count_data_dir,
        summary_dir=crc11_transition_count_summary_dir,
        workers=4,
        seed=42,
    log:
        "logs/sgootr/CRC11/transition_count_analysis.log",
    shell:
        "{params.python:q} -m src.analyses.crc11.transition_count_analysis "
        "--input-dir {params.input_dir:q} "
        "--output-dir {params.output_dir:q} "
        "--summary-dir {params.summary_dir:q} "
        "--existing-two-count-inference {input.inference:q} "
        "--workers {params.workers} --seed {params.seed} "
        "> {log:q} 2>&1"


rule sgootr_crc11_manuscript_figure:
    input:
        inference=crc11_fit_file,
        selected_fit=crc11_selected_fit_dir,
        transition_count_summary=crc11_transition_count_summary_dir,
        code=[
            "src/analyses/crc11/figure_generator/__main__.py",
            "src/analyses/crc11/figure_generator/generate.py",
        ],
    output:
        directory(crc11_figure_dir),
    params:
        python=sys.executable,
        profiles_dir=f"{crc11_selected_fit_dir}/profiles/data",
        tree_segments=f"{crc11_selected_fit_dir}/figures/CRC11_state_segments.csv",
        count_likelihood=(
            f"{crc11_transition_count_summary_dir}/"
            "likelihood_by_ptln_transition_count.csv"
        ),
        output_dir=crc11_figure_dir,
    log:
        "logs/sgootr/CRC11/manuscript_figure.log",
    shell:
        "{params.python:q} -m src.analyses.crc11.figure_generator "
        "--inference {input.inference:q} "
        "--profiles-dir {params.profiles_dir:q} "
        "--tree-segments {params.tree_segments:q} "
        "--count-likelihood {params.count_likelihood:q} "
        "--output-dir {params.output_dir:q} > {log:q} 2>&1"


rule sgootr_crc11_results:
    input:
        search=crc11_search_data_dir,
        search_summary=crc11_search_summary_dir,
        fit=crc11_fit_file,
        selected_fit=crc11_selected_fit_dir,
        transition_count_data=crc11_transition_count_data_dir,
        transition_count_summary=crc11_transition_count_summary_dir,
        figure=crc11_figure_dir,


# Numerical convergence: CSTM and MTBD/BDT-CTMC
rule infer:
    output:
        out="data/simulation/bdt_ctmc/{setting}/inference.out"
    log:
        std="data/simulation/bdt_ctmc/{setting}/log.inference",
        err="data/simulation/bdt_ctmc/{setting}/log.err.inference",
    benchmark:
        "data/simulation/bdt_ctmc/{setting}/benchmark.inference",
    shell:
        "../phylodynamic-carta-c/src/bdt_ctmc.out --num-states 3 --num-trees 10 --zero-death --sampling-rate 0.5 "
        " --transition-edges-file ../phylodynamic-carta-c/src/transition_edges.txt  --one-by-delta-t 1000 "
        " --output data/simulation/bdt_ctmc/{wildcards.setting}/inference.out --log data/simulation/bdt_ctmc/{wildcards.setting}/inference.log"
        " data/simulation/bdt_ctmc/{wildcards.setting} "
        " > {log.std} 2> {log.err}"
