configfile: "src/results/tls/config.yaml"
configfile: "src/results/sgootr/config.yaml"
configfile: "src/results/kptracer/config.yaml"

from pathlib import Path
import csv
import os
import shlex
import sys
import zlib
import yaml
from src.generate_coord_settings import SAMPLING_RATE, TRANSITION_TIME
from src.generate_beam_barcodes import read_tree
from src.results.kptracer.grouped_transition_schedules import (
    load_grouped_transition_schedules,
    read_fit_log_likelihood,
)
from src.coord_diff_top_batches import (
    LINEAR_TOP_FIFTY_BATCHES,
    LINEAR_TOP_TWENTY_BATCHES,
    STAR3_TOP_FIFTY_BATCHES,
    STAR3_TOP_TWENTY_BATCHES,
)

python_executable = sys.executable

tls_composite_config_dir = Path("src/results/tls")


def tls_composite_config_path(name):
    configured = Path(config["tls_composite"][name])
    if configured.is_absolute():
        return configured
    return (tls_composite_config_dir / configured).resolve()


tls_composite_config = config["tls_composite"]
tls_composite_data_root = tls_composite_config_path("data_root")
tls_composite_preprocessing_root = tls_composite_config_path("preprocessing_root")
tls_composite_figure_root = tls_composite_config_path("figure_root")
tls_composite_datasets = tls_composite_config["datasets"]
tls_composite_files = tls_composite_config["files"]
tls_composite_plotting = tls_composite_config["plotting"]
tls_composite_figure_basenames = [
    tls_composite_plotting["output_basename"],
    *tls_composite_plotting["parameter_output_basenames"].values(),
]
tls_composite_plot_outputs = [
    str(
        tls_composite_figure_root
        / f"{basename}.{extension}"
    )
    for basename in tls_composite_figure_basenames
    for extension in tls_composite_plotting["formats"]
]
tls_composite_leaf_count_table = str(
    tls_composite_preprocessing_root
    / tls_composite_plotting["leaf_state_counts_filename"]
)
tls_population_size_figure_basenames = [
    tls_composite_plotting["population_size_output_basename"],
    tls_composite_plotting["sampled_population_size_output_basename"],
]
tls_population_size_plot_outputs = [
    str(tls_composite_figure_root / f"{basename}.{extension}")
    for basename in tls_population_size_figure_basenames
    for extension in tls_composite_plotting["population_size_formats"]
]
sgootr_profile_config = config["sgootr_profiles"]
sgootr_profile_datasets = sgootr_profile_config["datasets"]
sgootr_profile_files = sgootr_profile_config["files"]
sgootr_profile_plotting = sgootr_profile_config["plotting"]
sgootr_profile_preprocessing_root = Path("outputs/sgootr/profiles")
sgootr_profile_config_dir = Path("src/results/sgootr")
sgootr_profile_figure_root = Path(sgootr_profile_config["figure_root"])
if not sgootr_profile_figure_root.is_absolute():
    sgootr_profile_figure_root = (
        sgootr_profile_config_dir / sgootr_profile_figure_root
    )
sgootr_profile_figure_basenames = [
    sgootr_profile_plotting["output_basename"],
    *sgootr_profile_plotting["parameter_output_basenames"].values(),
]
sgootr_profile_plot_outputs = [
    str(sgootr_profile_figure_root / f"{basename}.{extension}")
    for basename in sgootr_profile_figure_basenames
    for extension in sgootr_profile_plotting["formats"]
]
sgootr_fixed_sampling_config_path = (
    "src/results/sgootr/config_fixed_sampling_0.1.yaml"
)
with open(sgootr_fixed_sampling_config_path) as fixed_sampling_config_file:
    sgootr_fixed_sampling_config = yaml.safe_load(fixed_sampling_config_file)[
        "sgootr_profiles"
    ]
sgootr_fixed_sampling_root = Path("outputs/sgootr/fixed_sampling_0.1")
sgootr_fixed_sampling_datasets = sgootr_fixed_sampling_config["datasets"]
sgootr_fixed_sampling_files = sgootr_fixed_sampling_config["files"]
sgootr_fixed_sampling_plotting = sgootr_fixed_sampling_config["plotting"]
sgootr_fixed_sampling_figure_basenames = [
    sgootr_fixed_sampling_plotting["output_basename"],
    *sgootr_fixed_sampling_plotting["parameter_output_basenames"].values(),
]
sgootr_fixed_sampling_plot_outputs = [
    str(sgootr_fixed_sampling_root / "figures" / f"{basename}.{extension}")
    for basename in sgootr_fixed_sampling_figure_basenames
    for extension in sgootr_fixed_sampling_plotting["formats"]
]
kptracer_profile_config = config["kptracer_profiles"]
kptracer_profile_datasets = kptracer_profile_config["datasets"]
kptracer_profile_files = kptracer_profile_config["files"]
kptracer_profile_plotting = kptracer_profile_config["plotting"]
kptracer_profile_preprocessing_root = Path("src/results/kptracer/data")
kptracer_profile_config_dir = Path("src/results/kptracer")
kptracer_profile_figure_root = Path(kptracer_profile_config["figure_root"])
if not kptracer_profile_figure_root.is_absolute():
    kptracer_profile_figure_root = (
        kptracer_profile_config_dir / kptracer_profile_figure_root
    )
kptracer_profile_figure_basenames = [
    kptracer_profile_plotting["output_basename"],
    *kptracer_profile_plotting["parameter_output_basenames"].values(),
]
kptracer_profile_plot_outputs = [
    str(kptracer_profile_figure_root / f"{basename}.{extension}")
    for basename in kptracer_profile_figure_basenames
    for extension in kptracer_profile_plotting["formats"]
]
tls_fixed_sampling_config_path = Path(
    "src/results/tls/config_fixed_sampling_0.1.yaml"
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


def tls_composite_tree_files(dataset, filename):
    tree_root = tls_composite_data_root / dataset / "subtrees_kept_states"
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
tree_indices = range(trees_per_batch)
star_tree_indices = range(star_trees_per_batch)
setting_root = "data/simulation/coord_diff"
setting_ids = glob_wildcards(f"{setting_root}/setting_{{setting}}/setting.txt").setting
star_setting_root = f"{setting_root}/star"
star_setting_ids = glob_wildcards(
    f"{star_setting_root}/setting_{{setting}}/setting.txt"
).setting
star_transition_roots = {
    2: f"{setting_root}/star_2",
    3: f"{setting_root}/star_3",
}
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
def mach2_full_outputs(data_root):
    node_files = sorted(Path(data_root).glob("setting_*/batch_*/tree_*/nodes.csv"))
    if not node_files:
        raise ValueError(f"No simulated MACH2 input trees found under {data_root}")
    missing_edges = [
        nodes.with_name("edges.csv")
        for nodes in node_files
        if not nodes.with_name("edges.csv").is_file()
    ]
    if missing_edges:
        raise ValueError(f"Missing matching MACH2 tree edge files: {missing_edges}")
    return [str(nodes.parent / "mach2_full") for nodes in node_files]


star3_mach2_full_outputs = mach2_full_outputs(star_transition_roots[3])
linear_mach2_full_outputs = mach2_full_outputs(linear_transition_root)
star_transition_edges_file = "src/transition_edges_star.txt"
transition_edges_file = "src/transition_edges.txt"
transition_times_file = "transition.txt"


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
            "`python -m src.generate_star_transition_settings` first"
        )
    return expand(
        f"{root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/edges.csv",
        setting=setting_ids,
        batch=batch_indices,
        tree=star_tree_indices,
    )
kptracer_root = "data/KPTracer-Data/processed_data"
kptracer_sharptni_cotransmission_run_directory = Path(
    config.get(
        "kptracer_sharptni_cotransmission_run_directory",
        "sharp-tni/results/kptracer_sharptni_20261101_1118",
    )
).resolve()
kptracer_sharptni_cotransmission_max_steps = int(
    config.get("kptracer_sharptni_cotransmission_max_steps", 50000)
)
kptracer_sharptni_cotransmission_patience = int(
    config.get("kptracer_sharptni_cotransmission_patience", 500)
)
kptracer_grouped_transition_schedules_directory = (
    kptracer_sharptni_cotransmission_run_directory / "enumerated_solutions"
)
kptracer_grouped_transition_schedule_tables = [
    str(
        kptracer_grouped_transition_schedules_directory
        / f"gamma{gamma}_mixed_direction_transition_times.tsv"
    )
    for gamma in (21, 22)
]
kptracer_grouped_transition_schedules = load_grouped_transition_schedules(
    kptracer_grouped_transition_schedule_tables
)
kptracer_grouped_transition_schedule_ids = list(
    kptracer_grouped_transition_schedules
)
kptracer_grouped_transition_fits_directory = Path(
    config.get(
        "kptracer_grouped_transition_fits_directory",
        kptracer_sharptni_cotransmission_run_directory
        / "concert_initialized"
        / "grouped_schedule_fits",
    )
).resolve()
kptracer_grouped_transition_initial_parameters = str(
    Path(
        config.get(
            "kptracer_grouped_transition_initial_parameters",
            f"{kptracer_root}/inference_progressive_fixed_sampling_0.05_5_20_regenerated_grid_100/transition_count_10.out",
        )
    ).resolve()
)
kptracer_grouped_transition_max_steps = int(
    config.get("kptracer_grouped_transition_max_steps", 50000)
)
kptracer_grouped_transition_patience = int(
    config.get("kptracer_grouped_transition_patience", 500)
)
kptracer_grouped_transition_sampling_probability = float(
    config.get("kptracer_grouped_transition_sampling_probability", 0.05)
)
kptracer_grouped_transition_fit_outputs = [
    str(
        kptracer_grouped_transition_fits_directory
        / "fits"
        / f"{schedule_id}.out"
    )
    for schedule_id in kptracer_grouped_transition_schedule_ids
]
kptracer_grouped_transition_fit_summary = str(
    kptracer_grouped_transition_fits_directory / "fit_summary.tsv"
)
kptracer_progressive_grid_sizes = (100, 200, 500, 1000)
kptracer_progressive_stage_prefix = (
    "inference_progressive_fixed_sampling_0.05_20"
)
kptracer_progressive_outputs = [
    f"{kptracer_root}/{kptracer_progressive_stage_prefix}_grid_{grid_size}.{suffix}"
    for grid_size in kptracer_progressive_grid_sizes
    for suffix in ("out", "txt")
]
kptracer_progressive_checkpoint_outputs = [
    f"{kptracer_root}/{kptracer_progressive_stage_prefix}_grid_{grid_size}/"
    f"transition_count_{count}{suffix}"
    for grid_size in kptracer_progressive_grid_sizes
    for count in range(10, 21)
    for suffix in (".out", "_times.txt")
]
kptracer_progressive_outputs.extend(kptracer_progressive_checkpoint_outputs)
kptracer_progressive_optimizer_logs = [
    f"{kptracer_root}/{kptracer_progressive_stage_prefix}_grid_{grid_size}.log"
    for grid_size in kptracer_progressive_grid_sizes
]
kptracer_progressive_optimizer_logs.extend(
    f"{kptracer_root}/{kptracer_progressive_stage_prefix}_grid_{grid_size}/"
    f"transition_count_{count}.log"
    for grid_size in kptracer_progressive_grid_sizes
    for count in range(10, 20)
)
kptracer_progressive_stdout = (
    f"{kptracer_root}/{kptracer_progressive_stage_prefix}.stdout.log"
)
kptracer_progressive_stderr = (
    f"{kptracer_root}/{kptracer_progressive_stage_prefix}.stderr.log"
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
tls_transition_counts = (5, 10, 20, 50, 100)
tls_transition_inference_outputs = [
    f"{tls_root}/inference_transition_times_{count}.out"
    for count in tls_transition_counts
]
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
sgootr_root = "data/sgootr/inference"
sgootr_upgma_root = "data/sgootr/processed/upgma"
sgootr_upgma_min_subtracted_root = "data/sgootr/processed/upgma_min_subtracted"
sgootr_upgma_min_subtracted_height_1_root = (
    "data/sgootr/processed/upgma_min_subtracted_height_1"
)
sgootr_upgma_min_subtracted_height_1_transition_cut_root = (
    "data/sgootr/processed/"
    "upgma_min_subtracted_height_1_transition_cut_upgma"
)
sgootr_upgma_min_subtracted_eps_root = (
    "data/sgootr/processed/upgma_min_subtracted_eps_1e-4"
)
sgootr_transition_anchored_root = "data/sgootr/processed/transition_anchored"
sgootr_birth_shifted_root = "data/sgootr/processed/birth_shifted_min_branch"
tlsc_nodes_files = sorted(Path(tlsc_tree_root).rglob("nodes.csv"))
tlsc_tree_pairs = [
    (nodes_file, nodes_file.with_name("edges.csv"))
    for nodes_file in tlsc_nodes_files
    if nodes_file.with_name("edges.csv").is_file()
]
if len(tlsc_tree_pairs) != len(tlsc_nodes_files):
    raise ValueError(f"Some TLSC node files have no matching edges file in {tlsc_tree_root}")
tlsc_transition_times_file = "data/troupe/transition-times.txt"


def setting_num_states(setting):
    setting_file = f"{setting_root}/setting_{setting}/setting.txt"
    with open(setting_file) as settings:
        for line in settings:
            value = line.split("#", 1)[0].strip()
            if value:
                return int(value.split()[0])
    raise ValueError(f"No state count found in {setting_file}")


def generated_batch_inference_outputs(wildcards):
    if not setting_ids:
        raise FileNotFoundError(
            f"No setting files found under {setting_root}; "
            "run `python -m src.generate_coord_settings` first"
        )
    return expand(
        f"{setting_root}/setting_{{setting}}/batch_{{batch}}/"
        f"inference_transition_{TRANSITION_TIME}_sampling_{SAMPLING_RATE}.out",
        setting=setting_ids,
        batch=batch_indices,
    )


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


def star2_diversitree_tree_files(wildcards, filename):
    return star_diversitree_tree_files(2, wildcards, filename)


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


def star2_diversitree_output_files(wildcards):
    batch_pairs = sorted(
        {
            (nodes_file.parent.parent.parent.name, nodes_file.parent.parent.name)
            for nodes_file in Path(star_transition_roots[2]).glob(
                "setting_*/batch_*/tree_*/nodes.csv"
            )
        }
    )
    return [
        (
            f"{star_transition_roots[2]}/{setting}/{batch}/"
            "diversitree_musse/joint_musse_fit.txt"
        )
        for setting, batch in batch_pairs
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
        # expand(
        #     f"{setting_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/edges.csv",
        #     setting=setting_ids,
        #     batch=batch_indices,
        #     tree=tree_indices,
        # ),
        # expand(
        #     f"{setting_root}/setting_{{setting}}/batch_{{batch}}/inference.out",
        #     setting=setting_ids,
        #     batch=batch_indices,
        # ),
        # f"{tls_root}/inference.out",
        # f"{tlsc_root}/inference.out",
        # f"{kptracer_root}/inference_per_edge.out",
        # f"{sgootr_root}/inference_per_edge.out",
        # f"{sgootr_transition_anchored_root}/inference_result.txt",
        # f"{sgootr_birth_shifted_root}/inference_result.txt",
        # tls_transition_inference_outputs,
        tls_fixed_sampling_plot_outputs,
        # tls_composite_plot_outputs,
        # sgootr_profile_plot_outputs,
        # kptracer_profile_plot_outputs,


rule tls_tree_cell_state_counts:
    input:
        settings="src/results/tls/config.yaml",
        code=[
            "src/results/tls/leaf_counts.py",
            "src/results/tls/config.py",
        ],
        trees=expand(
            str(
                tls_composite_data_root
                / "{dataset}"
                / "processed_data"
                / "trees.pkl"
            ),
            dataset=tls_composite_datasets,
        ),
    output:
        tls_composite_leaf_count_table,
    params:
        python=sys.executable,
        mach2_executable_arg=mach2_executable_arg,
    shell:
        "{params.python:q} -m src.results.tls.leaf_counts "
        "--config {input.settings:q}"


rule tls_composite_preprocess:
    input:
        settings="src/results/tls/config.yaml",
        code=[
            "src/results/tls/preprocess.py",
            "src/results/profile_tools.py",
            "src/results/tls/data.py",
            "src/results/tls/likelihood.py",
            "src/results/tls/config.py",
            "src/mle2_given_transitions.py",
            "src/closed_form_given_transitions.py",
            "src/closed_form_coord2.py",
            "src/tree_utils.py",
            "src/dataprocessor.py",
        ],
        inference=lambda wildcards: (
            tls_composite_data_root / wildcards.dataset / "inference.out"
        ),
        nodes=lambda wildcards: tls_composite_tree_files(
            wildcards.dataset, "nodes.csv"
        ),
        edges=lambda wildcards: tls_composite_tree_files(
            wildcards.dataset, "edges.csv"
        ),
    output:
        profiles=str(
            tls_composite_preprocessing_root
            / "{dataset}"
            / tls_composite_files["profiles"]
        ),
        ranges=str(
            tls_composite_preprocessing_root
            / "{dataset}"
            / tls_composite_files["ranges"]
        ),
        metadata=str(
            tls_composite_preprocessing_root
            / "{dataset}"
            / tls_composite_files["metadata"]
        ),
    shell:
        "python -m src.results.tls.preprocess "
        "--config {input.settings:q} --dataset {wildcards.dataset:q}"


rule tls_composite_plot:
    input:
        settings="src/results/tls/config.yaml",
        code=[
            "src/results/tls/plot.py",
            "src/results/tls/config.py",
        ],
        ranges=expand(
            str(
                tls_composite_preprocessing_root
                / "{dataset}"
                / tls_composite_files["ranges"]
            ),
            dataset=tls_composite_datasets,
        ),
        metadata=expand(
            str(
                tls_composite_preprocessing_root
                / "{dataset}"
                / tls_composite_files["metadata"]
            ),
            dataset=tls_composite_datasets,
        ),
    output:
        tls_composite_plot_outputs,
    shell:
        "python -m src.results.tls.plot --config {input.settings:q}"


rule tls_population_size:
    input:
        settings="src/results/tls/config.yaml",
        code=[
            "src/results/tls/plot.py",
            "src/results/tls/leaf_counts.py",
            "src/results/tls/config.py",
        ],
        leaf_counts=tls_composite_leaf_count_table,
        sampling_ranges=expand(
            str(
                tls_composite_data_root
                / "figures"
                / "{dataset}"
                / "parameter_value_ranges.csv"
            ),
            dataset=tls_composite_datasets,
        ),
        sampling_metadata=expand(
            str(
                tls_composite_data_root
                / "figures"
                / "{dataset}"
                / "parameter_metadata.csv"
            ),
            dataset=tls_composite_datasets,
        ),
    output:
        tls_population_size_plot_outputs,
    shell:
        "python -m src.results.tls.plot --config {input.settings:q} "
        "--population-size-only"


rule tls_fixed_sampling_01:
    input:
        inference=tls_fixed_sampling_inference_outputs,
        profiles=tls_fixed_sampling_profile_outputs,
        plots=tls_fixed_sampling_plot_outputs,


rule tls_fixed_sampling_01_inference:
    input:
        nodes=lambda wildcards: tls_composite_tree_files(
            wildcards.dataset,
            "nodes.csv",
        ),
        edges=lambda wildcards: tls_composite_tree_files(
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
            tls_composite_data_root
            / wildcards.dataset
            / "subtrees_kept_states"
        ),
        num_trees=lambda wildcards: len(
            tls_composite_tree_files(wildcards.dataset, "nodes.csv")
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
        "python -m src.main2 {params.tree_directory:q} "
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
            "src/results/tls/preprocess.py",
            "src/results/profile_tools.py",
            "src/results/tls/data.py",
            "src/results/tls/likelihood.py",
            "src/results/tls/config.py",
            "src/mle2_given_transitions.py",
            "src/closed_form_given_transitions.py",
            "src/closed_form_coord2.py",
            "src/tree_utils.py",
            "src/dataprocessor.py",
        ],
        inference=str(
            tls_fixed_sampling_paths["inference_root"]
            / "{dataset}"
            / "fixed_sampling_0.1"
            / "inference.out"
        ),
        nodes=lambda wildcards: tls_composite_tree_files(
            wildcards.dataset,
            "nodes.csv",
        ),
        edges=lambda wildcards: tls_composite_tree_files(
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
        "python -m src.results.tls.preprocess "
        "--config {input.settings:q} --dataset {wildcards.dataset:q}"


rule tls_fixed_sampling_01_plot:
    input:
        settings=str(tls_fixed_sampling_config_path),
        code=[
            "src/results/tls/plot.py",
            "src/results/tls/config.py",
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
        "python -m src.results.tls.plot --config {input.settings:q}"


rule simulate:
    input:
        setting_file=f"{setting_root}/setting_{{setting}}/setting.txt",
    output:
        nodes=f"{setting_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/nodes.csv",
        edges=f"{setting_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/edges.csv",
        T_fig=f"{setting_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/T.png",
        T_extant_fig=f"{setting_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/T_extant.png",
        transition_times=f"{setting_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/transition_times.csv",
        transitions_with_edges=f"{setting_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/transition_times_with_edges.csv",
    params:
        source_seed=lambda wildcards: (
            int(wildcards.batch) * trees_per_batch + int(wildcards.tree)
        ) * seed_stride,
        output_dir=lambda wildcards: (
            f"{setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}/tree_{wildcards.tree}"
        ),
    log:
        std=f"{setting_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/stdout.log",
        err=f"{setting_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/stderr.log",
    benchmark:
        f"{setting_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/benchmark.txt",
    shell:
        "python -m src.simulator_coord {input.setting_file:q} {params.source_seed} {params.output_dir:q} "
        " > {log.std:q} 2> {log.err:q}"


rule generate_coord_trees:
    input:
        expand(
            f"{setting_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/edges.csv",
            setting=setting_ids,
            batch=batch_indices,
            tree=tree_indices,
        )


rule simulate_star:
    input:
        setting_file=f"{star_setting_root}/setting_{{setting}}/setting.txt",
    output:
        nodes=f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/nodes.csv",
        edges=f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/edges.csv",
        T_fig=f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/T.png",
        T_extant_fig=f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/T_extant.png",
        transition_times=f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/transition_times.csv",
        transitions_with_edges=f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/transition_times_with_edges.csv",
    params:
        source_seed=star_source_seed,
        output_dir=lambda wildcards: (
            f"{star_setting_root}/setting_{wildcards.setting}/"
            f"batch_{wildcards.batch}/tree_{wildcards.tree}"
        ),
        python=sys.executable,
    log:
        std=f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/stdout.log",
        err=f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/stderr.log",
    benchmark:
        f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/tree_{{tree}}/benchmark.txt",
    shell:
        "{params.python:q} -m src.simulator_coord {input.setting_file:q} "
        "{params.source_seed} {params.output_dir:q} "
        "> {log.std:q} 2> {log.err:q}"


rule simulate_star_trees:
    input:
        expand(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "tree_{tree}/edges.csv",
            setting=star_setting_ids,
            batch=batch_indices,
            tree=star_tree_indices,
        )


rule simulate_star_3:
    input:
        setting_file=f"{star_transition_roots[3]}/setting_{{setting}}/setting.txt",
        code="src/simulator_coord.py",
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
        "{params.python:q} -m src.simulator_coord {input.setting_file:q} "
        "{params.source_seed} {params.output_dir:q} "
        "> {log.std:q} 2> {log.err:q}"


rule simulate_star_trees_3:
    input:
        lambda wildcards: star_transition_tree_edge_inputs(3)


rule generate_linear_transition_settings:
    input:
        code="src/generate_linear_transition_settings.py",
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
        "{params.python:q} -m src.generate_linear_transition_settings "
        "--output-dir {params.output_dir:q} "
        "--num-settings {params.num_settings} --seed {params.seed}"


rule simulate_linear_transition_3:
    input:
        setting_file=(
            f"{linear_transition_root}/setting_{{setting}}/setting.txt"
        ),
        code="src/simulator_coord.py",
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
        "{params.python:q} -m src.simulator_coord {input.setting_file:q} "
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
            "src/run_mach2_simulated_tree.py",
            "src/process_data/sgootr/executables.py",
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
        "{params.python:q} -m src.run_mach2_simulated_tree "
        "--nodes {input.nodes:q} --edges {input.edges:q} "
        "--output-dir {output.mach2_dir:q} --threads {threads} "
        "--timelimit {params.timelimit} "
        "{params.mach2_executable_arg} > {log:q} 2>&1"


rule mach2_star_3_all:
    input:
        star3_mach2_full_outputs


rule mach2_linear_A_to_D_all:
    input:
        linear_mach2_full_outputs


rule mach2_all:
    input:
        star3_mach2_full_outputs,
        linear_mach2_full_outputs


star3_top_twenty_concert = [
    f"{star_transition_roots[3]}/setting_{setting}/batch_{batch}/"
    "inference_progressive_joint_first_8_trees.out"
    for setting, batch in STAR3_TOP_TWENTY_BATCHES
]
linear_top_twenty_concert = [
    f"{linear_transition_root}/setting_{setting}/batch_{batch}/"
    "inference_joint_coordinate_first_8_trees.out"
    for setting, batch in LINEAR_TOP_TWENTY_BATCHES
]
star3_top_twenty_diversitree = [
    f"{star_transition_roots[3]}/setting_{setting}/batch_{batch}/"
    "diversitree_musse_first8/joint_musse_fit.txt"
    for setting, batch in STAR3_TOP_TWENTY_BATCHES
]
linear_top_twenty_diversitree = [
    f"{linear_transition_root}/setting_{setting}/batch_{batch}/"
    "diversitree_musse_first8/joint_musse_fit.txt"
    for setting, batch in LINEAR_TOP_TWENTY_BATCHES
]
star3_top_twenty_beam = [
    f"{star_transition_roots[3]}/setting_{setting}/batch_{batch}/"
    f"tree_{tree}/beam_snakemake"
    for setting, batch in STAR3_TOP_TWENTY_BATCHES
    for tree in first_eight_tree_indices
]
linear_top_twenty_beam = [
    f"{linear_transition_root}/setting_{setting}/batch_{batch}/"
    f"tree_{tree}/beam_snakemake"
    for setting, batch in LINEAR_TOP_TWENTY_BATCHES
    for tree in linear_transition_first8_tree_indices
]
star3_top_twenty_mach2 = [
    f"{star_transition_roots[3]}/setting_{setting}/batch_{batch}/"
    f"tree_{tree}/mach2_full"
    for setting, batch in STAR3_TOP_TWENTY_BATCHES
    for tree in first_eight_tree_indices
]
linear_top_twenty_mach2 = [
    f"{linear_transition_root}/setting_{setting}/batch_{batch}/"
    f"tree_{tree}/mach2_full"
    for setting, batch in LINEAR_TOP_TWENTY_BATCHES
    for tree in linear_transition_first8_tree_indices
]


rule coord_diff_top_twenty_concert:
    input:
        star3=star3_top_twenty_concert,
        linear=linear_top_twenty_concert


rule coord_diff_top_twenty_diversitree:
    input:
        star3=star3_top_twenty_diversitree,
        linear=linear_top_twenty_diversitree


rule coord_diff_top_twenty_beam:
    input:
        star3=star3_top_twenty_beam,
        linear=linear_top_twenty_beam


rule coord_diff_top_twenty_mach2:
    input:
        star3=star3_top_twenty_mach2,
        linear=linear_top_twenty_mach2


rule coord_diff_top_twenty_batches_methods:
    input:
        concert=star3_top_twenty_concert + linear_top_twenty_concert,
        diversitree=(
            star3_top_twenty_diversitree + linear_top_twenty_diversitree
        ),
        beam=star3_top_twenty_beam + linear_top_twenty_beam,
        mach2=star3_top_twenty_mach2 + linear_top_twenty_mach2


star3_top_fifty_concert = [
    f"{star_transition_roots[3]}/setting_{setting}/batch_{batch}/"
    "inference_progressive_joint_first_8_trees.out"
    for setting, batch in STAR3_TOP_FIFTY_BATCHES
]
linear_top_fifty_concert = [
    f"{linear_transition_root}/setting_{setting}/batch_{batch}/"
    "inference_joint_coordinate_first_8_trees.out"
    for setting, batch in LINEAR_TOP_FIFTY_BATCHES
]
star3_top_fifty_diversitree = [
    f"{star_transition_roots[3]}/setting_{setting}/batch_{batch}/"
    "diversitree_musse_first8/joint_musse_fit.txt"
    for setting, batch in STAR3_TOP_FIFTY_BATCHES
]
linear_top_fifty_diversitree = [
    f"{linear_transition_root}/setting_{setting}/batch_{batch}/"
    "diversitree_musse_first8/joint_musse_fit.txt"
    for setting, batch in LINEAR_TOP_FIFTY_BATCHES
]
star3_top_fifty_beam = [
    f"{star_transition_roots[3]}/setting_{setting}/batch_{batch}/"
    f"tree_{tree}/beam_snakemake"
    for setting, batch in STAR3_TOP_FIFTY_BATCHES
    for tree in first_eight_tree_indices
]
linear_top_fifty_beam = [
    f"{linear_transition_root}/setting_{setting}/batch_{batch}/"
    f"tree_{tree}/beam_snakemake"
    for setting, batch in LINEAR_TOP_FIFTY_BATCHES
    for tree in linear_transition_first8_tree_indices
]
star3_top_fifty_mach2 = [
    f"{star_transition_roots[3]}/setting_{setting}/batch_{batch}/"
    f"tree_{tree}/mach2_full"
    for setting, batch in STAR3_TOP_FIFTY_BATCHES
    for tree in first_eight_tree_indices
]
linear_top_fifty_mach2 = [
    f"{linear_transition_root}/setting_{setting}/batch_{batch}/"
    f"tree_{tree}/mach2_full"
    for setting, batch in LINEAR_TOP_FIFTY_BATCHES
    for tree in linear_transition_first8_tree_indices
]


rule coord_diff_top_fifty_concert:
    input:
        star3=star3_top_fifty_concert,
        linear=linear_top_fifty_concert


rule coord_diff_top_fifty_diversitree:
    input:
        star3=star3_top_fifty_diversitree,
        linear=linear_top_fifty_diversitree


rule coord_diff_top_fifty_beam:
    input:
        star3=star3_top_fifty_beam,
        linear=linear_top_fifty_beam


rule coord_diff_top_fifty_mach2:
    input:
        star3=star3_top_fifty_mach2,
        linear=linear_top_fifty_mach2


rule coord_diff_top_fifty_batches_methods:
    input:
        concert=star3_top_fifty_concert + linear_top_fifty_concert,
        diversitree=(
            star3_top_fifty_diversitree + linear_top_fifty_diversitree
        ),
        beam=star3_top_fifty_beam + linear_top_fifty_beam,
        mach2=star3_top_fifty_mach2 + linear_top_fifty_mach2


rule run_diversitree_star_3_batch:
    input:
        nodes=lambda wildcards: ancient(
            star3_diversitree_tree_files(wildcards, "nodes.csv")
        ),
        edges=lambda wildcards: ancient(
            star3_diversitree_tree_files(wildcards, "edges.csv")
        ),
        code="src/run_diversitree_musse_batch.R",
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
        code="src/run_diversitree_musse_batch.R",
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
        code="src/run_diversitree_musse_batch.R",
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
        code="src/run_diversitree_musse_batch.R",
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
        code="src/generate_beam_barcodes.py",
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
        "{params.python:q} -m src.generate_beam_barcodes "
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
        template="src/beam_fixed_tree_template.xml",
        code=[
            "src/run_beam_fixed_tree.py",
            "src/generate_beam_barcodes.py",
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
        "{params.python:q} -m src.run_beam_fixed_tree "
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
        code="src/combine_beam_summaries.py",
    output:
        parameters=star3_beam_parameter_summary,
        status=star3_beam_run_status,
    params:
        python=sys.executable,
        root=star_transition_roots[3],
    shell:
        "{params.python:q} -m src.combine_beam_summaries "
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
        code="src/generate_beam_barcodes.py",
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
        "{params.python:q} -m src.generate_beam_barcodes "
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
        template="src/beam_fixed_tree_template.xml",
        code=[
            "src/run_beam_fixed_tree.py",
            "src/generate_beam_barcodes.py",
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
        "{params.python:q} -m src.run_beam_fixed_tree "
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


rule infer_star_3_joint_progressive_first8:
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
            "src/progressive_joint_transition_times.py",
            "src/main2_joint_transition_times.py",
            "src/main2.py",
            "src/dataprocessor.py",
            "src/mle_closed_form2.py",
            "src/closed_form_coord2.py",
            "src/tree_utils.py",
            "src/dtos.py",
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
        "{params.python:q} -m src.progressive_joint_transition_times "
        "--tree-directory {params.tree_directory:q} "
        "--num-states 4 --num-trees 8 "
        "--transition-edges-file {input.transition_edges:q} "
        "--num-transition-times 3 --max-time 1 --grid-size 1000 "
        "--random-seed {params.random_seed} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer adam --normalize-root-likelihood "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule infer_star_3_joint_progressive_first8_all:
    input:
        expand(
            f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/"
            "inference_progressive_joint_first_8_trees.out",
            setting=star_transition_setting_ids[3],
            batch=batch_indices,
        )


rule infer_star_3_joint_progressive_first32:
    input:
        nodes=lambda wildcards: expand(
            f"{star_transition_roots[3]}/setting_{wildcards.setting}/"
            "batch_{batch}/tree_{tree}/nodes.csv",
            batch=[wildcards.batch],
            tree=range(32),
        ),
        edges=lambda wildcards: expand(
            f"{star_transition_roots[3]}/setting_{wildcards.setting}/"
            "batch_{batch}/tree_{tree}/edges.csv",
            batch=[wildcards.batch],
            tree=range(32),
        ),
        transition_edges=star_transition_edges_file,
        code=[
            "src/progressive_joint_transition_times.py",
            "src/main2_joint_transition_times.py",
            "src/main2.py",
            "src/dataprocessor.py",
            "src/mle_closed_form2.py",
            "src/closed_form_coord2.py",
            "src/tree_utils.py",
            "src/dtos.py",
        ],
    output:
        out=(
            f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/"
            "inference_progressive_joint_first_32_trees.out"
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
            "inference_progressive_joint_first_32_trees.log"
        ),
        stdout=(
            f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/"
            "inference_progressive_joint_first_32_trees.stdout.log"
        ),
        stderr=(
            f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/"
            "inference_progressive_joint_first_32_trees.stderr.log"
        ),
    benchmark:
        f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/inference_progressive_joint_first_32_trees.benchmark",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && "
        "{params.python:q} -m src.progressive_joint_transition_times "
        "--tree-directory {params.tree_directory:q} "
        "--num-states 4 --num-trees 32 "
        "--transition-edges-file {input.transition_edges:q} "
        "--num-transition-times 3 --max-time 1 --grid-size 1000 "
        "--random-seed {params.random_seed} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer adam --normalize-root-likelihood "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule infer_star_3_joint_progressive_first32_all:
    input:
        expand(
            f"{star_transition_roots[3]}/setting_{{setting}}/batch_{{batch}}/"
            "inference_progressive_joint_first_32_trees.out",
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
        transition_edges="src/transition_edges_linear.txt",
        code=[
            "src/joint_state_specific_transition_times.py",
            "src/mle_state_specific_transitions.py",
            "src/closed_form_state_specific_transitions.py",
            "src/main2_joint_transition_times.py",
            "src/main2.py",
            "src/mle_closed_form2.py",
            "src/dataprocessor.py",
            "src/tree_utils.py",
            "src/dtos.py",
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
        "{params.python:q} -m src.joint_state_specific_transition_times "
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
        code="src/plot_linear_transition_inference_accuracy.py",
    output:
        plot=(
            "outputs/simulation/linear_A_to_D/figures/"
            "identifiable_parameters_ground_truth_vs_inferred.png"
        ),
    params:
        data_dir=linear_transition_root,
        python=sys.executable,
    shell:
        "{params.python:q} -m src.plot_linear_transition_inference_accuracy "
        "--data-dir {params.data_dir:q} --output {output.plot:q}"


rule extract_star_transition_times:
    input:
        setting_file=f"{star_setting_root}/setting_{{setting}}/setting.txt",
        code=["src/extract_star_transition_times.py", "src/simulator_coord.py"],
    output:
        f"{star_setting_root}/setting_{{setting}}/transition-times.txt",
    params:
        python=sys.executable,
    shell:
        "{params.python:q} -m src.extract_star_transition_times "
        "{input.setting_file:q} {output:q}"


rule infer_star_first4_all:
    input:
        expand(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_first_4_trees.out",
            setting=star_setting_ids,
            batch=batch_indices,
        )


rule infer_star_first4:
    input:
        nodes=lambda wildcards: expand(
            f"{star_setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}/"
            "tree_{tree}/nodes.csv",
            tree=range(4),
        ),
        edges=lambda wildcards: expand(
            f"{star_setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}/"
            "tree_{tree}/edges.csv",
            tree=range(4),
        ),
        transition_edges=star_transition_edges_file,
        transition_times=(
            f"{star_setting_root}/setting_{{setting}}/transition-times.txt"
        ),
    output:
        out=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_first_4_trees.out"
        ),
    params:
        tree_directory=lambda wildcards: (
            f"{star_setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}"
        ),
        num_trees=4,
        python=sys.executable,
    log:
        optimization=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_first_4_trees.log"
        ),
        stdout=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_first_4_trees.stdout.log"
        ),
        stderr=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_first_4_trees.stderr.log"
        ),
    benchmark:
        f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/inference_first_4_trees.benchmark",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && "
        "{params.python:q} -m src.main2 {params.tree_directory:q} "
        "--num-states 4 --num-trees {params.num_trees} "
        "--transition-edges-file {input.transition_edges:q} "
        "--transition-times-file {input.transition_times:q} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer adam --normalize-root-likelihood "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule infer_star_first8_all:
    input:
        expand(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_first_8_trees.out",
            setting=star_setting_ids,
            batch=batch_indices,
        )


rule infer_star_first8:
    input:
        nodes=lambda wildcards: expand(
            f"{star_setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}/"
            "tree_{tree}/nodes.csv",
            tree=range(8),
        ),
        edges=lambda wildcards: expand(
            f"{star_setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}/"
            "tree_{tree}/edges.csv",
            tree=range(8),
        ),
        transition_edges=star_transition_edges_file,
        transition_times=(
            f"{star_setting_root}/setting_{{setting}}/transition-times.txt"
        ),
    output:
        out=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_first_8_trees.out"
        ),
    params:
        tree_directory=lambda wildcards: (
            f"{star_setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}"
        ),
        num_trees=8,
        python=sys.executable,
    log:
        optimization=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_first_8_trees.log"
        ),
        stdout=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_first_8_trees.stdout.log"
        ),
        stderr=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_first_8_trees.stderr.log"
        ),
    benchmark:
        f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/inference_first_8_trees.benchmark",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && "
        "{params.python:q} -m src.main2 {params.tree_directory:q} "
        "--num-states 4 --num-trees {params.num_trees} "
        "--transition-edges-file {input.transition_edges:q} "
        "--transition-times-file {input.transition_times:q} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer adam --normalize-root-likelihood "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule infer_star_first32_all:
    input:
        expand(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_first_32_trees.out",
            setting=star_setting_ids,
            batch=batch_indices,
        )


rule infer_star_first32:
    input:
        nodes=lambda wildcards: expand(
            f"{star_setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}/"
            "tree_{tree}/nodes.csv",
            tree=range(32),
        ),
        edges=lambda wildcards: expand(
            f"{star_setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}/"
            "tree_{tree}/edges.csv",
            tree=range(32),
        ),
        transition_edges=star_transition_edges_file,
        transition_times=(
            f"{star_setting_root}/setting_{{setting}}/transition-times.txt"
        ),
    output:
        out=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_first_32_trees.out"
        ),
    params:
        tree_directory=lambda wildcards: (
            f"{star_setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}"
        ),
        num_trees=32,
        python=sys.executable,
    log:
        optimization=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_first_32_trees.log"
        ),
        stdout=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_first_32_trees.stdout.log"
        ),
        stderr=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_first_32_trees.stderr.log"
        ),
    benchmark:
        f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/inference_first_32_trees.benchmark",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && "
        "{params.python:q} -m src.main2 {params.tree_directory:q} "
        "--num-states 4 --num-trees {params.num_trees} "
        "--transition-edges-file {input.transition_edges:q} "
        "--transition-times-file {input.transition_times:q} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer adam --normalize-root-likelihood "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule infer_star_first64_all:
    input:
        expand(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_first_64_trees.out",
            setting=star_setting_ids,
            batch=batch_indices,
        )


rule infer_star_first64:
    input:
        nodes=lambda wildcards: expand(
            f"{star_setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}/"
            "tree_{tree}/nodes.csv",
            tree=range(64),
        ),
        edges=lambda wildcards: expand(
            f"{star_setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}/"
            "tree_{tree}/edges.csv",
            tree=range(64),
        ),
        transition_edges=star_transition_edges_file,
        transition_times=(
            f"{star_setting_root}/setting_{{setting}}/transition-times.txt"
        ),
    output:
        out=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_first_64_trees.out"
        ),
    params:
        tree_directory=lambda wildcards: (
            f"{star_setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}"
        ),
        num_trees=64,
        python=sys.executable,
    log:
        optimization=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_first_64_trees.log"
        ),
        stdout=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_first_64_trees.stdout.log"
        ),
        stderr=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_first_64_trees.stderr.log"
        ),
    benchmark:
        f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/inference_first_64_trees.benchmark",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && "
        "{params.python:q} -m src.main2 {params.tree_directory:q} "
        "--num-states 4 --num-trees {params.num_trees} "
        "--transition-edges-file {input.transition_edges:q} "
        "--transition-times-file {input.transition_times:q} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer adam --normalize-root-likelihood "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule infer_star_first4_joint_times_all:
    input:
        expand(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_joint_times_first_4_trees.out",
            setting=star_setting_ids,
            batch=batch_indices,
        )


rule infer_star_first4_joint_times:
    input:
        nodes=lambda wildcards: expand(
            f"{star_setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}/"
            "tree_{tree}/nodes.csv",
            tree=range(4),
        ),
        edges=lambda wildcards: expand(
            f"{star_setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}/"
            "tree_{tree}/edges.csv",
            tree=range(4),
        ),
        transition_edges=star_transition_edges_file,
        code=[
            "src/main2_joint_transition_times.py",
            "src/main2.py",
            "src/dataprocessor.py",
            "src/mle_closed_form2.py",
            "src/closed_form_coord2.py",
            "src/tree_utils.py",
            "src/dtos.py",
        ],
    output:
        out=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_joint_times_first_4_trees.out"
        ),
    params:
        tree_directory=lambda wildcards: (
            f"{star_setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}"
        ),
        random_seed=lambda wildcards: (
            int(wildcards.setting.split("_")[-1]) * 1000 + int(wildcards.batch)
        ),
        python=sys.executable,
    log:
        optimization=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_joint_times_first_4_trees.log"
        ),
        stdout=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_joint_times_first_4_trees.stdout.log"
        ),
        stderr=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_joint_times_first_4_trees.stderr.log"
        ),
    benchmark:
        f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/inference_joint_times_first_4_trees.benchmark",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && "
        "{params.python:q} -m src.main2_joint_transition_times "
        "{params.tree_directory:q} --num-states 4 --num-trees 4 "
        "--transition-edges-file {input.transition_edges:q} "
        "--num-transition-times 5 --max-time 1 --search-rounds 3 "
        "--random-seed {params.random_seed} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer adam --normalize-root-likelihood "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule infer_star_first8_joint_times_all:
    input:
        expand(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_joint_times_first_8_trees.out",
            setting=star_setting_ids,
            batch=batch_indices,
        )


rule infer_star_first8_joint_times:
    input:
        nodes=lambda wildcards: expand(
            f"{star_setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}/"
            "tree_{tree}/nodes.csv",
            tree=range(8),
        ),
        edges=lambda wildcards: expand(
            f"{star_setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}/"
            "tree_{tree}/edges.csv",
            tree=range(8),
        ),
        transition_edges=star_transition_edges_file,
        code=[
            "src/main2_joint_transition_times.py",
            "src/main2.py",
            "src/dataprocessor.py",
            "src/mle_closed_form2.py",
            "src/closed_form_coord2.py",
            "src/tree_utils.py",
            "src/dtos.py",
        ],
    output:
        out=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_joint_times_first_8_trees.out"
        ),
    params:
        tree_directory=lambda wildcards: (
            f"{star_setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}"
        ),
        random_seed=lambda wildcards: (
            int(wildcards.setting.split("_")[-1]) * 1000 + int(wildcards.batch)
        ),
        python=sys.executable,
    log:
        optimization=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_joint_times_first_8_trees.log"
        ),
        stdout=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_joint_times_first_8_trees.stdout.log"
        ),
        stderr=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_joint_times_first_8_trees.stderr.log"
        ),
    benchmark:
        f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/inference_joint_times_first_8_trees.benchmark",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && "
        "{params.python:q} -m src.main2_joint_transition_times "
        "{params.tree_directory:q} --num-states 4 --num-trees 8 "
        "--transition-edges-file {input.transition_edges:q} "
        "--num-transition-times 5 --max-time 1 --search-rounds 3 "
        "--random-seed {params.random_seed} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer adam --normalize-root-likelihood "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule infer_star_first32_joint_times_all:
    input:
        expand(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_joint_times_first_32_trees.out",
            setting=star_setting_ids,
            batch=batch_indices,
        )


rule infer_star_first32_joint_times:
    input:
        nodes=lambda wildcards: expand(
            f"{star_setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}/"
            "tree_{tree}/nodes.csv",
            tree=range(32),
        ),
        edges=lambda wildcards: expand(
            f"{star_setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}/"
            "tree_{tree}/edges.csv",
            tree=range(32),
        ),
        transition_edges=star_transition_edges_file,
        code=[
            "src/main2_joint_transition_times.py",
            "src/main2.py",
            "src/dataprocessor.py",
            "src/mle_closed_form2.py",
            "src/closed_form_coord2.py",
            "src/tree_utils.py",
            "src/dtos.py",
        ],
    output:
        out=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_joint_times_first_32_trees.out"
        ),
    params:
        tree_directory=lambda wildcards: (
            f"{star_setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}"
        ),
        random_seed=lambda wildcards: (
            int(wildcards.setting.split("_")[-1]) * 1000 + int(wildcards.batch)
        ),
        python=sys.executable,
    log:
        optimization=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_joint_times_first_32_trees.log"
        ),
        stdout=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_joint_times_first_32_trees.stdout.log"
        ),
        stderr=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_joint_times_first_32_trees.stderr.log"
        ),
    benchmark:
        f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/inference_joint_times_first_32_trees.benchmark",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && "
        "{params.python:q} -m src.main2_joint_transition_times "
        "{params.tree_directory:q} --num-states 4 --num-trees 32 "
        "--transition-edges-file {input.transition_edges:q} "
        "--num-transition-times 5 --max-time 1 --search-rounds 3 "
        "--random-seed {params.random_seed} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer adam --normalize-root-likelihood "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule infer_star_population_search_first8:
    input:
        nodes=lambda wildcards: expand(
            f"{star_setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}/"
            "tree_{tree}/nodes.csv",
            tree=range(8),
        ),
        edges=lambda wildcards: expand(
            f"{star_setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}/"
            "tree_{tree}/edges.csv",
            tree=range(8),
        ),
        initial_output=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_joint_times_first_8_trees.out"
        ),
        transition_edges=star_transition_edges_file,
        transition_times=(
            f"{star_setting_root}/setting_{{setting}}/transition-times.txt"
        ),
        code=[
            "src/population_transition_time_search.py",
            "src/main2_joint_transition_times.py",
            "src/main2.py",
            "src/dataprocessor.py",
            "src/mle_closed_form2.py",
            "src/closed_form_coord2.py",
            "src/tree_utils.py",
            "src/dtos.py",
        ],
    output:
        out=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "population_search_first_8_trees.json"
        ),
    params:
        tree_directory=lambda wildcards: (
            f"{star_setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}"
        ),
        random_seed=lambda wildcards: (
            int(wildcards.setting.split("_")[-1]) * 1000 + int(wildcards.batch)
        ),
        grid_resolutions="25 100 250 1000",
        stage_budgets="100 200 300 400",
        parent_cooldown=5,
        crossover_probability=0.5,
        max_steps=100,
        max_total_steps=300,
        patience=30,
        plateau_tolerance=0.0001,
        python=sys.executable,
    log:
        stdout=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "population_search_first_8_trees.stdout.log"
        ),
        stderr=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "population_search_first_8_trees.stderr.log"
        ),
        queue=(
            f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "population_search_first_8_trees.queue.jsonl"
        ),
    threads: 1
    benchmark:
        f"{star_setting_root}/setting_{{setting}}/batch_{{batch}}/population_search_first_8_trees.benchmark",
    shell:
        "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && "
        "{params.python:q} -m src.population_transition_time_search "
        "--tree-directory {params.tree_directory:q} --num-trees 8 "
        "--initial-output {input.initial_output:q} "
        "--transition-edges-file {input.transition_edges:q} "
        "--truth-times-file {input.transition_times:q} "
        "--grid-resolutions {params.grid_resolutions} "
        "--stage-budgets {params.stage_budgets} "
        "--parent-cooldown {params.parent_cooldown} "
        "--crossover-probability {params.crossover_probability} "
        "--max-steps {params.max_steps} "
        "--max-total-steps {params.max_total_steps} "
        "--patience {params.patience} "
        "--plateau-tolerance {params.plateau_tolerance} "
        "--random-seed {params.random_seed} "
        "--output {output.out:q} "
        "--queue-log {log.queue:q} "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule infer_batch:
    input:
        nodes=lambda wildcards: expand(
            f"{setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}/tree_{{tree}}/nodes.csv",
            tree=tree_indices,
        ),
        edges=lambda wildcards: expand(
            f"{setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}/tree_{{tree}}/edges.csv",
            tree=tree_indices,
        ),
        transition_times=transition_times_file,
    output:
        out=f"{setting_root}/setting_{{setting}}/batch_{{batch}}/inference.out",
    params:
        tree_directory=lambda wildcards: (
            f"{setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}"
        ),
        num_trees=trees_per_batch,
        num_states=lambda wildcards: setting_num_states(wildcards.setting),
    log:
        optimization=f"{setting_root}/setting_{{setting}}/batch_{{batch}}/inference.log",
        stdout=f"{setting_root}/setting_{{setting}}/batch_{{batch}}/inference.stdout.log",
        stderr=f"{setting_root}/setting_{{setting}}/batch_{{batch}}/inference.stderr.log",
    benchmark:
        f"{setting_root}/setting_{{setting}}/batch_{{batch}}/inference.benchmark",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export DYLD_FALLBACK_LIBRARY_PATH=\"/opt/homebrew/opt/llvm/lib/c++\" && "
        "python -m src.main2 {params.tree_directory:q} "
        "--num-states {params.num_states} --num-trees {params.num_trees} "
        "--transition-edges-file {transition_edges_file:q} "
        "--transition-times-file {input.transition_times:q} "
        "--output {output.out:q} --log {log.optimization:q} "
        " --device cpu --optimizer adam --normalize-root-likelihood "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule infer_generated_batches:
    input:
        generated_batch_inference_outputs,


rule infer_generated_batch:
    input:
        setting_file=f"{setting_root}/setting_{{setting}}/setting.txt",
        nodes=lambda wildcards: expand(
            f"{setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}/"
            "tree_{tree}/nodes.csv",
            tree=tree_indices,
        ),
        edges=lambda wildcards: expand(
            f"{setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}/"
            "tree_{tree}/edges.csv",
            tree=tree_indices,
        ),
        transition_edges=transition_edges_file,
    output:
        out=(
            f"{setting_root}/setting_{{setting}}/batch_{{batch}}/"
            f"inference_transition_{TRANSITION_TIME}_sampling_{SAMPLING_RATE}.out"
        ),
    params:
        tree_directory=lambda wildcards: (
            f"{setting_root}/setting_{wildcards.setting}/batch_{wildcards.batch}"
        ),
        num_trees=trees_per_batch,
        num_states=lambda wildcards: setting_num_states(wildcards.setting),
        transition_time=TRANSITION_TIME,
        sampling_rate=SAMPLING_RATE,
    log:
        optimization=(
            f"{setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_transition_sampling.log"
        ),
        stdout=(
            f"{setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_transition_sampling.stdout.log"
        ),
        stderr=(
            f"{setting_root}/setting_{{setting}}/batch_{{batch}}/"
            "inference_transition_sampling.stderr.log"
        ),
    benchmark:
        f"{setting_root}/setting_{{setting}}/batch_{{batch}}/"
        "inference_transition_sampling.benchmark",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export DYLD_FALLBACK_LIBRARY_PATH=\"/opt/homebrew/opt/llvm/lib/c++\" && "
        "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && "
        "printf '%s\\n' {params.transition_time:q} "
        "> \"$job_tmp/transition_times.txt\" && "
        "python -m src.main2 {params.tree_directory:q} "
        "--num-states {params.num_states} --num-trees {params.num_trees} "
        "--transition-edges-file {input.transition_edges:q} "
        "--transition-times-file \"$job_tmp/transition_times.txt\" "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer adam --normalize-root-likelihood "
        "--fixed-shared-sampling-probability {params.sampling_rate} "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule infer_tls:
    input:
        nodes=[str(nodes_file) for nodes_file, _ in tls_tree_pairs],
        edges=[str(edges_file) for _, edges_file in tls_tree_pairs],
        transition_edges=tls_transition_edges_file,
        transition_times=tls_transition_times_file,
    output:
        out=f"{tls_root}/inference.out",
    params:
        tree_directory=tls_tree_root,
        num_trees=len(tls_tree_pairs),
        num_states=3,
    log:
        optimization=f"{tls_root}/inference.log",
        stdout=f"{tls_root}/inference.stdout.log",
        stderr=f"{tls_root}/inference.stderr.log",
    benchmark:
        f"{tls_root}/inference.benchmark",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export DYLD_FALLBACK_LIBRARY_PATH=\"/opt/homebrew/opt/llvm/lib/c++\" && "
        "python -m src.main2 {params.tree_directory:q} "
        "--num-states {params.num_states} --num-trees {params.num_trees} "
        "--transition-edges-file {input.transition_edges:q} "
        "--transition-times-file {input.transition_times:q} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer adam --normalize-root-likelihood "
        # "--shared-sampling-rate "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule infer_tls_transition_count:
    input:
        nodes=[str(nodes_file) for nodes_file, _ in tls_tree_pairs],
        edges=[str(edges_file) for _, edges_file in tls_tree_pairs],
        transition_edges=tls_transition_edges_file,
        transition_times=lambda wildcards: (
            f"data/troupe/transition-times-{wildcards.transition_count}.txt"
        ),
    output:
        out=f"{tls_root}/inference_transition_times_{{transition_count}}.out",
    params:
        tree_directory=tls_tree_root,
        num_trees=len(tls_tree_pairs),
        num_states=3,
    log:
        optimization=(
            f"{tls_root}/inference_transition_times_"
            "{transition_count}.log"
        ),
        stdout=(
            f"{tls_root}/inference_transition_times_"
            "{transition_count}.stdout.log"
        ),
        stderr=(
            f"{tls_root}/inference_transition_times_"
            "{transition_count}.stderr.log"
        ),
    benchmark:
        f"{tls_root}/inference_transition_times_{{transition_count}}.benchmark",
    wildcard_constraints:
        transition_count="5|10|20|50|100",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export DYLD_FALLBACK_LIBRARY_PATH=\"/opt/homebrew/opt/llvm/lib/c++\" && "
        "python -m src.main2 {params.tree_directory:q} "
        "--num-states {params.num_states} --num-trees {params.num_trees} "
        "--transition-edges-file {input.transition_edges:q} "
        "--transition-times-file {input.transition_times:q} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer adam --normalize-root-likelihood "
        "> {log.stdout:q} 2> {log.stderr:q}"


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
        "{params.python:q} -m src.continue_main2 {params.tree_directory:q} "
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
        "{params.python:q} -m src.continue_main2 {params.tree_directory:q} "
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
        "python -m src.main2 {params.tree_directory:q} "
        "--num-states {params.num_states} --num-trees {params.num_trees} "
        "--transition-edges-file {input.transition_edges:q} "
        "--transition-times-file {input.transition_times:q} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer adam --normalize-root-likelihood "
        "--shared-sampling-rate "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule preprocess_sgootr_upgma:
    input:
        tree="data/sgootr/processed/input/tree.nwk",
        code=[
            "src/process_data/sgootr/upgma.py",
            "src/process_data/sgootr/transitions.py",
            "src/process_data/sgootr/tree.py",
            "src/process_data/sgootr/executables.py",
        ],
    output:
        tree=f"{sgootr_upgma_root}/tree.nwk",
        distance_matrix=f"{sgootr_upgma_root}/distance_matrix.csv",
        nodes=f"{sgootr_upgma_root}/nodes.csv",
        edges=f"{sgootr_upgma_root}/edges.csv",
        transition_times=f"{sgootr_upgma_root}/transition_times.csv",
        transition_times_text=f"{sgootr_upgma_root}/transition_times.txt",
        plot=f"{sgootr_upgma_root}/tree.png",
    params:
        python=sys.executable,
    shell:
        "{params.python:q} -m src.process_data.sgootr.upgma "
        "--input-tree {input.tree:q} --output-dir {sgootr_upgma_root:q}"


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
