configfile: "src/results/tls/config.yaml"
configfile: "src/results/sgootr/config.yaml"
configfile: "src/results/kptracer/config.yaml"

from pathlib import Path
import sys
import yaml
from src.generate_coord_settings import SAMPLING_RATE, TRANSITION_TIME

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
sgootr_profile_preprocessing_root = Path("src/results/sgootr/data")
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
sgootr_fixed_sampling_root = Path("src/results/sgootr/fixed_sampling_0.1")
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
    str(tls_fixed_sampling_paths["inference_root"] / dataset / "inference.out")
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
seed_stride = 100
batch_indices = range(num_batches)
tree_indices = range(trees_per_batch)
setting_root = "data/simulation/coord_diff"
setting_ids = glob_wildcards(f"{setting_root}/setting_{{setting}}/setting.txt").setting
transition_edges_file = "src/transition_edges.txt"
transition_times_file = "transition.txt"
kptracer_root = "data/KPTracer-Data/processed_data"
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
tls_fixed_sampling_transition_counts = (5, 10, 15, 20, 25, 30, 50, 100)
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
            / "inference.log"
        ),
        stdout=str(
            tls_fixed_sampling_paths["inference_root"]
            / "{dataset}"
            / "stdout.log"
        ),
        stderr=str(
            tls_fixed_sampling_paths["inference_root"]
            / "{dataset}"
            / "stderr.log"
        ),
    benchmark:
        str(
            tls_fixed_sampling_paths["inference_root"]
            / "{dataset}"
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


rule sgootr_profile_preprocess:
    input:
        settings="src/results/sgootr/config.yaml",
        code=[
            "src/results/sgootr/preprocess.py",
            "src/results/sgootr/data.py",
            "src/results/profile_tools.py",
            "src/results/tls/config.py",
            "src/results/tls/data.py",
            "src/results/tls/likelihood.py",
            "src/main2_given_transitions.py",
            "src/mle2_given_transitions.py",
            "src/closed_form_given_transitions.py",
            "src/tree_utils.py",
            "src/dataprocessor.py",
        ],
        inference=lambda wildcards: (
            f"data/sgootr/processed/{wildcards.dataset}/inference_result.txt"
        ),
        nodes=lambda wildcards: (
            f"data/sgootr/processed/{wildcards.dataset}/nodes.csv"
        ),
        edges=lambda wildcards: (
            f"data/sgootr/processed/{wildcards.dataset}/edges.csv"
        ),
        transition_times=lambda wildcards: (
            f"data/sgootr/processed/{wildcards.dataset}/transition_times.txt"
        ),
        transition_events=lambda wildcards: (
            f"data/sgootr/processed/{wildcards.dataset}/transition_times.csv"
        ),
    output:
        profiles=str(
            sgootr_profile_preprocessing_root
            / "{dataset}"
            / sgootr_profile_files["profiles"]
        ),
        ranges=str(
            sgootr_profile_preprocessing_root
            / "{dataset}"
            / sgootr_profile_files["ranges"]
        ),
        metadata=str(
            sgootr_profile_preprocessing_root
            / "{dataset}"
            / sgootr_profile_files["metadata"]
        ),
    params:
        python=sys.executable,
    shell:
        "{params.python:q} -m src.results.sgootr.preprocess "
        "--config {input.settings:q} --dataset {wildcards.dataset:q}"


rule sgootr_profile_plot:
    input:
        settings="src/results/sgootr/config.yaml",
        code=[
            "src/results/sgootr/plot.py",
            "src/results/tls/plot.py",
            "src/results/tls/config.py",
        ],
        ranges=expand(
            str(
                sgootr_profile_preprocessing_root
                / "{dataset}"
                / sgootr_profile_files["ranges"]
            ),
            dataset=sgootr_profile_datasets,
        ),
        metadata=expand(
            str(
                sgootr_profile_preprocessing_root
                / "{dataset}"
                / sgootr_profile_files["metadata"]
            ),
            dataset=sgootr_profile_datasets,
        ),
    output:
        sgootr_profile_plot_outputs,
    params:
        python=sys.executable,
    shell:
        "{params.python:q} -m src.results.sgootr.plot --config {input.settings:q}"


rule sgootr_fixed_sampling_01:
    input:
        sgootr_fixed_sampling_plot_outputs,


rule infer_sgootr_fixed_sampling_01:
    input:
        nodes="data/sgootr/processed/{dataset}/nodes.csv",
        edges="data/sgootr/processed/{dataset}/edges.csv",
        transition_times="data/sgootr/processed/{dataset}/transition_times.txt",
        per_edge_transition_times=(
            "data/sgootr/processed/{dataset}/transition_times.csv"
        ),
        code=[
            "src/main2_given_transitions.py",
            "src/mle2_given_transitions.py",
            "src/closed_form_coord2.py",
        ],
    output:
        out=(
            "src/results/sgootr/fixed_sampling_0.1/inference/"
            "{dataset}/inference.out"
        ),
    params:
        tree_directory="data/sgootr/processed/{dataset}",
        num_trees=1,
        num_states=3,
    log:
        optimization=(
            "src/results/sgootr/fixed_sampling_0.1/inference/"
            "{dataset}/optimizer.log"
        ),
        stdout=(
            "src/results/sgootr/fixed_sampling_0.1/inference/"
            "{dataset}/inference.stdout.log"
        ),
        stderr=(
            "src/results/sgootr/fixed_sampling_0.1/inference/"
            "{dataset}/inference.stderr.log"
        ),
    benchmark:
        "src/results/sgootr/fixed_sampling_0.1/inference/"
        "{dataset}/inference.benchmark",
    wildcard_constraints:
        dataset="|".join(sgootr_fixed_sampling_datasets),
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export DYLD_FALLBACK_LIBRARY_PATH=\"/opt/homebrew/opt/llvm/lib/c++\" && "
        "python -m src.main2_given_transitions {params.tree_directory:q} "
        "--num-states {params.num_states} --num-trees {params.num_trees} "
        "--transition-times-file {input.transition_times:q} "
        "--per-edge-transition-times-pattern "
        "{input.per_edge_transition_times:q} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer lbfgs --no-stem "
        "--fixed-shared-sampling-probability 0.1 "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule sgootr_fixed_sampling_profile_preprocess:
    input:
        settings=sgootr_fixed_sampling_config_path,
        inference=(
            "src/results/sgootr/fixed_sampling_0.1/inference/"
            "{dataset}/inference.out"
        ),
        nodes="data/sgootr/processed/{dataset}/nodes.csv",
        edges="data/sgootr/processed/{dataset}/edges.csv",
        transition_times="data/sgootr/processed/{dataset}/transition_times.txt",
        transition_events=(
            "data/sgootr/processed/{dataset}/transition_times.csv"
        ),
        code=[
            "src/results/sgootr/preprocess.py",
            "src/results/sgootr/data.py",
            "src/results/profile_tools.py",
            "src/results/tls/config.py",
            "src/results/tls/data.py",
            "src/results/tls/likelihood.py",
            "src/main2_given_transitions.py",
            "src/mle2_given_transitions.py",
            "src/closed_form_coord2.py",
            "src/tree_utils.py",
            "src/dataprocessor.py",
        ],
    output:
        profiles=(
            "src/results/sgootr/fixed_sampling_0.1/data/"
            "{dataset}/parameter_profiles.csv"
        ),
        ranges=(
            "src/results/sgootr/fixed_sampling_0.1/data/"
            "{dataset}/parameter_value_ranges.csv"
        ),
        metadata=(
            "src/results/sgootr/fixed_sampling_0.1/data/"
            "{dataset}/parameter_metadata.csv"
        ),
    params:
        python=sys.executable,
    wildcard_constraints:
        dataset="|".join(sgootr_fixed_sampling_datasets),
    shell:
        "{params.python:q} -m src.results.sgootr.preprocess "
        "--config {input.settings:q} --dataset {wildcards.dataset:q}"


rule sgootr_fixed_sampling_profile_plot:
    input:
        settings=sgootr_fixed_sampling_config_path,
        code=[
            "src/results/sgootr/plot.py",
            "src/results/tls/plot.py",
            "src/results/tls/config.py",
        ],
        ranges=expand(
            "src/results/sgootr/fixed_sampling_0.1/data/"
            "{dataset}/parameter_value_ranges.csv",
            dataset=sgootr_fixed_sampling_datasets,
        ),
        metadata=expand(
            "src/results/sgootr/fixed_sampling_0.1/data/"
            "{dataset}/parameter_metadata.csv",
            dataset=sgootr_fixed_sampling_datasets,
        ),
    output:
        sgootr_fixed_sampling_plot_outputs,
    params:
        python=sys.executable,
    shell:
        "{params.python:q} -m src.results.sgootr.plot --config {input.settings:q}"


rule kptracer_profile_preprocess:
    input:
        settings="src/results/kptracer/config.yaml",
        code=[
            "src/results/kptracer/preprocess.py",
            "src/results/kptracer/data.py",
            "src/results/profile_tools.py",
            "src/results/tls/config.py",
            "src/results/tls/data.py",
            "src/results/tls/likelihood.py",
            "src/main2_given_transitions.py",
            "src/mle2_given_transitions.py",
            "src/closed_form_given_transitions.py",
            "src/tree_utils.py",
            "src/dataprocessor.py",
        ],
        inference=f"{kptracer_root}/inference_per_edge.out",
        nodes=f"{kptracer_root}/nodes_0.csv",
        edges=f"{kptracer_root}/edges_0.csv",
        transitions=f"{kptracer_root}/transitions_0_adjusted.csv",
        transition_times=f"{kptracer_root}/transition-times.txt",
    output:
        profiles=str(
            kptracer_profile_preprocessing_root
            / "{dataset}"
            / kptracer_profile_files["profiles"]
        ),
        ranges=str(
            kptracer_profile_preprocessing_root
            / "{dataset}"
            / kptracer_profile_files["ranges"]
        ),
        metadata=str(
            kptracer_profile_preprocessing_root
            / "{dataset}"
            / kptracer_profile_files["metadata"]
        ),
    shell:
        "python -m src.results.kptracer.preprocess "
        "--config {input.settings:q} --dataset {wildcards.dataset:q}"


rule kptracer_profile_plot:
    input:
        settings="src/results/kptracer/config.yaml",
        code=[
            "src/results/kptracer/plot.py",
            "src/results/tls/plot.py",
            "src/results/tls/config.py",
        ],
        ranges=expand(
            str(
                kptracer_profile_preprocessing_root
                / "{dataset}"
                / kptracer_profile_files["ranges"]
            ),
            dataset=kptracer_profile_datasets,
        ),
        metadata=expand(
            str(
                kptracer_profile_preprocessing_root
                / "{dataset}"
                / kptracer_profile_files["metadata"]
            ),
            dataset=kptracer_profile_datasets,
        ),
    output:
        kptracer_profile_plot_outputs,
    shell:
        "python -m src.results.kptracer.plot --config {input.settings:q}"


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
        transition_count="5|10|15|20|25|30|50|100",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export DYLD_FALLBACK_LIBRARY_PATH=\"/opt/homebrew/opt/llvm/lib/c++\" && "
        "python -m src.continue_main2 {params.tree_directory:q} "
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
        transition_count="5|10|15|20|25|30|50|100",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export DYLD_FALLBACK_LIBRARY_PATH=\"/opt/homebrew/opt/llvm/lib/c++\" && "
        "python -m src.continue_main2 {params.tree_directory:q} "
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


rule infer_sgootr:
    input:
        nodes=f"{sgootr_root}/nodes.csv",
        edges=f"{sgootr_root}/edges.csv",
        transition_edges=f"{sgootr_root}/transition_edges.txt",
        transition_times=f"{sgootr_root}/transition-times.txt",
    output:
        out=f"{sgootr_root}/inference.out",
    params:
        tree_directory=sgootr_root,
        num_trees=1,
        num_states=3,
    log:
        optimization=f"{sgootr_root}/inference.log",
        stdout=f"{sgootr_root}/inference.stdout.log",
        stderr=f"{sgootr_root}/inference.stderr.log",
    benchmark:
        f"{sgootr_root}/inference.benchmark",
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
        # "--fixed-shared-sampling-probability 0.05 "
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


rule preprocess_sgootr_upgma_min_subtracted:
    input:
        tree="data/sgootr/processed/input/tree.nwk",
        code=[
            "src/process_data/sgootr/upgma.py",
            "src/process_data/sgootr/transitions.py",
            "src/process_data/sgootr/tree.py",
            "src/process_data/sgootr/executables.py",
        ],
    output:
        tree=f"{sgootr_upgma_min_subtracted_root}/tree.nwk",
        distance_matrix=f"{sgootr_upgma_min_subtracted_root}/distance_matrix.csv",
        nodes=f"{sgootr_upgma_min_subtracted_root}/nodes.csv",
        edges=f"{sgootr_upgma_min_subtracted_root}/edges.csv",
        transition_times=f"{sgootr_upgma_min_subtracted_root}/transition_times.csv",
        transition_times_text=f"{sgootr_upgma_min_subtracted_root}/transition_times.txt",
        plot=f"{sgootr_upgma_min_subtracted_root}/tree.png",
    params:
        python=sys.executable,
    shell:
        "{params.python:q} -m src.process_data.sgootr.upgma "
        "--input-tree {input.tree:q} "
        "--output-dir {sgootr_upgma_min_subtracted_root:q} "
        "--subtract-minimum-distance"


rule preprocess_sgootr_upgma_min_subtracted_height_1:
    input:
        tree="data/sgootr/processed/input/tree.nwk",
        code=[
            "src/process_data/sgootr/upgma.py",
            "src/process_data/sgootr/transitions.py",
            "src/process_data/sgootr/tree.py",
            "src/process_data/sgootr/executables.py",
        ],
    output:
        tree=f"{sgootr_upgma_min_subtracted_height_1_root}/tree.nwk",
        distance_matrix=f"{sgootr_upgma_min_subtracted_height_1_root}/distance_matrix.csv",
        nodes=f"{sgootr_upgma_min_subtracted_height_1_root}/nodes.csv",
        edges=f"{sgootr_upgma_min_subtracted_height_1_root}/edges.csv",
        transition_times=f"{sgootr_upgma_min_subtracted_height_1_root}/transition_times.csv",
        transition_times_text=f"{sgootr_upgma_min_subtracted_height_1_root}/transition_times.txt",
        plot=f"{sgootr_upgma_min_subtracted_height_1_root}/tree.png",
    params:
        python=sys.executable,
    shell:
        "{params.python:q} -m src.process_data.sgootr.upgma "
        "--input-tree {input.tree:q} "
        "--output-dir {sgootr_upgma_min_subtracted_height_1_root:q} "
        "--subtract-minimum-distance --normalize-height"


rule preprocess_sgootr_upgma_min_subtracted_height_1_transition_cut:
    input:
        tree=f"{sgootr_upgma_min_subtracted_height_1_root}/tree.nwk",
        transition_times=(
            f"{sgootr_upgma_min_subtracted_height_1_root}/transition_times.txt"
        ),
        transition_events=(
            f"{sgootr_upgma_min_subtracted_height_1_root}/transition_times.csv"
        ),
        code=[
            "src/process_data/sgootr/transition_cut_upgma.py",
            "src/process_data/sgootr/upgma.py",
            "src/process_data/sgootr/tree.py",
        ],
    output:
        tree=f"{sgootr_upgma_min_subtracted_height_1_transition_cut_root}/tree.nwk",
        distance_matrix=(
            f"{sgootr_upgma_min_subtracted_height_1_transition_cut_root}/"
            "subtree_pairwise_distances.csv"
        ),
        subtree_summary=(
            f"{sgootr_upgma_min_subtracted_height_1_transition_cut_root}/"
            "subtree_summary.csv"
        ),
        nodes=f"{sgootr_upgma_min_subtracted_height_1_transition_cut_root}/nodes.csv",
        edges=f"{sgootr_upgma_min_subtracted_height_1_transition_cut_root}/edges.csv",
        transition_times=(
            f"{sgootr_upgma_min_subtracted_height_1_transition_cut_root}/"
            "transition_times.csv"
        ),
        transition_times_text=(
            f"{sgootr_upgma_min_subtracted_height_1_transition_cut_root}/"
            "transition_times.txt"
        ),
        plot=f"{sgootr_upgma_min_subtracted_height_1_transition_cut_root}/tree.png",
    params:
        python=sys.executable,
        input_dir=sgootr_upgma_min_subtracted_height_1_root,
        output_dir=sgootr_upgma_min_subtracted_height_1_transition_cut_root,
    shell:
        "{params.python:q} -m src.process_data.sgootr.transition_cut_upgma "
        "--input-dir {params.input_dir:q} "
        "--output-dir {params.output_dir:q}"


rule infer_sgootr_transition_cut_single_state:
    input:
        nodes=f"{sgootr_upgma_min_subtracted_height_1_transition_cut_root}/nodes.csv",
        edges=f"{sgootr_upgma_min_subtracted_height_1_transition_cut_root}/edges.csv",
        transitions=(
            f"{sgootr_upgma_min_subtracted_height_1_transition_cut_root}/"
            "transition_times.csv"
        ),
        code=["src/sgootr_single_state.py"],
    output:
        inference=(
            f"{sgootr_upgma_min_subtracted_height_1_transition_cut_root}/"
            "single_state_inference.json"
        ),
        segments=(
            f"{sgootr_upgma_min_subtracted_height_1_transition_cut_root}/"
            "single_state_segments.csv"
        ),
    params:
        python=sys.executable,
    shell:
        "{params.python:q} -m src.sgootr_single_state "
        "--nodes {input.nodes:q} --edges {input.edges:q} "
        "--transitions {input.transitions:q} --output {output.inference:q} "
        "--estimate-sampling --max-birth-rate 100 --max-death-rate 50 "
        "--segments-output {output.segments:q}"


rule prepare_sgootr_per_edge_transitions:
    input:
        transitions="data/sgootr/CRC11_transitions.csv",
        nodes=f"{sgootr_root}/nodes.csv",
        edges=f"{sgootr_root}/edges.csv",
    output:
        transitions=f"{sgootr_root}/transitions_0_adjusted.csv",
        transition_times=f"{sgootr_root}/transition-times-per-edge.txt",
    shell:
        "python -m src.prepare_sgootr_transitions "
        "--transitions {input.transitions:q} "
        "--nodes {input.nodes:q} --edges {input.edges:q} "
        "--output {output.transitions:q} "
        "--transition-times-output {output.transition_times:q}"


rule infer_sgootr_per_edge:
    input:
        nodes=f"{sgootr_root}/nodes.csv",
        edges=f"{sgootr_root}/edges.csv",
        transition_edges=f"{sgootr_root}/transition_edges.txt",
        transition_times=f"{sgootr_root}/transition-times-per-edge.txt",
        per_edge_transition_times=f"{sgootr_root}/transitions_0_adjusted.csv",
    output:
        out=f"{sgootr_root}/inference_per_edge.out",
    params:
        tree_directory=sgootr_root,
        num_trees=1,
        num_states=3,
        per_edge_transition_times_pattern=lambda wildcards: (
            f"{sgootr_root}/transitions_{{index}}_adjusted.csv"
        ),
    log:
        optimization=f"{sgootr_root}/inference_per_edge.log",
        stdout=f"{sgootr_root}/inference_per_edge.stdout.log",
        stderr=f"{sgootr_root}/inference_per_edge.stderr.log",
    benchmark:
        f"{sgootr_root}/inference_per_edge.benchmark",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export DYLD_FALLBACK_LIBRARY_PATH=\"/opt/homebrew/opt/llvm/lib/c++\" && "
        "python -m src.main2_given_transitions {params.tree_directory:q} "
        "--num-states {params.num_states} --num-trees {params.num_trees} "
        "--transition-edges-file {input.transition_edges:q} "
        "--transition-times-file {input.transition_times:q} "
        "--per-edge-transition-times-pattern "
        "{params.per_edge_transition_times_pattern:q} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer adam --normalize-root-likelihood "
        "--shared-sampling-rate "
        # "--fixed-shared-sampling-probability 0.05 "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule infer_sgootr_transition_anchored:
    input:
        nodes=f"{sgootr_transition_anchored_root}/nodes.csv",
        edges=f"{sgootr_transition_anchored_root}/edges.csv",
        transition_times=f"{sgootr_transition_anchored_root}/transition_times.txt",
        per_edge_transition_times=f"{sgootr_transition_anchored_root}/transition_times.csv",
        code=[
            "src/main2_given_transitions.py",
            "src/mle2_given_transitions.py",
            "src/closed_form_coord2.py",
        ],
    output:
        out=f"{sgootr_transition_anchored_root}/inference_result.txt",
    params:
        tree_directory=sgootr_transition_anchored_root,
        num_trees=1,
        num_states=3,
    log:
        optimization=f"{sgootr_transition_anchored_root}/optimizer.log",
        stdout=f"{sgootr_transition_anchored_root}/inference.stdout.log",
        stderr=f"{sgootr_transition_anchored_root}/inference.stderr.log",
    benchmark:
        f"{sgootr_transition_anchored_root}/inference.benchmark",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export DYLD_FALLBACK_LIBRARY_PATH=\"/opt/homebrew/opt/llvm/lib/c++\" && "
        "python -m src.main2_given_transitions {params.tree_directory:q} "
        "--num-states {params.num_states} --num-trees {params.num_trees} "
        "--transition-times-file {input.transition_times:q} "
        "--per-edge-transition-times-pattern "
        "{input.per_edge_transition_times:q} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer lbfgs --no-stem "
        "--shared-sampling-rate "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule infer_sgootr_upgma_min_subtracted:
    input:
        nodes=f"{sgootr_upgma_min_subtracted_root}/nodes.csv",
        edges=f"{sgootr_upgma_min_subtracted_root}/edges.csv",
        transition_times=f"{sgootr_upgma_min_subtracted_root}/transition_times.txt",
        per_edge_transition_times=f"{sgootr_upgma_min_subtracted_root}/transition_times.csv",
        code=[
            "src/main2_given_transitions.py",
            "src/mle2_given_transitions.py",
            "src/closed_form_coord2.py",
        ],
    output:
        out=f"{sgootr_upgma_min_subtracted_root}/inference_result.txt",
    params:
        tree_directory=sgootr_upgma_min_subtracted_root,
        num_trees=1,
        num_states=3,
        python=sys.executable,
    log:
        optimization=f"{sgootr_upgma_min_subtracted_root}/optimizer.log",
        stdout=f"{sgootr_upgma_min_subtracted_root}/inference.stdout.log",
        stderr=f"{sgootr_upgma_min_subtracted_root}/inference.stderr.log",
    benchmark:
        f"{sgootr_upgma_min_subtracted_root}/inference.benchmark",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export DYLD_FALLBACK_LIBRARY_PATH=\"/opt/homebrew/opt/llvm/lib/c++\" && "
        "{params.python:q} -m src.main2_given_transitions "
        "{params.tree_directory:q} "
        "--num-states {params.num_states} --num-trees {params.num_trees} "
        "--transition-times-file {input.transition_times:q} "
        "--per-edge-transition-times-pattern "
        "{input.per_edge_transition_times:q} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer lbfgs --no-stem "
        "--shared-sampling-rate "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule infer_sgootr_upgma_min_subtracted_height_1:
    input:
        nodes=f"{sgootr_upgma_min_subtracted_height_1_root}/nodes.csv",
        edges=f"{sgootr_upgma_min_subtracted_height_1_root}/edges.csv",
        transition_times=f"{sgootr_upgma_min_subtracted_height_1_root}/transition_times.txt",
        per_edge_transition_times=f"{sgootr_upgma_min_subtracted_height_1_root}/transition_times.csv",
        code=[
            "src/main2_given_transitions.py",
            "src/mle2_given_transitions.py",
            "src/closed_form_coord2.py",
        ],
    output:
        out=f"{sgootr_upgma_min_subtracted_height_1_root}/inference_result.txt",
    params:
        tree_directory=sgootr_upgma_min_subtracted_height_1_root,
        num_trees=1,
        num_states=3,
        python=sys.executable,
    log:
        optimization=f"{sgootr_upgma_min_subtracted_height_1_root}/optimizer.log",
        stdout=f"{sgootr_upgma_min_subtracted_height_1_root}/inference.stdout.log",
        stderr=f"{sgootr_upgma_min_subtracted_height_1_root}/inference.stderr.log",
    benchmark:
        f"{sgootr_upgma_min_subtracted_height_1_root}/inference.benchmark",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export DYLD_FALLBACK_LIBRARY_PATH=\"/opt/homebrew/opt/llvm/lib/c++\" && "
        "{params.python:q} -m src.main2_given_transitions "
        "{params.tree_directory:q} "
        "--num-states {params.num_states} --num-trees {params.num_trees} "
        "--transition-times-file {input.transition_times:q} "
        "--per-edge-transition-times-pattern "
        "{input.per_edge_transition_times:q} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer lbfgs --no-stem "
        "--shared-sampling-rate "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule infer_sgootr_upgma_min_subtracted_height_1_transition_cut:
    input:
        nodes=f"{sgootr_upgma_min_subtracted_height_1_transition_cut_root}/nodes.csv",
        edges=f"{sgootr_upgma_min_subtracted_height_1_transition_cut_root}/edges.csv",
        transition_times=(
            f"{sgootr_upgma_min_subtracted_height_1_transition_cut_root}/"
            "transition_times.txt"
        ),
        per_edge_transition_times=(
            f"{sgootr_upgma_min_subtracted_height_1_transition_cut_root}/"
            "transition_times.csv"
        ),
        code=[
            "src/main2_given_transitions.py",
            "src/mle2_given_transitions.py",
            "src/closed_form_coord2.py",
        ],
    output:
        out=(
            f"{sgootr_upgma_min_subtracted_height_1_transition_cut_root}/"
            "inference_result.txt"
        ),
    params:
        tree_directory=sgootr_upgma_min_subtracted_height_1_transition_cut_root,
        num_trees=1,
        num_states=3,
        python=sys.executable,
    log:
        optimization=(
            f"{sgootr_upgma_min_subtracted_height_1_transition_cut_root}/"
            "optimizer.log"
        ),
        stdout=(
            f"{sgootr_upgma_min_subtracted_height_1_transition_cut_root}/"
            "inference.stdout.log"
        ),
        stderr=(
            f"{sgootr_upgma_min_subtracted_height_1_transition_cut_root}/"
            "inference.stderr.log"
        ),
    benchmark:
        f"{sgootr_upgma_min_subtracted_height_1_transition_cut_root}/inference.benchmark",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export DYLD_FALLBACK_LIBRARY_PATH=\"/opt/homebrew/opt/llvm/lib/c++\" && "
        "{params.python:q} -m src.main2_given_transitions "
        "{params.tree_directory:q} "
        "--num-states {params.num_states} --num-trees {params.num_trees} "
        "--transition-times-file {input.transition_times:q} "
        "--per-edge-transition-times-pattern "
        "{input.per_edge_transition_times:q} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer lbfgs --no-stem "
        "--shared-sampling-rate "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule shift_sgootr_upgma_min_subtracted_transition_times:
    input:
        per_edge_transition_times=(
            f"{sgootr_upgma_min_subtracted_root}/transition_times.csv"
        ),
        transition_times=f"{sgootr_upgma_min_subtracted_root}/transition_times.txt",
        code=["src/process_data/sgootr/shift_transition_times.py"],
    output:
        per_edge_transition_times=(
            f"{sgootr_upgma_min_subtracted_eps_root}/transition_times.csv"
        ),
        transition_times=(
            f"{sgootr_upgma_min_subtracted_eps_root}/transition_times.txt"
        ),
    params:
        python=sys.executable,
        epsilon=1e-4,
    shell:
        "{params.python:q} -m src.process_data.sgootr.shift_transition_times "
        "--input-csv {input.per_edge_transition_times:q} "
        "--input-text {input.transition_times:q} "
        "--output-dir {sgootr_upgma_min_subtracted_eps_root:q} "
        "--epsilon {params.epsilon}"


rule infer_sgootr_upgma_min_subtracted_eps:
    input:
        nodes=f"{sgootr_upgma_min_subtracted_root}/nodes.csv",
        edges=f"{sgootr_upgma_min_subtracted_root}/edges.csv",
        transition_times=(
            f"{sgootr_upgma_min_subtracted_eps_root}/transition_times.txt"
        ),
        per_edge_transition_times=(
            f"{sgootr_upgma_min_subtracted_eps_root}/transition_times.csv"
        ),
        code=[
            "src/main2_given_transitions.py",
            "src/mle2_given_transitions.py",
            "src/closed_form_given_transitions.py",
            "src/mle_given_transitions.py",
            "src/tree_utils.py",
            "src/dataprocessor.py",
            "src/dtos.py",
            "src/process_data/sgootr/tree.py",
        ],
    output:
        out=f"{sgootr_upgma_min_subtracted_eps_root}/inference_result.txt",
    params:
        tree_directory=sgootr_upgma_min_subtracted_root,
        num_trees=1,
        num_states=3,
        python=sys.executable,
    log:
        optimization=f"{sgootr_upgma_min_subtracted_eps_root}/optimizer.log",
        stdout=f"{sgootr_upgma_min_subtracted_eps_root}/inference.stdout.log",
        stderr=f"{sgootr_upgma_min_subtracted_eps_root}/inference.stderr.log",
    benchmark:
        f"{sgootr_upgma_min_subtracted_eps_root}/inference.benchmark",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export DYLD_FALLBACK_LIBRARY_PATH=\"/opt/homebrew/opt/llvm/lib/c++\" && "
        "{params.python:q} -m src.main2_given_transitions "
        "{params.tree_directory:q} "
        "--num-states {params.num_states} --num-trees {params.num_trees} "
        "--transition-times-file {input.transition_times:q} "
        "--per-edge-transition-times-pattern "
        "{input.per_edge_transition_times:q} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer lbfgs --no-stem "
        "--shared-sampling-rate "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule infer_sgootr_birth_shifted:
    input:
        nodes=f"{sgootr_birth_shifted_root}/nodes.csv",
        edges=f"{sgootr_birth_shifted_root}/edges.csv",
        transition_times=f"{sgootr_birth_shifted_root}/transition_times.txt",
        per_edge_transition_times=f"{sgootr_birth_shifted_root}/transition_times.csv",
        code=[
            "src/main2_given_transitions.py",
            "src/mle2_given_transitions.py",
            "src/closed_form_coord2.py",
        ],
    output:
        out=f"{sgootr_birth_shifted_root}/inference_result.txt",
    params:
        tree_directory=sgootr_birth_shifted_root,
        num_trees=1,
        num_states=3,
    log:
        optimization=f"{sgootr_birth_shifted_root}/optimizer.log",
        stdout=f"{sgootr_birth_shifted_root}/inference.stdout.log",
        stderr=f"{sgootr_birth_shifted_root}/inference.stderr.log",
    benchmark:
        f"{sgootr_birth_shifted_root}/inference.benchmark",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export DYLD_FALLBACK_LIBRARY_PATH=\"/opt/homebrew/opt/llvm/lib/c++\" && "
        "python -m src.main2_given_transitions {params.tree_directory:q} "
        "--num-states {params.num_states} --num-trees {params.num_trees} "
        "--transition-times-file {input.transition_times:q} "
        "--per-edge-transition-times-pattern "
        "{input.per_edge_transition_times:q} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer lbfgs --no-stem "
        "--shared-sampling-rate "
        "> {log.stdout:q} 2> {log.stderr:q}"


rule process_kptracer:
    input:
        newick="data/laml/problin_1461.nwk",
        metadata="data/laml/KPTracer_meta.csv",
        refined_tree=f"{kptracer_root}/0-T-0.refined.tree",
        location_labels=f"{kptracer_root}/0-T-0.location.labeling",
        code=[
            "src/process_data/kptracer/pipeline.py",
            "src/process_data/kptracer/tree.py",
            "src/process_data/kptracer/transitions.py",
            "src/likelihood.py",
            "src/dataprocessor.py",
        ],
    output:
        tree_edges=f"{kptracer_root}/1461.tree",
        leaf_labels=f"{kptracer_root}/1461.labeling",
        nodes=f"{kptracer_root}/nodes_0.csv",
        edges=f"{kptracer_root}/edges_0.csv",
        transitions=f"{kptracer_root}/transitions_0.csv",
        adjusted_transitions=f"{kptracer_root}/transitions_0_adjusted.csv",
        transition_times=f"{kptracer_root}/transition-times.txt",
    shell:
        "python -m src.process_data.kptracer.pipeline "
        "--tree {input.newick:q} --metadata {input.metadata:q} "
        "--refined-tree {input.refined_tree:q} "
        "--location-labels {input.location_labels:q} "
        "--output-dir {kptracer_root:q}"


rule infer_kptracer:
    input:
        nodes=f"{kptracer_root}/nodes_0.csv",
        edges=f"{kptracer_root}/edges_0.csv",
        transition_edges=f"{kptracer_root}/transition-edges.txt",
        transition_times=f"{kptracer_root}/transition-times.txt",
    output:
        out=f"{kptracer_root}/inference.out",
    params:
        tree_directory=kptracer_root,
        num_trees=1,
        num_states=5,
    log:
        optimization=f"{kptracer_root}/inference.log",
        stdout=f"{kptracer_root}/inference.stdout.log",
        stderr=f"{kptracer_root}/inference.stderr.log",
    benchmark:
        f"{kptracer_root}/inference.benchmark",
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


rule infer_kptracer_per_edge:
    input:
        nodes=f"{kptracer_root}/nodes_0.csv",
        edges=f"{kptracer_root}/edges_0.csv",
        transition_edges=f"{kptracer_root}/transition-edges.txt",
        transition_times=f"{kptracer_root}/transition-times.txt",
        per_edge_transition_times=f"{kptracer_root}/transitions_0_adjusted.csv",
    output:
        out=f"{kptracer_root}/inference_per_edge.out",
    params:
        tree_directory=kptracer_root,
        num_trees=1,
        num_states=5,
        per_edge_transition_times_pattern=lambda wildcards: (
            f"{kptracer_root}/transitions_{{index}}_adjusted.csv"
        ),
    log:
        optimization=f"{kptracer_root}/inference_per_edge.log",
        stdout=f"{kptracer_root}/inference_per_edge.stdout.log",
        stderr=f"{kptracer_root}/inference_per_edge.stderr.log",
    benchmark:
        f"{kptracer_root}/inference_per_edge.benchmark",
    shell:
        "job_tmp=$(mktemp -d) && "
        "trap 'rm -rf \"$job_tmp\"' EXIT && "
        "mkdir -p \"$job_tmp/inductor-cache\" \"$job_tmp/tmp\" && "
        "export TORCHINDUCTOR_CACHE_DIR=\"$job_tmp/inductor-cache\" && "
        "export TMPDIR=\"$job_tmp/tmp\" && "
        "export DYLD_FALLBACK_LIBRARY_PATH=\"/opt/homebrew/opt/llvm/lib/c++\" && "
        "python -m src.main2_given_transitions {params.tree_directory:q} "
        "--num-states {params.num_states} --num-trees {params.num_trees} "
        "--transition-edges-file {input.transition_edges:q} "
        "--transition-times-file {input.transition_times:q} "
        "--per-edge-transition-times-pattern "
        "{params.per_edge_transition_times_pattern:q} "
        "--output {output.out:q} --log {log.optimization:q} "
        "--device cpu --optimizer adam --no-stem "
        "--shared-sampling-rate "
        "> {log.stdout:q} 2> {log.stderr:q}"


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
