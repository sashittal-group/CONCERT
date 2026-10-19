from pathlib import Path

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
tlsc_root = "data/troupe/TLSC"
tlsc_tree_root = f"{tlsc_root}/subtrees_kept_states"
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
        f"{kptracer_root}/inference_per_edge.out",


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
        "python -m src.simulator {input.setting_file:q} {params.source_seed} {params.output_dir:q} "
        " > {log.std:q} 2> {log.err:q}"


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
        "--shared-sampling-rate "
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
        "--device cpu --optimizer adam --normalize-root-likelihood "
        "--fixed-shared-sampling-probability 0.05"
        # "--shared-sampling-rate "
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
