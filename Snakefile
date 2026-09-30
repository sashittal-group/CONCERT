num_batches = 10
trees_per_batch = 10
num_states = 2
seed_stride = 100
batch_indices = range(num_batches)
tree_indices = range(trees_per_batch)
setting_file = "setting.txt"
simulation_dir = "data/simulation/coord_diff/setting_0"
transition_edges_file = "src/transition_edges.txt"

rule all:
    input:
        expand(f"{simulation_dir}/batch_{{batch}}/inference.out", batch=batch_indices),
        # expand(
        #     f"{simulation_dir}/batch_{{batch}}/tree_{{tree}}/edges.csv",
        #     batch=batch_indices,
        #     tree=tree_indices,
        # )


rule simulate:
    output:
        nodes=f"{simulation_dir}/batch_{{batch}}/tree_{{tree}}/nodes.csv",
        edges=f"{simulation_dir}/batch_{{batch}}/tree_{{tree}}/edges.csv",
        T_fig=f"{simulation_dir}/batch_{{batch}}/tree_{{tree}}/T.png",
        T_extant_fig=f"{simulation_dir}/batch_{{batch}}/tree_{{tree}}/T_extant.png",
        transition_times=f"{simulation_dir}/batch_{{batch}}/tree_{{tree}}/transition_times.csv",
        transitions_with_edges=f"{simulation_dir}/batch_{{batch}}/tree_{{tree}}/transition_times_with_edges.csv",
    params:
        source_seed=lambda wildcards: (
            int(wildcards.batch) * trees_per_batch + int(wildcards.tree)
        ) * seed_stride,
        output_dir=lambda wildcards: (
            f"{simulation_dir}/batch_{wildcards.batch}/tree_{wildcards.tree}"
        ),
    log:
        std=f"{simulation_dir}/batch_{{batch}}/tree_{{tree}}/stdout.log",
        err=f"{simulation_dir}/batch_{{batch}}/tree_{{tree}}/stderr.log",
    benchmark:
        f"{simulation_dir}/batch_{{batch}}/tree_{{tree}}/benchmark.txt",
    shell:
        "python -m src.simulator {setting_file:q} {params.source_seed} {params.output_dir:q} "
        " > {log.std:q} 2> {log.err:q}"


rule infer_batch:
    input:
        nodes=lambda wildcards: expand(
            f"{simulation_dir}/batch_{wildcards.batch}/tree_{{tree}}/nodes.csv",
            tree=tree_indices,
        ),
        edges=lambda wildcards: expand(
            f"{simulation_dir}/batch_{wildcards.batch}/tree_{{tree}}/edges.csv",
            tree=tree_indices,
        ),
    output:
        out=f"{simulation_dir}/batch_{{batch}}/inference.out",
    params:
        tree_path_regex=lambda wildcards: (
            rf"{simulation_dir}/batch_{wildcards.batch}/tree_\d+/edges\.csv"
        ),
        num_trees=trees_per_batch,
        num_states=num_states,
    log:
        optimization=f"{simulation_dir}/batch_{{batch}}/inference.log",
        stdout=f"{simulation_dir}/batch_{{batch}}/inference.stdout.log",
        stderr=f"{simulation_dir}/batch_{{batch}}/inference.stderr.log",
    benchmark:
        f"{simulation_dir}/batch_{{batch}}/inference.benchmark",
    shell:
        "python -m src.main {params.tree_path_regex:q} "
        "--num-states {params.num_states} --num-trees {params.num_trees} "
        "--transition-edges-file {transition_edges_file:q} "
        "--output {output.out:q} --log {log.optimization:q} "
        " --device cpu --optimizer adam "
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
