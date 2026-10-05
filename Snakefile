num_batches = 10
trees_per_batch = 10
seed_stride = 100
batch_indices = range(num_batches)
tree_indices = range(trees_per_batch)
setting_root = "data/simulation/coord_diff"
setting_ids = glob_wildcards(f"{setting_root}/setting_{{setting}}/setting.txt").setting
transition_edges_file = "src/transition_edges.txt"


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
        expand(
            f"{setting_root}/setting_{{setting}}/batch_{{batch}}/inference.out",
            setting=setting_ids,
            batch=batch_indices,
        ),


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
        "--output {output.out:q} --log {log.optimization:q} "
        " --device cpu --optimizer adam --normalize-root-likelihood "
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
