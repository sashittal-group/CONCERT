seeds = [ i for i in range(10)]
settings = [ i for i in range(1124)]
# settings = [ i for i in range(500)]

rule all:
    input:
        expand("data/simulation/bdt_ctmc/{setting}/edges_{seed}.csv", setting=settings, seed=seeds),
        # expand("data/simulation/bdt_ctmc/{setting}/inference.out", setting=settings)


rule simulate:
    output:
        edges="data/simulation/bdt_ctmc/{setting}/edges_{seed}.csv",
        nodes="data/simulation/bdt_ctmc/{setting}/nodes_{seed}.csv",
        T_fig="data/simulation/bdt_ctmc/{setting}/T_{seed}.png",
        # T_extant_fig="data/simulation/bdt_ctmc/{setting}/T_extant_{seed}.png",
    log:
        std="data/simulation/bdt_ctmc/{setting}/log.{seed}",
        err="data/simulation/bdt_ctmc/{setting}/log.err.{seed}",
    benchmark:
        "data/simulation/bdt_ctmc/{setting}/benchmark.{seed}",
    shell:
        "python -m src.simulator --setting {wildcards.setting} --seed {wildcards.seed} "
        " > {log.std} 2> {log.err}"


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
