configfile: "config.yaml"

seeds = [ i for i in range(config["nseeds"])]

rule all:
    input:
        # expand("data/simulation/tree_diff/T_{seed}/T_reconstructed.json", seed=seeds)
        expand("data/simulation/tree_diff_params/T_{seed}/params.json", seed=seeds)

rule simulate:
    output:
        T_reconstructed="data/simulation/tree_diff/T_{seed}/T_reconstructed.json",
        T="data/simulation/tree_diff/T_{seed}/T.json",
        # T_reconstructed_fig="data/simulation/tree_diff/T_{seed}/T_reconstructed.png",
        # T_fig="data/simulation/tree_diff/T_{seed}/T.png",
    params:
        output_folder="data/simulation/tree_diff/T_{seed}",
    log:
        std="data/simulation/tree_diff/T_{seed}/sim.log",
        err="data/simulation/tree_diff/T_{seed}/sim.err.log",
    benchmark:
        "data/simulation/tree_diff/T_{seed}/sim.benchmark",
    shell:
        "python -m src.simulator -s {wildcards.seed} -o {params.output_folder} "
        " > {log.std} 2> {log.err}"

rule find_params:
    input:
        T_reconstructed="data/simulation/tree_diff/T_{seed}/T_reconstructed.json",
    output:
        params="data/simulation/tree_diff_params/T_{seed}/params.json",
    log:
        std="data/simulation/tree_diff_params/T_{seed}/sim.log",
        err="data/simulation/tree_diff_params/T_{seed}/sim.err.log",
    benchmark:
        "data/simulation/tree_diff_params/T_{seed}/sim.benchmark",
    shell:
        "python -m src.simulation.find_parameters -i {input.T_reconstructed} -o {output.params} "
        " > {log.std} 2> {log.err}"
