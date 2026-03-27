configfile: "config.yaml"

seeds = [ i for i in range(config["nseeds"])]
eids = [ i for i in range(1, 3311)]

import pandas as pd
paths = list(pd.read_csv("data/simulation/single/paths.csv")["path"])

param_ids = [ i for i in range(400)]
param_ids2 = [ i for i in range(324)]
gen_seeds = [ i for i in range(32) ]


rule all:
    input:
        # expand("data/simulation/tree_diff/T_{seed}/T_reconstructed.json", seed=seeds)
        # expand("data/simulation/tree_diff_params/T_{seed}/params.json", seed=seeds)
        # expand("outputs/icefase/commitment_bias_predictions_50/{eid}/q.csv", eid=eids)
        # expand("{path}/birth/inferred_parameters.json", path=paths)
        expand("data/simulation/linear/{param_id}/{seed}/T_extant.json", param_id=param_ids, seed=gen_seeds),
        # expand("data/simulation/tree/{param_id}/{seed}/T_extant.json", param_id=param_ids2, seed=gen_seeds),


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


rule icefase_commitment_bias:
    output:
        q="outputs/icefase/commitment_bias_predictions_50/{eid}/q.csv"
    log:
        std="outputs/icefase/commitment_bias_predictions_50/{eid}/log",
        err="outputs/icefase/commitment_bias_predictions_50/{eid}/err.log",
    benchmark:
        "outputs/icefase/commitment_bias_predictions_50/{eid}/benchmark",
    shell:
        "python -m src.simulation.icefase_commitment_bias --id {wildcards.eid} "
        " > {log.std} 2> {log.err}"


rule infer_single:
    output:
        params="{path}/birth/inferred_parameters.json"
    log:
        std="{path}/birth/log",
        err="{path}/birth/err.log",
    benchmark:
        "{path}/birth/benchmark",
    shell:
        "python -m src.simulation.infer_birth --path {wildcards.path} "
        " > {log.std} 2> {log.err}"
    

rule simulate_2:
    output:
        T_extant="data/simulation/linear/{param_id}/{seed}/T_extant.json"
    log:
        std="data/simulation/linear/{param_id}/{seed}/log",
        err="data/simulation/linear/{param_id}/{seed}/err.log",
    benchmark:
        "data/simulation/linear/{param_id}/{seed}/benchmark",
    shell:
        "python -m src.simulation.simulator_2 --param_id {wildcards.param_id} --seed {wildcards.seed} "
        " > {log.std} 2> {log.err} "


rule simulate_3:
    input:
        dir="data/simulation/tree/{param_id}"
    output:
        T_extant="data/simulation/tree/{param_id}/{seed}/T_extant.json"
    log:
        std="data/simulation/tree/{param_id}/{seed}/log",
        err="data/simulation/tree/{param_id}/{seed}/err.log",
    benchmark:
        "data/simulation/tree/{param_id}/{seed}/benchmark",
    shell:
        "python -m src.simulation.simulator_3 --input_dir {input.dir} --seed {wildcards.seed} "
        " > {log.std} 2> {log.err} "
