configfile: "config.yaml"

seeds = [ i for i in range(config["nseeds"])]
eids = [ i for i in range(1, 3311)]

import pandas as pd
paths = list(pd.read_csv("data/simulation/single/paths.csv")["path"])

param_ids = [ i for i in range(400)]
param_ids2 = [ i for i in range(324)]
param_ids3 = [ i for i in range(432)]
gen_seeds = [ i for i in range(32) ]


rule all:
    input:
        # expand("data/simulation/tree_diff/T_{seed}/T_reconstructed.json", seed=seeds)
        # expand("data/simulation/tree_diff_params/T_{seed}/params.json", seed=seeds)
        # expand("outputs/icefase/commitment_bias_predictions_50/{eid}/q.csv", eid=eids)
        # expand("{path}/birth/inferred_parameters.json", path=paths)
        # expand("data/simulation/linear/{param_id}/{seed}/T_extant.json", param_id=param_ids, seed=gen_seeds),
        # expand("data/simulation/tree/{param_id}/{seed}/T_extant.json", param_id=param_ids2, seed=gen_seeds),
        # expand("data/simulation/2_stage/{param_id}/{seed}/T_extant.json", param_id=param_ids3, seed=gen_seeds),
        # expand("data/simulation/linear/{param_id}/{seed}/birth_given_diff_times/inferred_parameters.json", param_id=param_ids, seed=gen_seeds),
        # expand("data/simulation/tree/{param_id}/{seed}/birth_given_diff_times/inferred_parameters.json", param_id=param_ids2, seed=gen_seeds),
        # expand("data/simulation/2_stage/{param_id}/{seed}/birth_given_diff_times/inferred_parameters.json", param_id=param_ids3, seed=gen_seeds),
        # [
        #     f"data/simulation/tree_comb/{param_id}/{seed}_{(seed+1)%32}/birth_given_diff_times/inferred_parameters.json"
        #     for param_id in param_ids2
        #     for seed in gen_seeds
        # ],
        # [
        #     f"data/simulation/tree_comb_4/{param_id}/{seed}_{(seed+1)%32}_{(seed+2)%32}_{(seed+3)%32}/birth_given_diff_times/inferred_parameters.json"
        #     for param_id in param_ids2
        #     for seed in gen_seeds
        # ],
        [
            f"data/simulation/tree_comb_8/{param_id}/{seed}_{(seed+1)%32}_{(seed+2)%32}_{(seed+3)%32}_{(seed+4)%32}_{(seed+5)%32}_{(seed+6)%32}_{(seed+7)%32}/birth_given_diff_times/inferred_parameters.json"
            for param_id in param_ids2
            for seed in gen_seeds
        ],
        [
            f"data/simulation/2_stage_comb_2/{param_id}/{seed}_{(seed+1)%32}/birth_given_diff_times/inferred_parameters.json"
            for param_id in param_ids3
            for seed in gen_seeds
        ],


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
    

rule simulate_4:
    input:
        dir="data/simulation/2_stage/{param_id}"
    output:
        T_extant="data/simulation/2_stage/{param_id}/{seed}/T_extant.json"
    log:
        std="data/simulation/2_stage/{param_id}/{seed}/log",
        err="data/simulation/2_stage/{param_id}/{seed}/err.log",
    benchmark:
        "data/simulation/2_stage/{param_id}/{seed}/benchmark",
    shell:
        "python -m src.simulation.simulator_3 --input_dir {input.dir} --seed {wildcards.seed} "
        " > {log.std} 2> {log.err} "


rule infer_params_given_diff_times:
    input:
        T_extant="data/simulation/linear/{param_id}/{seed}/T_extant.json"
    output:
        params="data/simulation/linear/{param_id}/{seed}/birth_given_diff_times/inferred_parameters.json"
    log:
        std="data/simulation/linear/{param_id}/{seed}/birth_given_diff_times/log",
        err="data/simulation/linear/{param_id}/{seed}/birth_given_diff_times/err.log",
    benchmark:
        "data/simulation/linear/{param_id}/{seed}/birth_given_diff_times/benchmark",
    shell:
        "python -m src.simulation.infer_parameters --path data/simulation/linear/{wildcards.param_id}/{wildcards.seed} "
        " > {log.std} 2> {log.err} "


rule infer_params_given_diff_times_2:
    input:
        T_extant="data/simulation/tree/{param_id}/{seed}/T_extant.json"
    output:
        params="data/simulation/tree/{param_id}/{seed}/birth_given_diff_times/inferred_parameters.json",
        inferred_q="data/simulation/tree/{param_id}/{seed}/birth_given_diff_times/inferred_q.csv",
    log:
        std="data/simulation/tree/{param_id}/{seed}/birth_given_diff_times/log",
        err="data/simulation/tree/{param_id}/{seed}/birth_given_diff_times/err.log",
    benchmark:
        "data/simulation/tree/{param_id}/{seed}/birth_given_diff_times/benchmark",
    shell:
        "python -m src.simulation.infer_parameters --path data/simulation/tree/{wildcards.param_id}/{wildcards.seed} "
        " > {log.std} 2> {log.err} "


rule infer_params_given_diff_times_3:
    input:
        T_extant="data/simulation/2_stage/{param_id}/{seed}/T_extant.json"
    output:
        params="data/simulation/2_stage/{param_id}/{seed}/birth_given_diff_times/inferred_parameters.json",
        inferred_q="data/simulation/2_stage/{param_id}/{seed}/birth_given_diff_times/inferred_q.csv"
    log:
        std="data/simulation/2_stage/{param_id}/{seed}/birth_given_diff_times/log",
        err="data/simulation/2_stage/{param_id}/{seed}/birth_given_diff_times/err.log",
    benchmark:
        "data/simulation/2_stage/{param_id}/{seed}/birth_given_diff_times/benchmark",
    shell:
        "python -m src.simulation.infer_parameters --path data/simulation/2_stage/{wildcards.param_id}/{wildcards.seed} "
        " --graph "
        " > {log.std} 2> {log.err} "


rule infer_params_two_trees:
    input:
        tree1="data/simulation/tree/{param_id}/{seed1}/T_extant.json",
        tree2="data/simulation/tree/{param_id}/{seed2}/T_extant.json"
    output:
        params="data/simulation/tree_comb/{param_id}/{seed1}_{seed2}/birth_given_diff_times/inferred_parameters.json"
    log:
        std="data/simulation/tree_comb/{param_id}/{seed1}_{seed2}/birth_given_diff_times/log",
        err="data/simulation/tree_comb/{param_id}/{seed1}_{seed2}/birth_given_diff_times/err.log"
    benchmark:
        "data/simulation/tree_comb/{param_id}/{seed1}_{seed2}/birth_given_diff_times/benchmark"
    shell:
        "python -m src.simulation.infer_parameters --paths {input} > {log.std} 2> {log.err}"


rule infer_params_four_trees:
    input:
        tree1="data/simulation/tree/{param_id}/{seed1}/T_extant.json",
        tree2="data/simulation/tree/{param_id}/{seed2}/T_extant.json",
        tree3="data/simulation/tree/{param_id}/{seed3}/T_extant.json",
        tree4="data/simulation/tree/{param_id}/{seed4}/T_extant.json",
    output:
        params_four_tree="data/simulation/tree_comb_4/{param_id}/{seed1}_{seed2}_{seed3}_{seed4}/birth_given_diff_times/inferred_parameters.json"
    log:
        std="data/simulation/tree_comb_4/{param_id}/{seed1}_{seed2}_{seed3}_{seed4}/birth_given_diff_times/log",
        err="data/simulation/tree_comb_4/{param_id}/{seed1}_{seed2}_{seed3}_{seed4}/birth_given_diff_times/err.log"
    benchmark:
        "data/simulation/tree_comb_4/{param_id}/{seed1}_{seed2}_{seed3}_{seed4}/birth_given_diff_times/benchmark"
    shell:
        "python -m src.simulation.infer_parameters --paths {input} > {log.std} 2> {log.err}"


rule infer_params_eight_trees:
    input:
        tree1="data/simulation/tree/{param_id}/{seed1}/T_extant.json",
        tree2="data/simulation/tree/{param_id}/{seed2}/T_extant.json",
        tree3="data/simulation/tree/{param_id}/{seed3}/T_extant.json",
        tree4="data/simulation/tree/{param_id}/{seed4}/T_extant.json",
        tree5="data/simulation/tree/{param_id}/{seed5}/T_extant.json",
        tree6="data/simulation/tree/{param_id}/{seed6}/T_extant.json",
        tree7="data/simulation/tree/{param_id}/{seed7}/T_extant.json",
        tree8="data/simulation/tree/{param_id}/{seed8}/T_extant.json",
    output:
        params_eight_tree="data/simulation/tree_comb_8/{param_id}/{seed1}_{seed2}_{seed3}_{seed4}_{seed5}_{seed6}_{seed7}_{seed8}/birth_given_diff_times/inferred_parameters.json"
    log:
        std="data/simulation/tree_comb_8/{param_id}/{seed1}_{seed2}_{seed3}_{seed4}_{seed5}_{seed6}_{seed7}_{seed8}/birth_given_diff_times/log",
        err="data/simulation/tree_comb_8/{param_id}/{seed1}_{seed2}_{seed3}_{seed4}_{seed5}_{seed6}_{seed7}_{seed8}/birth_given_diff_times/err.log"
    benchmark:
        "data/simulation/tree_comb_8/{param_id}/{seed1}_{seed2}_{seed3}_{seed4}_{seed5}_{seed6}_{seed7}_{seed8}/birth_given_diff_times/benchmark"
    shell:
        "python -m src.simulation.infer_parameters --paths {input} > {log.std} 2> {log.err}"


rule infer_2_stage_two_trees:
    input:
        tree1="data/simulation/2_stage/{param_id}/{seed1}/T_extant.json",
        tree2="data/simulation/2_stage/{param_id}/{seed2}/T_extant.json",
    output:
        params_2_stage_2_trees="data/simulation/2_stage_comb_2/{param_id}/{seed1}_{seed2}/birth_given_diff_times/inferred_parameters.json",
        inferred_q_2_stage_2_trees="data/simulation/2_stage_comb_2/{param_id}/{seed1}_{seed2}/birth_given_diff_times/inferred_q.csv",
    log:
        std="data/simulation/2_stage_comb_2/{param_id}/{seed1}_{seed2}/birth_given_diff_times/log",
        err="data/simulation/2_stage_comb_2/{param_id}/{seed1}_{seed2}/birth_given_diff_times/err.log",
    benchmark:
        "data/simulation/2_stage_comb_2/{param_id}/{seed1}_{seed2}/birth_given_diff_times/benchmark",
    shell:
        "python -m src.simulation.infer_parameters --paths {input} "
        " --graph "
        " > {log.std} 2> {log.err} "
