from src.dataprocessor import load_nodes, load_edges
from src.tree_utils import prepare_tree
from src.closed_form_coord import calculate_log_likelihood_closed, calculate_Us

import torch
from pathlib import Path

def main():
    num_transitions_list = [5, 10, 25, 50, 100, 500, 1000, 10000]
    output_path = Path("outputs/supplementary/bdt_ctmc/tree_likelihoods_cstm.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for tree_id in range(1, 101):
            dir = f"data/simulation/bdt_ctmc/emperical_proof_data/tree_{tree_id:03d}"
            node_path = f"{dir}/nodes.csv"
            edge_path = f"{dir}/edges.csv"
            loaded_nodes = load_nodes(str(node_path))
            loaded_edges = load_edges(str(edge_path))
            tree = prepare_tree(loaded_nodes, loaded_edges)

            birth_rates = torch.as_tensor([7.0, 6.0])
            death_rates = torch.as_tensor([1.5, 0.5])
            sampling_rates = torch.as_tensor([0.5, 0.4])
            for num_transition_times in num_transitions_list:

                print(tree_id, num_transition_times)

                delta_t = 1 / num_transition_times
                transition_rates = torch.as_tensor([
                    [1 - 1.0 * delta_t , 1.0 * delta_t],
                    [0, 1],
                ])
                transition_times = torch.as_tensor([0, 1e-15] + [delta_t * i for i in range(1, num_transition_times)])

                print(transition_rates)
                print(transition_times)

                U_result = calculate_Us(birth_rates, death_rates, sampling_rates, transition_rates, transition_times, [1])
                calculated_log_likelihood = calculate_log_likelihood_closed(
                    tree, birth_rates, death_rates, transition_rates, sampling_rates, U_result.Cs, transition_times
                )

                print("Calculated log likelihood:", calculated_log_likelihood.root_loglikelihood)
                print(
                    f"{calculated_log_likelihood.root_loglikelihood:.15f}",
                    end=",",
                    file=f,
                    flush=True,
                )            
            print(file=f, flush=True)






if __name__ == "__main__":
    main()
