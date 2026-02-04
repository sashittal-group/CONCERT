from math import exp
import pandas as pd
import networkx as nx

def calculate_likelihood(
    birth_rates: dict,
    death_rates: dict,
    transition_probabilities: pd.DataFrame,
    cell_lineage_T,
    diff_T,
    diff_T_root='A',
):
    prob = 1
    # lambdas
    internal_nodes = [n for n in cell_lineage_T.nodes 
                      if list(cell_lineage_T.successors(n)) and list(cell_lineage_T.predecessors(n))
                      ]

    for node in internal_nodes:
        prob *= birth_rates[cell_lineage_T.nodes[node]['type']]
    
    # edges
    # calculate end Es:
    end_Es = {}
    Cks = {}
    dfs_nodes = list(nx.dfs_postorder_nodes(diff_T, source=diff_T_root))
    for node in dfs_nodes:
        if diff_T.nodes[node]["diff_time_rev"] is None:
            end_Es[node] = 0
            Cks[node] = birth_rates[node] / death_rates[node]
        
        else:
            # calculate boundary Es for children
            children_Es = {}
            children = list(diff_T.successors(node))
            for child in children:
                numerator = (birth_rates[child] - death_rates[child])
                node_t = diff_T.nodes[node]["diff_time_rev"]
                child_t = diff_T.nodes[child].get("diff_time_rev", 0)
                if child_t is None: child_t = 0
                denominator = birth_rates[child] * (1 - 1 / Cks[child] * 
                    exp( -(birth_rates[child] - death_rates[child]) * 
                        (node_t - child_t)))
                E = 1 - numerator / denominator
                children_Es[child] = E

            total_Es = 0; total_transition_w = 0
            for child, E in children_Es.items():
                t_prb = transition_probabilities.loc[node, child]
                total_Es += E * t_prb
                total_transition_w += t_prb
            
            end_Es[node] = total_Es / total_transition_w
            Cks[node] = (end_Es[node] - 1) / (end_Es[node] - death_rates[node] / birth_rates[node])
    
    for edge in cell_lineage_T.edges:
        
        u, v = edge
        u_type = cell_lineage_T.nodes[u]["type"]
        v_type = cell_lineage_T.nodes[v]["type"]

        type_path = nx.shortest_path(diff_T, source=u_type, target=v_type)

        # back calculate Es and Cks
        fixed_Cks = {}
        fixed_end_Es = {}

        last_t = type_path[-1]

        fixed_end_Es[last_t] = end_Es[last_t]
        fixed_Cks[last_t] = (fixed_end_Es[last_t] - 1) / (fixed_end_Es[last_t] - death_rates[last_t] / birth_rates[last_t])

        for t, child in zip(type_path[-2::-1], type_path[::-1]):
            # print(curr, child)
            numerator = (birth_rates[child] - death_rates[child])
            node_t = diff_T.nodes[t]["diff_time_rev"]
            child_t = diff_T.nodes[child].get("diff_time_rev", 0)
            if child_t is None: child_t = 0
            denominator = birth_rates[child] * (1 - 1 / fixed_Cks[child] * 
                exp( -(birth_rates[child] - death_rates[child]) * 
                    (node_t - child_t)))
            E = 1 - numerator / denominator
            fixed_end_Es[t] = E
            fixed_Cks[t] = (fixed_end_Es[t] - 1) / (fixed_end_Es[t] - death_rates[t] / birth_rates[t])

        # calculate Probabilities
        start = cell_lineage_T.nodes[u]["T"]

        for t in type_path[:-1]:
            type_end = diff_T.nodes[t]['diff_time_rev']
            if type_end is None: type_end = 0

            birth_death_diff_rate = birth_rates[t] - death_rates[t]

            p = exp( -(birth_death_diff_rate) * (start - type_end)) * \
                (fixed_Cks[t] - 1) ** 2 / \
                (fixed_Cks[t] - exp( -(birth_death_diff_rate) * (start - type_end))) ** 2
            
            prob *= p
            start = type_end
    
        end = cell_lineage_T.nodes[v]["T"]
        type_end = diff_T.nodes[v_type]['diff_time_rev']
        if type_end is None: type_end = 0

        birth_death_diff_rate = birth_rates[v_type] - death_rates[v_type]
        p = exp( -(birth_death_diff_rate) * (start - end)) * \
            (Cks[v_type] - exp( -(birth_death_diff_rate) * (end - type_end))) ** 2 / \
            (Cks[v_type] - exp( -(birth_death_diff_rate) * (start - type_end))) ** 2

        # print(start, end, type_end, p)

        prob *= p
    
        # transition probs
        for i in range(len(type_path) - 1):
            cur_t = type_path[i]
            next_t = type_path[i+1]
            p = transition_probabilities.loc[cur_t, next_t]

            prob *= p

    return prob



def type_internal_nodes(cell_lineage_T, diff_T):

    lineage_nodes = list(cell_lineage_T.nodes(data=True))
    diff_nodes = list(diff_T.nodes(data=True))
    for d in diff_nodes:
        if d[1]['diff_time'] is not None:
            d[1]['diff_time_rev'] = 1 - d[1]['diff_time']
        else:
            d[1]['diff_time_rev'] = None

    combined_list = lineage_nodes + diff_nodes

    combined_list = [
        (n, d) for n, d in combined_list 
        if d.get('T') is not None or d.get('diff_time_rev') is not None
    ]
    combined_sorted = sorted(
        combined_list,
        key=lambda x: x[1].get('T', x[1].get('diff_time_rev'))
    )


    transform_map = {}

    for node in combined_sorted:
        node_id, node_data = node

        if 'T' in node_data:

            if 'type' in node_data:
                continue

            children = list(cell_lineage_T.successors(node_id))

            children_types = [cell_lineage_T.nodes[ch]['type'] 
                            for ch in children 
                            if 'type' in cell_lineage_T.nodes[ch]]
            
            children_types = [transform_map.get(t, t) for t in children_types]

            unique_types = list(set(children_types))

            if len(unique_types) > 1:
                raise ValueError(f"Children have > 1 types {unique_types}")
            
            cell_lineage_T.nodes[node_id]["type"] = unique_types[0]


        elif 'diff_time_rev' in node_data:

            children = list(diff_T.successors(node_id))
            for child in children:
                transform_map[child] = node_id
            
            for key, value in list(transform_map.items()):
                for child in children:
                    if child == value:
                        transform_map[key] = node_id
    
    return cell_lineage_T



if __name__ == "__main__":
        
    transition_prob = [
        [0, 0.5, 0.5, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0.5, 0.5],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ]
    types = ['A', 'B', 'C', 'D', 'E']
    transition_probabilities = pd.DataFrame(transition_prob, index=types, columns=types)

    calculate_likelihood(
        birth_rates={'A': 1.2, 'B': 1.0, 'C': 1.2, 'D': 0.9, 'E': 0.8,},
        death_rates={'A': 0.2, 'B': 0.1, 'C': 0.2, 'D': 0.3, 'E': 0.3,}, 
        transition_probabilities=transition_probabilities,
        cell_lineage_T=cell_lineage_T, 
        diff_T=diff_T
    )
