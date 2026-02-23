from math import exp
import pandas as pd
import networkx as nx


def calculate_likelihood_of_tree(
    birth_rates: dict,
    death_rates: dict,
    sampling_rate: float,
    transition_probabilities: pd.DataFrame,
    cell_lineage_T,
    diff_T,
    diff_T_root='A',
):
    
    states = list(diff_T.nodes())
    L = dict()
    U = dict()

    for state in states:
        U[(state, 0)] = 1 - sampling_rate
    
    for diff_node in nx.dfs_postorder_nodes(diff_T):
        diff_time = diff_T.nodes[diff_node]['diff_time']
        if diff_time is None: diff_time = 0
        else: diff_time = 1 - diff_time

        if diff_time > 0:
            U_t = 0
            children = list(diff_T.successors(diff_node))
            for child in children:
                child_node = diff_T.nodes[child]
                child_diff_time = child_node['diff_time']
                if child_diff_time is None: child_diff_time = 0
                else: child_diff_time = 1 - child_diff_time

                U_child = U[(child, child_diff_time)]

                C = (U_child - 1) / (U_child - (death_rates[child] / birth_rates[child]))
                
                U_child_s = 1 - (birth_rates[child] - death_rates[child]) / \
                    (birth_rates[child] * (1 - (1 / C) * 
                                           exp( -(birth_rates[child] - death_rates[child]) * (diff_time - child_diff_time) )))

                U_t += U_child_s * transition_probabilities.loc[diff_node, child]
            
            U[(diff_node, diff_time)] = U_t

    
    def calculate_likelihood_along_edge(u, v, t, state):
        key = (u, v, t, state)
        if key in L:
            return L[key]
    
        diff_time = diff_T.nodes[state]['diff_time']
        if diff_time is None: diff_time = 0
        else: diff_time = 1 - diff_time

        v_node = cell_lineage_T.nodes[v]
        v_t = v_node['T']
        
        if v_t < diff_time:  # Change of state within the edge
            
            U_t = U[(state, diff_time)]
            C = (U_t - 1) / (U_t - (death_rates[state] / birth_rates[state]))

            flow = exp( -(birth_rates[state] - death_rates[state]) * (t - diff_time)) * \
                    ( (C - 1) / 
                      (C - exp( -(birth_rates[state] - death_rates[state]) * (t - diff_time))) ) ** 2
            
            p = 0
            for child_state in diff_T.successors(state):
                p += transition_probabilities.loc[state, child_state] * calculate_likelihood_along_edge(u, v, diff_time, child_state)

            L[key] = flow * p

        else: # Edge ends with boundary condition
            
            U_t = U[(state, diff_time)]
            C = (U_t - 1) / (U_t - (death_rates[state] / birth_rates[state]))
            flow = exp( -(birth_rates[state] - death_rates[state]) * (t - v_t)) * \
                    ( (C - exp( -(birth_rates[state] - death_rates[state]) * (v_t - diff_time))) / 
                      (C - exp( -(birth_rates[state] - death_rates[state]) * (t - diff_time))) ) ** 2

            # print(u, v, '|', C, '|', t, v_t, '=>', flow)
            
            children = list(cell_lineage_T.successors(v))
            
            if len(children) == 0:  # Boundary condition: current time 
                v_state = v_node['type']
                if state == v_state: p = sampling_rate
                else: p = 0
            
            elif len(children) == 2:  # Boundary condition: birth event
                L1 = calculate_likelihood_along_edge(v, children[0], v_t, state)
                L2 = calculate_likelihood_along_edge(v, children[1], v_t, state)
                p = L1 * L2 * birth_rates[state]

                # print(birth_rates[state])
            
            else:
                raise Exception("#Children must be 0 or 2")
            
            L[key] = flow * p
        
        return L[key]


    for parent in nx.dfs_postorder_nodes(cell_lineage_T):
        node = cell_lineage_T.nodes[parent]
        node_time = node['T']
        node_state = node.get('type')
        children = list(cell_lineage_T.successors(parent))
        
        for child in children:
            if node_state is not None:
                calculate_likelihood_along_edge(parent, child, node_time, node_state)
            else:
                for state in states:
                    calculate_likelihood_along_edge(parent, child, node_time, state)
    
    root_node = 0
    root_child = next(cell_lineage_T.successors(root_node))
    return L[(0, root_child, 1.0, diff_T_root)]


def calculate_likelihood_2(
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
        # print(birth_rates[cell_lineage_T.nodes[node]['type']])
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

        # print(u, v, '|', Cks[v_type], '|', start, end, '=>', p)

        prob *= p
    
        # transition probs
        for i in range(len(type_path) - 1):
            cur_t = type_path[i]
            next_t = type_path[i+1]
            p = transition_probabilities.loc[cur_t, next_t]

            prob *= p

    return prob



def assign_states_to_internal_nodes(cell_lineage_T, diff_T):

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

            if 'type' in node_data: continue
            if 'state' in node_data: continue

            children = list(cell_lineage_T.successors(node_id))

            children_types = [cell_lineage_T.nodes[ch]['state'] 
                            for ch in children 
                            if 'state' in cell_lineage_T.nodes[ch]]
            
            children_types = [transform_map.get(t, t) for t in children_types]

            unique_types = list(set(children_types))

            if len(unique_types) > 1:
                # raise ValueError(f"Children have > 1 types {unique_types}")
                print(f"Children have > 1 types {unique_types}")
            
            cell_lineage_T.nodes[node_id]["state"] = unique_types[0]

        elif 'diff_time_rev' in node_data:

            children = list(diff_T.successors(node_id))
            for child in children:
                transform_map[child] = node_id
            
            for key, value in list(transform_map.items()):
                for child in children:
                    if child == value:
                        transform_map[key] = node_id
    
    return cell_lineage_T


def add_node_times(cell_lineage_T, root=0):
    distances = nx.single_source_dijkstra_path_length(cell_lineage_T, root, weight='weight')
    for node, dist in distances.items():
        cell_lineage_T.nodes[node]['T'] = 1 - dist
    
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

    calculate_likelihood_of_tree(
        birth_rates={'A': 1.2, 'B': 1.0, 'C': 1.2, 'D': 0.9, 'E': 0.8,},
        death_rates={'A': 0.2, 'B': 0.1, 'C': 0.2, 'D': 0.3, 'E': 0.3,}, 
        transition_probabilities=transition_probabilities,
        cell_lineage_T=cell_lineage_T, 
        diff_T=diff_T
    )
