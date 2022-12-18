import numpy as np
import networkx as nx

def geometrical_congruence(annealer):
    adjacency_matrix = annealer.adjacency_matrix
    reference_distance = np.arccosh(annealer.distance)

    G = nx.Graph()
    u, v = np.where(adjacency_matrix)
    G.add_edges_from(np.stack([u, v]).T)
    #topological_shortest_paths = nx.shortest_path(G)

    nonadjacent = np.where(np.triu(~adjacency_matrix, k=1))
    cong_sum = 0

    for i, j in zip(*nonadjacent):
        paths = nx.all_shortest_paths(G, i, j)
        ptsp = np.mean(np.sum([reference_distance[path[:-1], path[1:]] for path in paths], axis=1))
        rd = reference_distance[i, j]
        cong_sum += rd / ptsp
    return cong_sum / len(nonadjacent[0])
                                      
def greedy_routing_efficiency(annealer):
    adjacency_matrix = annealer.adjacency_matrix
    nonadjacent = np.where(np.triu(~adjacency_matrix, k=1))
    gcn = annealer.gcn
    gd = annealer.gd
    distance = np.arccosh(annealer.distance)

    eff_sum = 0

    for j, i in zip(*nonadjacent):
        if gd[i, j] == j:
            reference_distance = distance[i, j]
            grp_distance = 0
            v = i
            while v != j:
                u = gcn[v,j]
                grp_distance += distance[u, v]
                v = u
            eff_sum += reference_distance / grp_distance

    for j, i in zip(*nonadjacent):
        if gd[i, j] == j:
            reference_distance = distance[i, j]
            grp_distance = 0
            v = i
            while v != j:
                u = gcn[v,j]
                grp_distance += distance[u, v]
                v = u
            eff_sum += reference_distance / grp_distance
    return eff_sum / len(nonadjacent[0]) / 2