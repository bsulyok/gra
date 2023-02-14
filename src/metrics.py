from typing import Dict, Optional, Callable, Union, Tuple, List
from operator import itemgetter
import random
import networkx as nx
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score, auc
from scipy.stats import spearmanr
from math import inf
from itertools import combinations
import numpy as np
from math import cos, cosh, sinh, acosh, pi

def native_disk_distance(r_1: float, theta_1: float, r_2: float, theta_2: float) -> float:
    if theta_1 == theta_2:
        return abs(r_1-r_2)
    return acosh(cosh(r_1)*cosh(r_2)-sinh(r_1)*sinh(r_2)*cos(pi-abs(pi-abs(theta_1-theta_2))))


def mapping_accuracy(
    G: nx.Graph,
    coords: Dict[int, Union[Tuple, List]],
    distance_function: Callable = native_disk_distance,
    exclude_neighbours: bool = True,
    frac: Optional[float] = None
) -> float:
    """Compute the mapping accuracy for a subset of vertices as distance sources. Mapping accuracy is the spearman's correlation between the metric distance and topological distance of node pairs"""
    targets = list(G) if frac is None else random.sample(list(G.nodes), int(len(G) * frac))
    shortest_path_lengths, metric_distances = [], []
    for target in targets:
        shortest_path_length = dict(nx.single_target_shortest_path_length(G, target))
        del shortest_path_length[target]
        shortest_path_lengths += list(shortest_path_length.values())
        metric_distances += list(map(lambda v: distance_function(*coords[v], *coords[target]), shortest_path_length))
    return spearmanr(shortest_path_lengths, metric_distances).correlation


def node_pair_sample(
    G: nx.Graph,
    frac: Optional[float] = None
) -> float:
    """Helper function for sampling node pairs from a graph."""
    if frac is None or frac == 1.0:
        for edge in combinations(G.nodes, r=2):
            yield edge
    else:
        pool = tuple(G.nodes)
        n = len(pool)
        all_combinations = n * (n-1) // 2
        indices = random.sample(range(all_combinations), int(all_combinations * frac))
        if n % 2 == 0:
            for index in indices:
                i, j = index // (n-1), index % (n-1) + 1
                if j <= i:
                    i, j = n - 1 - i, n - j
                yield pool[i], pool[j]
        else:
            for index in indices:
                i, j = index // n, index % n
                if j <= i:
                    i, j = n - 2 - i, n - 1 - j
                yield pool[i], pool[j]


def edge_prediction_precision(
    G: nx.Graph,
    coords: Dict[int, Union[Tuple, List]],
    distance_function: Callable = native_disk_distance,
    exclude_neighbours: bool = True,
    frac: Optional[float] = None
) -> float:
    """Generate a sample of node pairs in the provided graph, compute the distance of each node pair, assume smaller distances to be quality scores representing a higher probability of being connected, finally compute the AUROC for the data sample."""
    inverse_distance, connected = [], []
    for node_pair in node_pair_sample(G, frac=frac):
        inverse_distance.append(1 / distance_function(*coords[node_pair[0]], *coords[node_pair[1]]))
        connected.append(int(node_pair in G.edges))
    edges_in_sample = sum(connected)
    return sum(map(itemgetter(1), sorted(zip(inverse_distance, connected), reverse=True)[:edges_in_sample])) / edges_in_sample


def edge_prediction_auroc(
    G: nx.Graph,
    coords: Dict[int, Union[Tuple, List]],
    distance_function: Callable = native_disk_distance,
    exclude_neighbours: bool = True,
    frac: Optional[float] = None
) -> float:
    """Generate a sample of node pairs in the provided graph, compute the distance of each node pair, assume smaller distances to be quality scores representing a higher probability of being connected, finally compute the AUROC for the data sample."""
    inverse_distance, connected = [], []
    for node_pair in node_pair_sample(G, frac=frac):
        #inverse_distance.append(1 / distance_function(*coords[node_pair[0]], *coords[node_pair[1]]))
        #connected.append(int(node_pair in G.edges))
        inverse_distance.append(distance_function(*coords[node_pair[0]], *coords[node_pair[1]]))
        connected.append(int(node_pair not in G.edges))
    assert any(connected), 'None of the vertex pairs in the sample are connected!'
    return roc_auc_score(connected, inverse_distance)


def edge_prediction_auprc(
    G: nx.Graph,
    coords: Dict[int, Union[Tuple, List]],
    distance_function: Callable = native_disk_distance,
    exclude_neighbours: bool = True,
    frac: Optional[float] = None
) -> float:
    """Generate a sample of node pairs in the provided graph, compute the distance of each node pair, assume smaller distances to be quality scores representing a higher probability of being connected, finally compute the AUROC for the data sample."""
    inverse_distance, connected = [], []
    for node_pair in node_pair_sample(G, frac=frac):
        #inverse_distance.append(1 / distance_function(*coords[node_pair[0]], *coords[node_pair[1]]))
        #connected.append(int(node_pair in G.edges))
        inverse_distance.append(distance_function(*coords[node_pair[0]], *coords[node_pair[1]]))
        connected.append(int(node_pair not in G.edges))
    assert any(connected), 'None of the vertex pairs in the sample are connected!'
    return average_precision_score(connected, inverse_distance)


def greedy_routing_success_rate(
    G: nx.Graph,
    coords: Dict[int, Union[Tuple, List]],
    distance_function: Callable = native_disk_distance,
    exclude_neighbours: bool = True,
    frac: Optional[float] = None
) -> float:
    """Compute the simplest type of greedy routing score for a subset of vertices as greedy routing targets."""
    targets = list(G) if frac is None else random.sample(list(G.nodes), int(len(G) * frac))
    score, total_paths = 0, 0
    for target in targets:
        target_coords = coords[target]
        distance_from_target = {vertex: distance_function(*target_coords, *coords[vertex]) for vertex in G.nodes}
        reached = defaultdict(bool)
        reached[target] = True
        for vertex, vertex_distance in sorted(distance_from_target.items(), key=itemgetter(1)):
            is_reached = reached[vertex]
            if not is_reached and reached[min(G[vertex], key=lambda v: distance_from_target[v])]:
                reached[vertex] = is_reached = True
            unvisited_filter = lambda v: distance_from_target[v] > vertex_distance and v not in reached
            reached.update(dict.fromkeys(filter(unvisited_filter, G[vertex]), is_reached))
        if exclude_neighbours:
            score += sum(reached.values()) - 1 - len(G[target])
            total_paths += len(G) - 1 - G.degree(target)
        else:
            score += sum(reached.values()) - 1
            total_paths += len(G) - 1
    return score / total_paths


def greedy_routing_score(
    G: nx.Graph,
    coords: Dict[int, Union[Tuple, List]],
    distance_function: Callable = native_disk_distance,
    exclude_neighbours: bool = True,
    frac: Optional[float] = None,
) -> float:
    """Compute a weighted greedy routing score for a subset of vertices as greedy routing targets where the weight is the ratio of the topological length of shortest paths and the greedy routing paths."""
    targets = list(G) if frac is None else random.sample(list(G.nodes), int(len(G) * frac))
    score, total_paths = 0.0, 0
    for target in targets:
        shortest_path_length = dict(nx.single_target_shortest_path_length(G, target))
        target_coords = coords[target]
        distance_from_target = {vertex: distance_function(*target_coords, *coords[vertex]) for vertex in G.nodes}
        greedy_path_length = defaultdict(lambda: inf)
        greedy_path_length[target] = 0
        for vertex, vertex_distance in sorted(distance_from_target.items(), key=itemgetter(1)):
            gpl = greedy_path_length[vertex]
            if gpl == inf:
                closest_neighbour_gpl = greedy_path_length[min(G[vertex], key=lambda v: distance_from_target[v])]
                if closest_neighbour_gpl != inf:
                    greedy_path_length[vertex] = gpl = closest_neighbour_gpl + 1
            unvisited_filter = lambda v: distance_from_target[v] > vertex_distance and v not in greedy_path_length
            greedy_path_length.update(dict.fromkeys(filter(unvisited_filter, G[vertex]), gpl+1))

        del greedy_path_length[target]
        if exclude_neighbours:
            for vertex in G[target]:
                del greedy_path_length[vertex]
            total_paths += len(G) - 1 - G.degree(target)
        else:
            total_paths += len(G) - 1
        score += sum(shortest_path_length[v]/grd for v, grd in greedy_path_length.items())
    return score / total_paths


def greedy_routing_efficiency(
    G: nx.Graph,
    coords: Dict[int, Union[Tuple, List]],
    distance_function: Callable = native_disk_distance,
    exclude_neighbours: bool = True,
    frac: Optional[float] = None
) -> float:
    """Compute the greedy routing efficiency for a subset of vertices as greedy routing targets. The geometrical congruence is the weighted average of successful greedy routing paths where the weight is the ratio of geodesic distance and the summed length of hops along greedy paths"""
    targets = list(G) if frac is None else random.sample(list(G.nodes), int(len(G) * frac))
    score, total_paths = 0.0, 0
    for target in targets:
        target_coords = coords[target]
        distance_from_target = {vertex: distance_function(*target_coords, *coords[vertex]) for vertex in G.nodes}
        greedy_path_distance = defaultdict(lambda: inf)
        greedy_path_distance[target] = 0.0
        for vertex, vertex_distance in sorted(distance_from_target.items(), key=itemgetter(1)):
            gpd = greedy_path_distance[vertex]
            if gpd == inf:
                closest_neighbour = min(G[vertex], key=lambda v: distance_from_target[v])
                closest_neighbour_gpd = greedy_path_distance[min(G[vertex], key=lambda v: distance_from_target[v])]
                if (closest_neighbour_gpd:=greedy_path_distance[closest_neighbour]) != inf:
                    greedy_path_distance[vertex] = gpd = closest_neighbour_gpd + distance_function(*coords[vertex], *coords[closest_neighbour])
            for neighbour in filter(lambda v: distance_from_target[v] > vertex_distance and v not in greedy_path_distance, G[vertex]):
                if gpd != inf:
                    greedy_path_distance[neighbour] = gpd + distance_function(*coords[vertex], *coords[neighbour])
                else:
                    greedy_path_distance[neighbour] = inf

        del greedy_path_distance[target]
        if exclude_neighbours:
            for vertex in G[target]:
                del greedy_path_distance[vertex]
            total_paths += len(G) - 1 - G.degree(target)
        else:
            total_paths += len(G) - 1
        score += sum(distance_from_target[v]/grd for v, grd in greedy_path_distance.items())
    return score / total_paths


def geometrical_congruence(
    G: nx.Graph,
    coords: Dict[int, Union[Tuple, List]],
    distance_function: Callable = native_disk_distance,
    exclude_neighbours: bool = True,
    frac: Optional[float] = None
) -> float:
    """Compute the greedy routing efficiency for a subset of vertices as shortest path routing targets. The geometrical congruence is the weighted average of shortest paths where the weight is the ratio of geodesic distance and the summed length of hops along topological shortest paths"""
    dist = lambda u, v: distance_function(*coords[u], *coords[v])
    targets = list(G) if frac is None else random.sample(list(G.nodes), int(len(G) * frac))
    score, total_pairs = 0.0, 0
    for target in targets:
        visited = defaultdict(lambda: [0, 0.0])
        visited = {target: [1, 0.0]}
        next_level = {target: [1, 0.0]}
        while next_level:
            level, next_level = next_level, defaultdict(lambda: [0, 0.0])
            for vertex, (path_count, path_sum) in level.items():
                for neighbour in G[vertex]:
                    if neighbour not in visited:
                        next_level[neighbour][0] += path_count
                        next_level[neighbour][1] += path_sum + path_count * dist(neighbour, vertex)
            visited.update(next_level)
        del visited[target]
        if exclude_neighbours:
            for vertex in G[target]:
                del visited[vertex]
        total_pairs += len(visited)
        score += sum(dist(v, target) / ps * pc for v, (pc, ps) in visited.items())
    return score / total_pairs


def new_metric_aucdiff(
    G: nx.Graph,
    coords: Dict[int, Union[Tuple, List]],
    distance_function: Callable = native_disk_distance,
    exclude_neighbours: bool = True,
    frac: float = 1.0,
    ensemble_size: int = 10,
    threshold_resolution: int = 100
) -> float:
    distance, connected = [], []
    threshold_resolution = 100

    for node_pair in node_pair_sample(G, frac=frac):
        distance.append(distance_function(*coords[node_pair[0]], *coords[node_pair[1]]))
        connected.append(node_pair in G.edges)

    distance = np.array(distance)
    connected = np.array(connected)
    distance = (distance - min(distance)) / (max(distance) - min(distance))

    threshold = np.linspace(0, 1, threshold_resolution)
    null_model_distance = np.random.choice(distance, size=(sum(connected), ensemble_size))
    auc_embedding = auc(threshold, np.mean(distance[connected] < threshold[:, None], axis=1))
    auc_null_model = auc(threshold, np.mean(null_model_distance[:, :, None] < threshold, axis=(0, 1)))
    return auc_embedding - auc_null_model


def new_metric_kolmogorov_smirnov(
    G: nx.Graph,
    coords: Dict[int, Union[Tuple, List]],
    distance_function: Callable = native_disk_distance,
    exclude_neighbours: bool = True,
    frac: float = 1.0,
    ensemble_size: int = 10,
    threshold_resolution: int = 100
) -> float:
    distance, connected = [], []
    threshold_resolution = 100

    for node_pair in node_pair_sample(G, frac=frac):
        distance.append(distance_function(*coords[node_pair[0]], *coords[node_pair[1]]))
        connected.append(node_pair in G.edges)

    distance = np.array(distance)
    connected = np.array(connected)
    distance = (distance - min(distance)) / (max(distance) - min(distance))

    threshold = np.linspace(0, 1, threshold_resolution)
    null_model_distance = np.random.choice(distance, size=(sum(connected), ensemble_size))

    curve_embedding = np.mean(distance[connected] < threshold[:, None], axis=1)
    curve_null_model = np.mean(null_model_distance[:, :, None] < threshold, axis=(0, 1))
    return max(curve_embedding - curve_null_model)


all_metrics = [
    'mapping_accuracy',
    'greedy_routing_success_rate',
    'greedy_routing_score',
    'greedy_routing_efficiency',
    'geometrical_congruence',
    'edge_prediction_auprc',
    'edge_prediction_auroc',
    'edge_prediction_precision',
    'new_metric_aucdiff',
    'new_metric_kolmogorov_smirnov'
]

def get_metrics(
    G: nx.Graph,
    coords: Dict[int, Union[Tuple, List]],
    requested_metrics: list = all_metrics,
    frac: float = 1.0,
    exclude_neighbours: bool = True,
    distance_function: Callable = native_disk_distance
):
    if not nx.is_connected(G):
        print('The provided graph is not connected, calculating metrics for the largest connected_component!')
        G = G.subgraph(max(nx.connected_components(G), key=len))
    metrics = {}
    for metric_name in requested_metrics:
        metrics[metric_name] = globals()[metric_name](
            G=G,
            coords=coords,
            frac=frac,
            distance_function=distance_function,
            exclude_neighbours=exclude_neighbours
        )
    return metrics