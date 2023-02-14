from tqdm.notebook import tqdm
import numpy as np
from scipy.stats import truncnorm
from copy import deepcopy
from pathlib import Path
import time
import shutil
import sys
from matplotlib import pyplot as plt
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import scipy.sparse as sp
import warnings

def get_distance(coord: np.ndarray) -> np.ndarray:
    d_theta = np.pi - np.abs(np.pi - np.abs(coord[:, 1]-coord[:, np.newaxis, 1]))
    distance = np.cosh(coord[:, 0]) * np.cosh(coord[:, np.newaxis, 0])
    distance -= np.sinh(coord[:, 0]) * np.sinh(coord[:, np.newaxis, 0]) * np.cos(d_theta)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'invalid value encountered in arccosh')
        distance = np.arccosh(distance)
    np.fill_diagonal(distance, 0)
    return distance

def greedy_closest_neighbour(distance: np.ndarray, adjacency_list:np.ndarray) -> np.ndarray:
    N = len(distance)
    gcn = np.array([neighbours[np.argmin(distance[neighbours], axis=0)] for neighbours in adjacency_list])
    np.fill_diagonal(gcn, np.arange(N))
    return gcn

def greedy_destination(gcn: np.ndarray) -> np.ndarray:
    gd = gcn.copy()
    num_grs_iter = int(np.ceil(np.log2(len(gcn)-1)))
    arange = np.arange(gd.shape[1])
    for _ in range(num_grs_iter):
        gd = gd[gd, arange]
    return gd

def greedy_routing_score(gd: np.ndarray) -> float:
    N = len(gd)
    successful = np.sum(gd == np.arange(N))
    return (successful - N) / N / (N-1)

def random_moves(coord, moved_vertex, K = 5) -> np.ndarray:
    N = len(coord)
    r, theta = coord[moved_vertex]
    new_coord = np.tile(coord, (K, 1, 1))
    clip_min, clip_max = 1, 2*np.log(N)
    loc, scale = r, 1
    a, b = (clip_min - loc) / scale, (clip_max - loc) / scale
    new_coord[:, moved_vertex, 0] = truncnorm.rvs(a=a, b=b, loc=loc, scale=scale, size=K)
    new_coord[:, moved_vertex, 1] = np.random.normal(loc=theta, scale=np.pi/4, size=K) % (2*np.pi)
    return new_coord

def recalc_distances(new_coord, distance, moved_vertex):
    new_distance = np.tile(distance, (len(new_coord), 1, 1))
    d_theta = np. pi - np.abs(np. pi - np.abs(new_coord[:, moved_vertex, 1][:, None] - new_coord[:, :, 1]))
    dist = np.cosh(new_coord[:, moved_vertex, 0][:, None]) * np.cosh(new_coord[:, :, 0])
    dist -= np.sinh(new_coord[:, moved_vertex, 0][:, None]) * np.sinh(new_coord[:, :, 0]) * np.cos(d_theta)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'invalid value encountered in arccosh')
        dist = np.arccosh(dist)
    dist[:, moved_vertex] = 0
    new_distance[:, moved_vertex] = new_distance[:, :, moved_vertex] = dist
    return new_distance

def reclac_greedy_closest_neighbour(new_distance, adjacency_list):
    return np.array([greedy_closest_neighbour(new_dist, adjacency_list) for new_dist in new_distance])

def reclac_greedy_destination(new_gcn):
    return np.array([greedy_destination(new_gcn_) for new_gcn_ in new_gcn])

def reclac_routing_score(new_gd):
    return np.array([greedy_routing_score(new_gd_) for new_gd_ in new_gd])

def greedy_closest_neighbour_(adjacency_list: np.ndarray, distance: np.ndarray) -> np.ndarray:
    gcn = np.array([neighbours[np.argmin(distance[neighbours], axis=0)] for neighbours in adjacency_list])
    np.fill_diagonal(gcn, np.arange(distance.shape[0]))
    return gcn

def greedy_destination_(gcn: np.ndarray, num_grs_iter: int):
    arange = np.arange(gcn.shape[1])
    for _ in range(num_grs_iter):
        gcn = gcn[gcn, arange]
    return gcn

def drip_down(gcn: np.ndarray, gd: np.ndarray, distance: np.ndarray) -> np.ndarray:
    N = gcn.shape[0]
    source, target = np.where(gd != np.arange(N))
    next_hop = gcn[source, target]
    dest_source = gd[source, target]
    dest_next_hop = gd[next_hop, target]
    need_to_move = distance[dest_source, target] > distance[dest_next_hop, target]
    gd[source[need_to_move], target[need_to_move]] = gd[next_hop[need_to_move], target[need_to_move]]
    return gd

def get_clogging_count(adjacency_matrix, coord):
    distance = get_distance(coord)
    adjacency_list = np.array([np.where(neigh)[0] for neigh in adjacency_matrix], dtype=object)
    N = adjacency_matrix.shape[0]
    num_grs_iter = int(np.ceil(np.log2(N-1)))
    gcn = greedy_closest_neighbour_(adjacency_list, distance)
    gd = greedy_destination_(gcn.copy(), num_grs_iter)
    gd = drip_down(gcn, gd, distance)
    return np.bincount(gd.flatten()) - np.sum(gd == np.arange(N), axis=0)

def get_marker_size(adjacency_matrix, coord, marker_size_min=10, marker_size_scale=200):
    N = len(adjacency_matrix)
    N_pairs = N * (N-1) / 2
    clogging_count = get_clogging_count(adjacency_matrix, coord)
    marker_size = np.sqrt(clogging_count / N_pairs)
    marker_size = np.clip(marker_size*marker_size_scale, a_min=marker_size_min, a_max=None)
    return marker_size


def vertex_move_improvement_heatmap(path, moved_vertex, resolution=64):
    subplot_titles=[
        'a)                                                      ',
        'b)                                                      '
    ]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=subplot_titles,
    )

    adjacency_matrix = sp.load_npz(path / 'sparse_adjacency_matrix.npz').toarray()
    adjacency_list = np.array([np.where(neigh)[0] for neigh in adjacency_matrix], dtype=object)
    G = nx.from_numpy_array(adjacency_matrix)
    edges = np.array(list(G.edges))
    N = len(adjacency_matrix)

    for idx, coord_path in enumerate([path / 'initial_coords.csv', path / 'coords.csv']):
        coord = np.loadtxt(coord_path, delimiter=',')
        distance = get_distance(coord)
        marker_size = get_marker_size(adjacency_matrix, coord)

        gcn = greedy_closest_neighbour(distance, adjacency_list)
        gd = greedy_destination(gcn)
        grs = greedy_routing_score(gd)

        R_max = 2*np.log(N)+1
        ls = np.linspace(-R_max, R_max, resolution)
        x_array, y_array = np.meshgrid(ls, ls)
        theta_array = np.arctan2(y_array, x_array)
        r_array = np.sqrt(x_array*x_array+y_array*y_array)

        within_disk_i, within_disk_j = np.where(r_array<R_max)
        K = len(within_disk_i)
        new_coord = np.tile(coord, (K, 1, 1))
        new_coord[:, moved_vertex, 0] = r_array[within_disk_i, within_disk_j]
        new_coord[:, moved_vertex, 1] = theta_array[within_disk_i, within_disk_j]

        new_distance = recalc_distances(new_coord, distance, moved_vertex)
        new_gcn = reclac_greedy_closest_neighbour(new_distance, adjacency_list)
        new_gd = reclac_greedy_destination(new_gcn)
        new_grs = reclac_routing_score(new_gd)

        heatmap = np.full_like(x_array, np.nan)
        heatmap[within_disk_i, within_disk_j] = new_grs - grs

        fig.add_heatmap(
            x=ls,
            y=ls,
            z=heatmap,
            colorscale='RdBu',
            colorbar_x=[0.45, 1.0][idx],
            colorbar_tickfont_size=24,
            zmid=0,
            row=1,
            col=idx+1
        )

        r, theta = coord.T
        x, y = r * np.cos(theta), r * np.sin(theta)

        x_edges, y_edges = [], []
        for u, v in edges:
            x_edges += [x[u], x[v], None]
            y_edges += [y[u], y[v], None]

        fig.add_scattergl(
            x=x_edges,
            y=y_edges,
            mode='lines',
            line_color='gray',
            line_width=0.5,
            hoverinfo='none',
            row=1,
            col=idx+1
        )

        fig.add_scattergl(
            x=x,
            y=y,
            mode='markers',
            marker_size=marker_size,
            marker_color=['purple' if i != moved_vertex else 'green' for i in range(N)],
            hovertext=np.arange(N),
            hoverinfo='text',
            row=1,
            col=idx+1
        )

        circle_r = R_max
        circle_theta = np.linspace(0, 2*np.pi, resolution)
        fig.add_scattergl(
            x=circle_r*np.cos(circle_theta),
            y=circle_r*np.sin(circle_theta),
            mode='lines',
            hoverinfo='none',
            line_color='black',
            line_width=5.0,
            row=1,
            col=idx+1
        )

    fig.update_layout(
        showlegend=False,
        width=1500,
        height=800,
        plot_bgcolor='rgba(0,0,0,0)',
        margin_l=0,
        margin_r=50,
        margin_t=120,
        margin_b=0,
    )

    fig.update_xaxes(
        tickvals=[],
        scaleanchor='y'
    )
    fig.update_yaxes(
        tickvals=[],
        scaleanchor='x'
    )

    fig.update_annotations(font_size=32)

    return fig

path = Path('experiments/beta/54d0eeb5aa1240d58f9e919f2d7f7c60')
moved_vertex = 73
fig = vertex_move_improvement_heatmap(path, moved_vertex, 256)
fig.write_image('figures/vertex_move_improvement_heatmap.pdf', scale=2.0)