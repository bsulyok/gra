import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots

def get_distance(coord: np.ndarray) -> np.ndarray:
    d_theta = np.pi - np.abs(np.pi - np.abs(coord[:, 1]-coord[:, np.newaxis, 1]))
    dist = np.cosh(coord[:, 0]) * np.cosh(coord[:, np.newaxis, 0])
    dist -= np.sinh(coord[:, 0]) * np.sinh(coord[:, np.newaxis, 0]) * np.cos(d_theta)
    return dist

def greedy_closest_neighbour(adjacency_list: np.ndarray, distance: np.ndarray) -> np.ndarray:
    gcn = np.array([neighbours[np.argmin(distance[neighbours], axis=0)] for neighbours in adjacency_list])
    np.fill_diagonal(gcn, np.arange(distance.shape[0]))
    return gcn

def greedy_destination(gcn: np.ndarray, num_grs_iter: int):
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
    gcn = greedy_closest_neighbour(adjacency_list, distance)
    gd = greedy_destination(gcn.copy(), num_grs_iter)
    gd = drip_down(gcn, gd, distance)
    return np.bincount(gd.flatten()) - np.sum(gd == np.arange(N), axis=0)

def draw_layout_compparison(adjacency_matrix: np.ndarray, coord_orig: np.ndarray, coord_emb: np.ndarray, marker_size_scale: float = 200, marker_size_min: float = 5, save_name: str = None) -> None:
    r_emb, theta_emb = coord_emb.T
    r_orig, theta_orig = coord_orig.T
    
    N = adjacency_matrix.shape[0]
    N_pairs = N * (N-1) / 2

    clogging_count_orig = get_clogging_count(adjacency_matrix, coord_orig)
    marker_size_orig = np.sqrt(clogging_count_orig / N_pairs)
    marker_size_orig = np.clip(marker_size_orig*marker_size_scale, a_min=marker_size_min, a_max=None)
    
    clogging_count_emb = get_clogging_count(adjacency_matrix, coord_emb)
    marker_size_emb = np.sqrt(clogging_count_emb / N_pairs)
    marker_size_emb = np.clip(marker_size_emb*marker_size_scale, a_min=marker_size_min, a_max=None)
    
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'polar'}, {'type': 'polar'}]])

    for source, target in zip(*np.where(np.triu(adjacency_matrix))):
        fig.add_trace(
            go.Scatterpolargl(
                r=r_orig[[source, target]],
                theta=theta_orig[[source, target]],
                thetaunit='radians',
                line_color='gray',
                mode='lines',
                line_width=0.5
            ),
            row=1,
            col=1
        )
        fig.add_trace(
            go.Scatterpolargl(
                r=r_emb[[source, target]],
                theta=theta_emb[[source, target]],
                thetaunit='radians',
                line_color='gray',
                mode='lines',
                line_width=0.5
            ),
            row=1,
            col=2
        )

    fig.add_trace(
        go.Scatterpolargl(
            r=r_orig,
            theta=theta_orig,
            thetaunit='radians',
            mode='markers',
            marker_size=marker_size_orig,
            marker_colorscale='Rainbow',
            marker_color=theta_orig/2/np.pi
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatterpolargl(
            r=r_emb,
            theta=theta_emb,
            thetaunit='radians',
            mode='markers',
            marker_size=marker_size_emb,
            marker_colorscale='Rainbow',
            marker_color=theta_orig/2/np.pi
        ),
        row=1,
        col=2
    )
    fig.update_polars(
        angularaxis_showgrid=False,
        angularaxis_showline=False,
        angularaxis_showticklabels=False,
        radialaxis_showgrid=False,
        radialaxis_showline=False,
        radialaxis_showticklabels=False,
    )
    fig.update_layout(
        showlegend=False,
        width=1600,
        height=800
    )
    if save_name is None:
        fig.show()
    else:
        fig.write_image(save_name)
        

def get_conditional_connection_probability(adjacency_matrix: np.ndarray, coord: np.ndarray, n_bins: int) -> np.ndarray:
    indices = np.triu_indices(adjacency_matrix.shape[0], 1)
    #with np.errstate(all='ignore'):
    distance = np.arccosh(get_distance(coord))
    np.fill_diagonal(distance, 0)
    distance = distance[indices]
    connected = adjacency_matrix[indices]
    hist_sum, bins = np.histogram(distance, bins=n_bins)
    hist_connected = np.histogram(distance[connected], bins=bins)[0]
    return bins, hist_connected / hist_sum


def draw_conditional_connection_probability(adjacency_matrix: np.ndarray, coord_orig: np.ndarray, coord_emb: np.ndarray, n_bins: int = 20, save_name: str = None) -> None:
    with np.errstate(all='ignore'):
        bins_emb, conn_prob_emb = get_conditional_connection_probability(adjacency_matrix, coord_emb, n_bins)
        bins_orig, conn_prob_orig = get_conditional_connection_probability(adjacency_matrix, coord_orig, n_bins)
    
    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=bins_emb[:-1],
        y=conn_prob_emb,
        name='simulated'
    ))

    fig.add_trace(go.Scattergl(
        x=bins_orig[:-1],
        y=conn_prob_orig,
        name='original'
    ))

    fig.update_layout(
        width=1600,
        height=800
    )
    if save_name is None:
        fig.show()
    else:
        fig.write_image(save_name)


def draw_clogging_representation(adjacency_matrix: np.ndarray, coord: np.ndarray, marker_size_scale: float = 200, marker_size_min: float = 5, save_name: str = None) -> None:
    
    N = adjacency_matrix.shape[0]
    N_pairs = N * (N-1) / 2
    clogging_count = get_clogging_count(adjacency_matrix, coord_emb)
    marker_size = np.sqrt(clogging_count / N_pairs)
    marker_size = np.clip(marker_size*marker_size_scale, a_min=marker_size_min, a_max=None)
    
    r, theta = coord.T

    fig = go.Figure()

    for source, target in zip(*np.where(np.triu(adjacency_matrix))):
        fig.add_trace(
            go.Scatterpolargl(
                r=r[[source, target]],
                theta=theta[[source, target]],
                thetaunit='radians',
                line_color='gray',
                mode='lines',
                line_width=0.5
            ))

    fig.add_trace(go.Scatterpolargl(
        r=r,
        theta=theta,
        thetaunit='radians',
        mode='markers',
        marker_size=50*marker_size,
        marker_color='blue'
        ))

    fig.update_polars(
        angularaxis_showgrid=False,
        angularaxis_showline=False,
        angularaxis_showticklabels=False,
        radialaxis_showgrid=False,
        radialaxis_showline=False,
        radialaxis_showticklabels=False,
    )
    fig.update_layout(
        showlegend=False,
        width=800,
        height=800
    )
    if save_name is None:
        fig.show()
    else:
        fig.write_image(save_name)
        
        
        
        
        

def recalc_greedy_routing(adjacency_list, distance, num_grs_iter, gcn, gd, moved_vertex):
    N = adjacency_list.shape[0]
    moved_vertex_neighbours = adjacency_list[moved_vertex]

    arange = np.arange(N)
    moved_vertex_neighbours_next_hop = np.array([neighbours[np.argmin(distance[neighbours], axis=0)] for neighbours in adjacency_list[moved_vertex_neighbours]])
    moved_vertex_neighbours_next_hop[np.arange(moved_vertex_neighbours.shape[0]), moved_vertex_neighbours] = moved_vertex_neighbours

    moved_vertex_next_hop = np.argmin(distance[moved_vertex_neighbours], axis=0)

    prev_next_next_hop = gcn[gcn[moved_vertex], arange]
    cur_next_next_hop = moved_vertex_neighbours_next_hop[moved_vertex_next_hop, arange]
    changed = prev_next_next_hop != cur_next_next_hop
    changed_too = np.any(gcn[moved_vertex_neighbours] != moved_vertex_neighbours_next_hop, axis=0)
    changed = changed | changed_too

    new_gcn = gcn.copy()

    new_gcn[:, moved_vertex] = np.array([neighbours[np.argmin(distance[neighbours, moved_vertex], axis=0)] for neighbours in adjacency_list])
    new_gcn[moved_vertex, moved_vertex] = moved_vertex
    changed[moved_vertex] = True

    new_gcn[moved_vertex_neighbours] = moved_vertex_neighbours_next_hop
    new_gd = gd.copy()
    new_gd[:, changed] = greedy_destination(new_gcn[:, changed].copy(), num_grs_iter)
    return new_gcn, new_gd