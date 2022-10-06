import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
from itertools import product

GRAPH_NAMES = [
    'football',
    'pso_128',
    'pso_256',
    'metabolic',
    'pso_512',
    'unicodelang',
    'crime',
    'pso_1024'
]


ANNEALER_NAMES = [
    'BaseAnnealer',
    'SourceDependent',
    'TargetDependent'
]


COLORS = [
    (253, 50, 22),
    (0, 254, 53),
    (106, 118, 252),
    #(254, 212, 196),
    (254, 0, 206),
    (13, 249, 255),
    (246, 249, 38),
    (255, 150, 22),
    (71, 155, 85),
    (238, 166, 251),
    (220, 88, 125),
    (214, 38, 255),
    (110, 137, 156),
    (0, 181, 247),
    (182, 142, 0),
    (201, 251, 229),
    (255, 0, 146),
    (34, 255, 167),
    (227, 238, 158),
    (134, 206, 0),
    (188, 113, 150),
    (126, 125, 205),
    (252, 105, 85),
    (228, 143, 114)
]

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

def layout_comparison(adjacency_matrix: np.ndarray, coord_orig: np.ndarray, coord_emb: np.ndarray, marker_size_scale: float = 200, marker_size_min: float = 5, save_name: str = None) -> None:
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
    
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'polar'}, {'type': 'polar'}]], subplot_titles=['A', 'B'])

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
    fig.update_annotations(
        font_size=72
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


def conditional_connection_probability(adjacency_matrix: np.ndarray, coord_orig: np.ndarray, coord_emb: np.ndarray, n_bins: int = 20, save_name: str = None) -> None:
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


def clogging_representation(adjacency_matrix: np.ndarray, coord: np.ndarray, marker_size_scale: float = 200, marker_size_min: float = 5, save_name: str = None) -> None:
    
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


def annealer_comparison(graphs=GRAPH_NAMES, annealers=ANNEALER_NAMES, coord='random', results_path=Path('../results'), save_name=None):
    fig = make_subplots(rows=2, cols=4, subplot_titles=graphs)
    for (graph_idx, graph), (annealer_idx, annealer) in product(enumerate(graphs), enumerate(annealers)):
        log_path = results_path / f'{graph}_{annealer}_{coord}.csv'
        if Path(log_path).exists():
            mean, std = np.loadtxt(log_path)
            mean = mean[::100]
            std = std[::100]
            t=np.arange(mean.shape[0])*100
            fig.add_scattergl(
                x=t,
                y=mean,
                mode='lines',
                legendgroup=annealer,
                name=annealer,
                line_color=COLORS[annealer_idx],
                showlegend=True if graph_idx == 0 else False,
                row=graph_idx//4+1,
                col=graph_idx%4+1
            )
    fig.update_layout(
        width=1200,
        height=600
    )
    if save_name is None:
        fig.show()
    else:
        fig.write_image(save_name)
        

def annealer_comparison(coord_name, annealer_names=ANNEALER_NAMES, results_path=Path('../results'), save_name=None):
    fig = make_subplots(rows=2, cols=4, subplot_titles=GRAPH_NAMES)
    for (graph_idx, graph_name), (annealer_idx, annealer_name) in product(enumerate(GRAPH_NAMES), enumerate(annealer_names)):
        log_path = results_path / f'{graph_name}_{annealer_name}_{coord_name}.csv'
        if Path(log_path).exists():
            mean, std = np.loadtxt(log_path)
            step = max(100, mean.shape[0]) // 100
            mean = mean[::step]
            std = std[::step]
            t=np.arange(mean.shape[0]) * step
            fig.add_scattergl(
                x=np.hstack([t, t[::-1]]),
                y=np.hstack([mean+std,(mean-std)[::-1]]),
                mode='lines',
                legendgroup=annealer_name,
                name=annealer_name,
                hoverinfo='skip',
                fill='toself',
                line_width=0,
                fillcolor=f'rgba({COLORS[annealer_idx][0]},{COLORS[annealer_idx][1]},{COLORS[annealer_idx][2]},0.2)',
                showlegend=False,
                row=graph_idx//4+1,
                col=graph_idx%4+1
            )
            fig.add_scattergl(
                x=t,
                y=mean,
                mode='lines',
                legendgroup=annealer_name,
                name=annealer_name,
                line_color=f'rgba({COLORS[annealer_idx][0]},{COLORS[annealer_idx][1]},{COLORS[annealer_idx][2]},1.0)',
                showlegend=True if graph_idx == 0 else False,
                row=graph_idx//4+1,
                col=graph_idx%4+1
            )
    fig.update_layout(
        width=1200,
        height=600,
        showlegend=True
    )
    #fig.update_yaxes(
    #    range=[0,1]
    #)
    if save_name is None:
        fig.show()
    else:
        fig.write_image(save_name)
        
        
def polar_heatmap_plot(coord, radial_resolution, angular_resolution, marker_colorscale):
    azimuthal_bins = np.linspace(0, 2*np.pi, angular_resolution)
    hist, radial_bins, _ = np.histogram2d(coord[:, 0], coord[:, 1], bins=(radial_resolution, azimuthal_bins))
    hist = hist.T
    height = radial_bins[1] - radial_bins[0]
    width = azimuthal_bins[1] - azimuthal_bins[0]
    r, theta = np.meshgrid(
        radial_bins,
        (azimuthal_bins[1:] + azimuthal_bins[:-1]) / 2
    )
    hits = hist > 0
    top = r[:, 1:][hits]
    base = r[:, :-1][hits]
    theta = theta[:, :-1][hits]

    bar_polar = go.Barpolar(
        r=np.full_like(top, height),
        theta=theta,
        width=width,
        base=base,
        thetaunit='radians',
        hovertext='',
        hoverinfo='skip',
        marker_color=hist[hits]/hist.sum(),
        marker_colorscale=marker_colorscale
    )
    
    return bar_polar


def vertex_move_heatmap(coord, moved_vertices=[0], radial_resolution=20, angular_resolution=100, sample_size=500, save_name=None):
    marker_colorscales = ['Reds', 'Greens', 'Blues']
    fig = go.Figure()
    for idx, moved_vertex in enumerate(moved_vertices):
        clip_min, clip_max = 1, 2*np.log(coord.shape[0])
        loc, scale = coord[moved_vertex, 0], 1
        a, b = (clip_min - loc) / scale, (clip_max - loc) / scale
        r = truncnorm.rvs(a=a, b=b, loc=loc, scale=scale, size=sample_size)
        theta = np.random.normal(loc=coord[moved_vertex, 1], scale=np.pi/4, size=sample_size) % (2*np.pi)
        fig.add_trace(polar_heatmap_plot(np.stack([r, theta]).T, radial_resolution, angular_resolution, marker_colorscales[idx]))
        print(marker_colorscales[idx])
        
    fig.update_polars(
        radialaxis_showticklabels=False,
        angularaxis_showticklabels=False,
        bgcolor='rgba(255,255,255,1)'
    )

    fig.update_layout(
        width=800,
        height=800,
    )
    if save_name is None:
        fig.show()
    else:
        fig.write_image(save_name)