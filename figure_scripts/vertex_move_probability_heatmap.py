from plotly import graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from scipy.stats import truncnorm, norm

def vertex_move_probability_heatmap(rs, thetas, N=128, resolution=100):
    
    subplot_titles=[
        'a)                           ',
        'b)                           ',
        'c)                           '      
    ]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.01,
        vertical_spacing=0.0,
    )

    for idx, (r, theta) in enumerate(zip(rs, thetas)):
        R_max = 2*np.log(N)

        ls = np.linspace(-R_max, R_max, resolution)
        x, y = np.meshgrid(ls, ls)
        theta_array = np.arctan2(y, x)
        r_array = np.sqrt(x*x+y*y)

        clip_min, clip_max = 0, R_max
        loc, scale = r, 1
        a, b = (clip_min - loc) / scale, (clip_max - loc) / scale
        radial_prob = truncnorm.pdf(x=r_array, a=a, b=b, loc=loc, scale=scale)
        angular_prob = norm.pdf(x=theta_array, loc=theta, scale=np.pi/4)
        angular_prob += norm.pdf(x=theta_array+2*np.pi, loc=theta, scale=np.pi/4)
        angular_prob += norm.pdf(x=theta_array-2*np.pi, loc=theta, scale=np.pi/4)
        prob = angular_prob * radial_prob
        prob /= prob.sum()
        prob[r_array > R_max] = np.nan

        circumference_theta = np.linspace(0, 2*np.pi, resolution)
        circumference_r = np.full(resolution, resolution//2)
        fig.add_heatmap(
            z=prob,
            colorscale='Greens',
            showscale=False,
            row=1,
            col=idx+1,
            hoverinfo='none'
        )
        fig.add_scattergl(
            x=np.cos(circumference_theta) * circumference_r + resolution//2,
            y=np.sin(circumference_theta) * circumference_r + resolution//2,
            mode='lines',
            line_color='black',
            line_width=1.5,
            row=1,
            col=idx+1,
            hoverinfo='none'
        )
        fig.add_scattergl(
            x=[r*np.cos(theta)/R_max*(resolution//2) + resolution // 2],
            y=[r*np.sin(theta)/R_max*(resolution//2) + resolution // 2],
            marker_color='red',
            marker_symbol='x',
            marker_size=10,
            row=1,
            col=idx+1
        )

    fig.update_xaxes(showticklabels=False)  
    fig.update_yaxes(showticklabels=False)  

    fig.update_layout(
        width=500,
        height=500//3+50,
        margin_l=0,
        margin_r=0,
        margin_t=24,
        margin_b=0,
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
    )
    fig.update_xaxes(
        scaleanchor='y',
        range=[-5, resolution+5]
    )
    fig.update_yaxes(
        scaleanchor='x'
    )
    fig.update_annotations(font_size=14)
    return fig

fig = vertex_move_probability_heatmap([2, 5.5, 9], [np.pi, 5/3*np.pi, 1/3*np.pi], resolution=2048)
fig.write_image('figures/vertex_move_probability_heatmap.pdf', scale=5.0)