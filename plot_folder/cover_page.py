import numpy as np
import plotly.graph_objs as go
from scipy.ndimage import gaussian_filter1d
import matplotlib.cm as cm
# Generate points on a sphere
num_points = 1000
phi = np.linspace(0, np.pi, num_points)
theta = np.linspace(0, 2*np.pi, num_points)
phi, theta = np.meshgrid(phi, theta)
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

# Divide the sphere into two parts
x_positive = x.copy()
x_negative = x.copy()
x_positive[x < 0] = None
x_negative[x >= 0] = None

# Create traces for each half
sphere_positive_trace = go.Surface(
    x=x_positive,
    y=y,
    z=z,
    opacity=0.5,
    colorscale='blues',
    showscale=False
)

sphere_negative_trace = go.Surface(
    x=x_negative,
    y=y,
    z=z,
    opacity=0.5,
    colorscale='reds',
    showscale=False
)

# Define trajectory
t = np.linspace(0, 8*np.pi, num_points)
sigma=0.01
trajectory_x = np.sin(t)
trajectory_y = np.cos(t)
trajectory_z=np.zeros_like(t)
for j in range(1,trajectory_z.size):
    trajectory_z[j]=trajectory_z[j-1]+np.random.normal(0, sigma, 1)
P=np.array([trajectory_x,trajectory_y,trajectory_z])
P=gaussian_filter1d(P,sigma=2,axis=1)
P=P/np.linalg.norm(P,axis=0).reshape((1,-1))
trajectory_x=P[0,:]
trajectory_y=P[1,:]
trajectory_z=P[2,:]

# Determine color scale for trajectory
color_scale = cm.viridis(np.linspace(0, 1, num_points))

# Create a trajectory trace
trajectory_trace = go.Scatter3d(
    x=trajectory_x,
    y=trajectory_y,
    z=trajectory_z,
    mode='lines',
    line=dict(color=color_scale, width=4)
)

# Create a figure
fig = go.Figure(data=[sphere_positive_trace, sphere_negative_trace, trajectory_trace])

# Set layout without axis
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='cube'
    )
)

# Show the plot
fig.show()
