# think about excluding outliers
import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots


str2mc = np.array([
    [0.99998049, -0.00624386, 0.00017223],
    [0.00624276,  0.99996378, 0.00578512],
    [-0.00014849, -0.00578431, 0.99998326]
])


def load_angle_vectors(json_path, which = 'b'):
    """
    Loads angular data from a JSON file and returns a dictionary
    mapping each name to its corresponding numpy array (vector).
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # Convert lists to numpy arrays
    vectors = {key: np.array(value[which]) for key, value in data.items()}
    return vectors

def getA(mdf):
    return mdf[0]

def getD(mdf):
    return mdf[1]

def spherical_to_cartesian(args):
    theta, phi = args
    theta, phi = np.radians(theta), np.radians(phi)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    return np.array([x, y, z])


def cartesian_to_spherical(x, y, z):
    theta = np.arctan2(y, x)
    phi = np.arccos(z)
    return theta, phi


def find_z_rotation_matrix(A, B):
    theta_A = np.arctan2(A[1], A[0])
    theta_B = np.arctan2(B[1], B[0])
    theta = theta_B - theta_A
    print(theta * 180 / np.pi)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

    return R




def cone_surface(v, angle, length=1, n_points=50):
    """Generate points for a cone surface given direction vector and angle"""
    v = v / np.linalg.norm(v)  # Normalize
    angle = np.radians(angle)
    
    # Create a circle perpendicular to the vector
    theta = np.linspace(0, 2*np.pi, n_points)
    r = length * np.tan(angle)
    
    # Find two perpendicular vectors to v
    if np.allclose(v, [0, 0, 1]):
        a = np.array([1, 0, 0])
    else:
        a = np.array([-v[1], v[0], 0])
        a = a / np.linalg.norm(a)
    b = np.cross(v, a)
    
    # Generate cone points
    t = np.linspace(0, length, 2)
    theta, t = np.meshgrid(theta, t)
    x = t * (v[0] + r * np.cos(theta) * a[0] + r * np.sin(theta) * b[0])
    y = t * (v[1] + r * np.cos(theta) * a[1] + r * np.sin(theta) * b[1])
    z = t * (v[2] + r * np.cos(theta) * a[2] + r * np.sin(theta) * b[2])
    norm_all = np.sqrt(x**2 + y**2 + z ** 2)

    return x/norm_all[1],y/norm_all[1],z/norm_all[1]

def find_intersections(angle1, angle2):
    """Find intersection points between two cones"""
    # v1 = np.array(v1) / np.linalg.norm(v1)
    # v2 = np.array(v2) / np.linalg.norm(v2)
    angle1 = np.radians(angle1)
    angle2 = np.radians(angle2)
    vector = [-np.cos(angle2),-np.sqrt(1 - np.cos(angle2)**2 - np.cos(angle1)**2) ,np.cos(angle1)]
    print(np.linalg.norm(vector))

    
    return [vector]




def plot_vectors_with_angle(vector1, vector2, origin=None, vector_length=3, arc_radius=1.5, 
                           arc_resolution=50, title="3D Vector Angle Visualization", 
                           additional_vectors=None):
    """
    Create a 3D plot showing two vectors and the circular arc representing the angle between them.
    
    Parameters:
    - vector1, vector2: 3D vectors (lists or numpy arrays)
    - origin: origin point for vectors (default: [0,0,0])
    - vector_length: length to draw vectors (default: 3)
    - arc_radius: radius of the angle arc (default: 1.5)
    - arc_resolution: number of points in the arc (default: 50)
    - title: plot title
    - additional_vectors: list of dicts with vector info, format:
      [{'vector': [x,y,z], 'name': 'legend_name', 'color': 'color_name', 'length': optional_length}, ...]
    
    Returns:
    - plotly figure object
    """
    
    # Set default origin
    if origin is None:
        origin = np.array([0, 0, 0])
    else:
        origin = np.array(origin)
    
    # Convert to numpy arrays and normalize
    v1 = np.array(vector1)
    v2 = np.array(vector2)
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # Calculate angle between vectors
    dot_product = np.dot(v1_norm, v2_norm)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    # Scale vectors to desired length
    v1_display = origin + vector_length * v1_norm
    v2_display = origin + vector_length * v2_norm
    
    # Create figure
    fig = go.Figure()
        # Plot additional vectors if provided
    if additional_vectors is not None:
        available_colors = ['orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'gray', 'olive', 'navy']
        color_index = 0
        
        for i, vec_info in enumerate(additional_vectors):
            # Extract vector information
            vector = np.array(vec_info['vector'])
            name = vec_info.get('name', f'Vector {i+3}')
            color = vec_info.get('color', available_colors[color_index % len(available_colors)])
            vec_length = vec_info.get('length', vector_length)
            
            # Normalize and scale vector
            vector_norm = vector / np.linalg.norm(vector)
            vector_display = origin + vec_length * vector_norm
            
            # Plot the additional vector
            fig.add_trace(go.Scatter3d(
                x=[origin[0], vector_display[0]], 
                y=[origin[1], vector_display[1]], 
                z=[origin[2], vector_display[2]],
                mode='lines+markers',
                line=dict(color=color, width=6),
                marker=dict(size=6, color=color),
                name=f'{name}: [{vector[0]:.2f}, {vector[1]:.2f}, {vector[2]:.2f}]',
                showlegend=True
            ))
            
            color_index += 1
    # # Plot vector 1
    # fig.add_trace(go.Scatter3d(
    #     x=[origin[0], v1_display[0]], 
    #     y=[origin[1], v1_display[1]], 
    #     z=[origin[2], v1_display[2]],
    #     mode='lines+markers',
    #     line=dict(color='red', width=8),
    #     marker=dict(size=8, color='red'),
    #     # name=f'Vector 1: [{v1[0]:.2f}, {v1[1]:.2f}, {v1[2]:.2f}]',
    #     name = None,
    #     showlegend=True
    # ))
    
    # Plot vector 2
    fig.add_trace(go.Scatter3d(
        x=[origin[0], v2_display[0]], 
        y=[origin[1], v2_display[1]], 
        z=[origin[2], v2_display[2]],
        mode='lines+markers',
        line=dict(color='orange', width=8),
        marker=dict(size=8, color='orange'),
        name=f'Beacon Camera: [{v2[0]:.4f}, {v2[1]:.4f}, {v2[2]:.4f}]',
        showlegend=True
    ))
    
    # Create circular arc to show angle
    if angle_rad > 1e-6:  # Only if vectors are not parallel
        # Find the plane containing both vectors
        # The normal to this plane is the cross product
        normal = np.cross(v1_norm, v2_norm)
        normal_mag = np.linalg.norm(normal)
        
        if normal_mag > 1e-6:  # Vectors are not parallel
            normal = normal / normal_mag
            
            # Create orthonormal basis in the plane
            u1 = v1_norm
            u2 = v2_norm - np.dot(v2_norm, u1) * u1
            u2 = u2 / np.linalg.norm(u2)
            
            # Generate arc points
            theta_values = np.linspace(0, angle_rad, arc_resolution)
            arc_points = []
            
            for theta in theta_values:
                # Point on arc: origin + arc_radius * (cos(theta)*u1 + sin(theta)*u2)
                point = origin + arc_radius * (np.cos(theta) * u1 + np.sin(theta) * u2)
                arc_points.append(point)
            
            arc_points = np.array(arc_points)
            
            # Plot the arc
            fig.add_trace(go.Scatter3d(
                x=arc_points[:, 0],
                y=arc_points[:, 1],
                z=arc_points[:, 2],
                mode='lines+markers',
                line=dict(color='green', width=6),
                marker=dict(size=4, color='green'),
                name=f'Angle Arc: θ = {angle_deg:.1f}°',
                showlegend=True
            ))
            
            # # Add radial lines to show the arc clearly
            # # Line from origin to start of arc
            # fig.add_trace(go.Scatter3d(
            #     x=[origin[0], arc_points[0, 0]],
            #     y=[origin[1], arc_points[0, 1]],
            #     z=[origin[2], arc_points[0, 2]],
            #     mode='lines',
            #     line=dict(color='green', width=3, dash='dash'),
            #     name='Arc Radius',
            #     showlegend=True
            # ))
            
            # Line from origin to end of arc
            fig.add_trace(go.Scatter3d(
                x=[origin[0], arc_points[-1, 0]],
                y=[origin[1], arc_points[-1, 1]],
                z=[origin[2], arc_points[-1, 2]],
                mode='lines',
                line=dict(color='green', width=3, dash='dash'),
                showlegend=False  # Don't duplicate legend entry
            ))
    
    # Plot origin point
    fig.add_trace(go.Scatter3d(
        x=[origin[0]], 
        y=[origin[1]], 
        z=[origin[2]],
        mode='markers',
        marker=dict(size=10, color='black', symbol='diamond'),
        name='Origin',
        showlegend=True
    ))
    
    # Add text annotation for angle
    mid_point = origin + arc_radius * 0.7 * (v1_norm + v2_norm) / np.linalg.norm(v1_norm + v2_norm)
    fig.add_trace(go.Scatter3d(
        x=[mid_point[0]],
        y=[mid_point[1]], 
        z=[mid_point[2]],
        mode='text',
        text=[f'θ = {angle_deg:.1f}°'],
        textfont=dict(size=16, color='purple'),
        name='Angle Label',
        showlegend=False
    ))

    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{title}<br><sub>Angle between vectors: {angle_deg:.2f}° ({angle_rad:.3f} rad)</sub>',
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube',  # Equal aspect ratio
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1
        ),
        width=1000,
        height=1000,
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    return fig

def plot_multiple_vector_pairs(vector_pairs, titles=None):
    """
    Plot multiple vector pairs in subplots.
    
    Parameters:
    - vector_pairs: list of tuples [(v1_1, v2_1), (v1_2, v2_2), ...]
    - titles: list of subplot titles
    """
    n_pairs = len(vector_pairs)
    cols = min(2, n_pairs)
    rows = (n_pairs + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=titles or [f'Pair {i+1}' for i in range(n_pairs)],
        specs=[[{'type': 'scatter3d'} for _ in range(cols)] for _ in range(rows)]
    )
    
    for i, (v1, v2) in enumerate(vector_pairs):
        row = i // cols + 1
        col = i % cols + 1
        
        # Create individual plot
        temp_fig = plot_vectors_with_angle(v1, v2, title="")
        
        # Add traces to subplot
        for trace in temp_fig.data:
            fig.add_trace(trace, row=row, col=col)
    
    fig.update_layout(height=400*rows, title="Multiple Vector Angle Visualizations")
    return fig



def plot_cone_surfaces(angle_str, angle_ss):
    # Input parameters
    v1 = [0, 0, 1]  # First vector (direction)
    v2 = [-1, 0, 0]  # Second vector (direction)

    # Generate cone surfaces
    x1, y1, z1 = cone_surface(v1, angle_str)
    if angle_str > 90:
        z1 = -z1
    x2, y2, z2 = cone_surface(v2, angle_ss)
    if angle_ss > 90:
        x2 = -x2

    # Find intersection points
    intersections = find_intersections(angle_str, angle_ss)
    # intersections = []

    # Create plot
    fig = go.Figure()

    # # Add cones
    fig.add_trace(go.Surface(x=x1, y=y1, z=z1, opacity=0.8, colorscale='Blues', showscale=False, name=f'Cone 1 ({angle_str:.3f}°)', showlegend=True))
    fig.add_trace(go.Surface(x=x2, y=y2, z=z2, opacity=0.4, colorscale='Reds', showscale=False, name=f'Cone 2 ({angle_ss:.3f}°)', showlegend=True))

    # Add vectors
    fig.add_trace(go.Scatter3d(x=[0, v1[0]], y=[0, v1[1]], z=[0, v1[2]], 
                            mode='lines+markers', line=dict(color='blue', width=5), name='STR Side (STR + Z)'))
    fig.add_trace(go.Scatter3d(x=[0, v2[0]], y=[0, v2[1]], z=[0, v2[2]], 
                            mode='lines+markers', line=dict(color='red', width=5), name='Sun Sensor Side (STR -X)'))

    # Add intersection points and vectors
    for i, point in enumerate(intersections):
        print(point)
        fig.add_trace(go.Scatter3d(x=[0, point[0]], y=[0, point[1]], z=[0, point[2]], 
                                mode='lines+markers', line=dict(color='orange', width=8), 
                                marker=dict(size=5), name=f'Camera Direction'))

    # Add annotations
    if intersections:
        annotation_text = f"Found {len(intersections)} intersection(s)"
    else:
        annotation_text = "No intersections found"

    fig.update_layout(
        title=f'Cone Intersections',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        height = 1000,
        width = 1000,
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1
        ),
    )

    fig.show()