"""PyVista 3D visualization helpers for graph rendering."""

import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, Tuple, Any

try:
    import pyvista as pv
    try:
        pv.start_xvfb()
    except (OSError, AttributeError):
        pass
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

FIGURE_DPI = 300


def visualize_graph_3d_pyvista(
    G: nx.Graph,
    pos_3d: Dict[int, Tuple[float, float, float]],
    edge_labels: Dict[Tuple[int, int], Any],
    output_path: Path,
    label_type: str = "scalar",
    t_samples: np.ndarray = None
) -> None:
    """Visualize graph in 3D using PyVista with label on closest edge."""
    if not PYVISTA_AVAILABLE:
        raise ImportError("PyVista is required for 3D visualization")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plotter = pv.Plotter(off_screen=True, window_size=[2000, 2000])
    
    plotter.renderer.SetAutomaticLightCreation(False)
    plotter.renderer.LightFollowCameraOff()
    
    light1 = pv.Light(position=(5, 5, 5), focal_point=(0, 0, 0), color='white', intensity=1.5)
    light1.positional = False
    plotter.add_light(light1)
    
    light2 = pv.Light(position=(-3, 3, 2), focal_point=(0, 0, 0), color='white', intensity=0.8)
    light2.positional = False
    plotter.add_light(light2)
    
    plotter.renderer.SetUseShadows(False)
    plotter.renderer.SetTwoSidedLighting(True)
    
    plotter.set_background([0, 0, 0, 0])
    
    n = G.number_of_nodes()
    node_positions = np.array([list(pos_3d[i]) for i in range(n)])
    
    plotter.camera.position = (5, 5, 5)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 0, 1)
    plotter.camera.SetViewUp(0, 0, 1)
    plotter.render()
    
    camera_pos = np.array(plotter.camera.position)
    camera_up = np.array(plotter.camera.up)
    camera_up = camera_up / (np.linalg.norm(camera_up) + 1e-10)
    
    graph_center = np.mean(node_positions, axis=0)
    consistent_view_dir = camera_pos - graph_center
    consistent_view_dir = consistent_view_dir / (np.linalg.norm(consistent_view_dir) + 1e-10)
    
    camera_forward = -consistent_view_dir
    
    camera_right = np.cross(camera_up, camera_forward)
    camera_right = camera_right / (np.linalg.norm(camera_right) + 1e-10)
    camera_up_corrected = np.cross(camera_forward, camera_right)
    camera_up_corrected = camera_up_corrected / (np.linalg.norm(camera_up_corrected) + 1e-10)
    
    consistent_rot_matrix = np.eye(3)
    consistent_rot_matrix[:, 0] = camera_right
    consistent_rot_matrix[:, 1] = camera_up_corrected
    consistent_rot_matrix[:, 2] = camera_forward
    
    edges = list(G.edges())
    edge_midpoints = {}
    for u, v in edges:
        mid_point = (np.array(pos_3d[u]) + np.array(pos_3d[v])) / 2
        edge_midpoints[(u, v)] = mid_point
    
    squiggliest_edge = (1, 2)
    closest_edge = None
    for e in edges:
        if (e[0] == squiggliest_edge[0] and e[1] == squiggliest_edge[1]) or \
           (e[0] == squiggliest_edge[1] and e[1] == squiggliest_edge[0]):
            closest_edge = e
            break
    
    if closest_edge is None:
        edge_distances = []
        for u, v in edges:
            mid_point = edge_midpoints[(u, v)]
            dist = np.linalg.norm(mid_point - camera_pos)
            edge_distances.append(((u, v), dist))
        edge_distances.sort(key=lambda x: x[1])
        if len(edge_distances) >= 1:
            closest_edge = edge_distances[0][0]
        else:
            closest_edge = edges[0] if edges else None
    
    edge_radius = 0.015
    
    def create_curved_edge_from_homeomorphism(u, v, phi_e, t_samples, start, end):
        """Create a curved edge based on homeomorphism phi_e."""
        if len(phi_e) != len(t_samples):
            t_old = np.linspace(0, 1, len(phi_e))
            phi_e = np.interp(t_samples, t_old, phi_e)
        
        n_points = len(t_samples)
        curve_points = []
        
        direction = end - start
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-10:
            direction_unit = direction / direction_norm
        else:
            direction_unit = np.array([1, 0, 0])
        
        if abs(direction_unit[2]) < 0.9:
            perp_vec = np.cross(direction_unit, np.array([0, 0, 1]))
        else:
            perp_vec = np.cross(direction_unit, np.array([1, 0, 0]))
        perp_vec = perp_vec / (np.linalg.norm(perp_vec) + 1e-10)
        
        max_offset = direction_norm * 0.15 * 2.0
        
        for i, t in enumerate(t_samples):
            pos_along_line = start + t * direction
            identity_value = t
            deviation = phi_e[i] - identity_value
            offset = perp_vec * deviation * max_offset
            curve_points.append(pos_along_line + offset)
        
        curve_points = np.array(curve_points)
        spline = pv.Spline(curve_points, n_points=n_points)
        return spline.tube(radius=edge_radius, n_sides=12)
    
    if label_type == "homeomorphism" and t_samples is not None:
        for u, v in G.edges():
            edge = tuple(sorted((u, v)))
            start = np.array(pos_3d[u])
            end = np.array(pos_3d[v])
            
            phi_e = edge_labels.get(edge, t_samples / t_samples[-1])
            
            is_highlighted = (closest_edge is not None and 
                            ((u, v) == closest_edge or (v, u) == closest_edge))
            
            tube = create_curved_edge_from_homeomorphism(u, v, phi_e, t_samples, start, end)
            edge_color = 'red' if is_highlighted else '#333333'
            actor = plotter.add_mesh(tube, color=edge_color, opacity=0.85, smooth_shading=True,
                            ambient=0.3, diffuse=0.6, specular=0.2, specular_power=20,
                            show_edges=False, pbr=False, metallic=0.0, roughness=0.6)
            actor.GetProperty().SetLighting(True)
    else:
        for u, v in G.edges():
            if closest_edge is not None and ((u, v) == closest_edge or (v, u) == closest_edge):
                continue
            start = np.array(pos_3d[u])
            end = np.array(pos_3d[v])
            line = pv.Line(start, end)
            tube = line.tube(radius=edge_radius, n_sides=12)
            actor = plotter.add_mesh(tube, color='#333333', opacity=0.85, smooth_shading=True,
                            ambient=0.3, diffuse=0.6, specular=0.2, specular_power=20,
                            show_edges=False, pbr=False, metallic=0.0, roughness=0.6)
            actor.GetProperty().SetLighting(True)
        
        if closest_edge is not None:
            u, v = closest_edge
            start = np.array(pos_3d[u])
            end = np.array(pos_3d[v])
            line = pv.Line(start, end)
            tube = line.tube(radius=edge_radius, n_sides=12)
            actor = plotter.add_mesh(tube, color='red', opacity=0.85, smooth_shading=True,
                            ambient=0.3, diffuse=0.6, specular=0.2, specular_power=20,
                            show_edges=False, pbr=False, metallic=0.0, roughness=0.6)
            actor.GetProperty().SetLighting(True)
    
    node_size = 0.1
    for i in range(n):
        sphere = pv.Sphere(radius=node_size, center=node_positions[i], theta_resolution=30, phi_resolution=30)
        actor = plotter.add_mesh(sphere, color='#333333', show_edges=False,
                        opacity=0.85, smooth_shading=True,
                        ambient=0.3, diffuse=0.6, specular=0.3, specular_power=30,
                        pbr=False, metallic=0.0, roughness=0.5)
        actor.GetProperty().SetLighting(True)
    
    label_text = None
    label_3d_pos = None
    
    if closest_edge is not None:
        u, v = closest_edge
        edge = tuple(sorted((u, v)))
        
        if u in pos_3d and v in pos_3d:
            start_pos = np.array(pos_3d[u])
            end_pos = np.array(pos_3d[v])
            mid_point = (start_pos + end_pos) / 2
            label_3d_pos = mid_point
        elif closest_edge in edge_midpoints:
            label_3d_pos = edge_midpoints[closest_edge]
        elif (v, u) in edge_midpoints:
            label_3d_pos = edge_midpoints[(v, u)]
        else:
            label_3d_pos = list(edge_midpoints.values())[0] if edge_midpoints else np.array([0, 0, 0])
        
        if label_type == "scalar":
            val = edge_labels.get(edge, 0.0)
            val_str = f"{val:.2f}"
            label_text = f"${val_str}$"
        elif label_type == "vector":
            vec = edge_labels.get(edge, np.zeros(3))
            v_strs = [f"{val:.2f}" for val in vec]
            label_text = f"$({v_strs[0]}, {v_strs[1]}, {v_strs[2]})$"
        elif label_type == "matrix":
            M = edge_labels.get(edge, np.zeros((3, 3)))
            matrix_rows = []
            for i in range(3):
                row_vals = [f"{M[i,j]:.2f}" for j in range(3)]
                matrix_rows.append(" & ".join(row_vals))
            matrix_str = r"$\left(\begin{array}{@{}c@{\hspace{0.6em}}c@{\hspace{0.6em}}c@{}}" + "\\\\".join(matrix_rows) + r"\end{array}\right)$"
            label_text = matrix_str
    
    plotter.renderer.SetUseFXAA(False)
    plotter.renderer.SetBackgroundAlpha(0.0)
    plotter.renderer.SetAutomaticLightCreation(False)
    plotter.renderer.SetTwoSidedLighting(True)
    
    plotter.render()
    
    import tempfile
    temp_img_path = output_path.parent / f"temp_{output_path.name}"
    plotter.screenshot(str(temp_img_path), transparent_background=True)
    plotter.close()
    
    if label_text is not None and label_3d_pos is not None:
        import matplotlib.pyplot as plt
        from PIL import Image
        
        img_3d = Image.open(temp_img_path)
        img_array = np.array(img_3d)
        img_width, img_height = img_3d.size
        
        red_mask = (img_array[:, :, 0] > 200) & (img_array[:, :, 1] < 100) & (img_array[:, :, 2] < 100)
        
        if np.any(red_mask):
            red_coords = np.argwhere(red_mask)
            if len(red_coords) > 0:
                center_y, center_x = red_coords.mean(axis=0)
                x_screen = center_x + img_width * 0.05
                y_screen = center_y - img_height * 0.02
            else:
                x_screen = img_width / 2
                y_screen = img_height / 2
        else:
            x_screen = img_width / 2
            y_screen = img_height / 2
        
        fig, ax = plt.subplots(figsize=(img_width/FIGURE_DPI, img_height/FIGURE_DPI), dpi=FIGURE_DPI)
        ax.imshow(img_3d, origin='upper')
        ax.axis('off')
        
        if label_type == "scalar":
            fontsize = 24
        elif label_type == "vector":
            fontsize = 22
        elif label_type == "matrix":
            fontsize = 20
        else:
            fontsize = 24
        
        text_obj = ax.text(x_screen, y_screen, label_text, fontsize=fontsize, 
                          ha='left', va='center', fontweight='bold', color='red',
                          transform=ax.transData)
        
        fig.canvas.draw()
        bbox = text_obj.get_window_extent(renderer=fig.canvas.renderer)
        bbox_data = bbox.transformed(ax.transData.inverted())
        
        text_center_x = (bbox_data.x0 + bbox_data.x1) / 2
        text_center_y = (bbox_data.y0 + bbox_data.y1) / 2
        
        ax_xlim = ax.get_xlim()
        ax_ylim = ax.get_ylim()
        pad_x_pixels = 15
        pad_y_pixels = 15
        pad_x = pad_x_pixels * (ax_xlim[1] - ax_xlim[0]) / img_width
        pad_y = pad_y_pixels * (ax_ylim[1] - ax_ylim[0]) / img_height
        
        box_width = bbox_data.width + 2 * pad_x
        box_height = bbox_data.height + 2 * pad_y
        box_x = text_center_x - box_width / 2
        box_y = text_center_y - box_height / 2
        
        from matplotlib.patches import FancyBboxPatch
        
        box_face = FancyBboxPatch(
            (box_x, box_y), box_width, box_height,
            boxstyle="round,pad=0.01",
            facecolor='white',
            edgecolor='none',
            alpha=0.6,
            zorder=1
        )
        ax.add_patch(box_face)
        
        box_border = FancyBboxPatch(
            (box_x, box_y), box_width, box_height,
            boxstyle="round,pad=0.01",
            facecolor='none',
            edgecolor='black',
            linewidth=2.0,
            alpha=1.0,
            zorder=2
        )
        ax.add_patch(box_border)
        
        text_obj.set_zorder(3)
        
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=FIGURE_DPI, 
                   facecolor='none', transparent=True)
        plt.close(fig)
        
        temp_img_path.unlink()
    else:
        import matplotlib.pyplot as plt
        from PIL import Image
        
        img_3d = Image.open(temp_img_path)
        img_width, img_height = img_3d.size
        
        fig, ax = plt.subplots(figsize=(img_width/FIGURE_DPI, img_height/FIGURE_DPI), dpi=FIGURE_DPI)
        ax.imshow(img_3d, origin='upper')
        ax.axis('off')
        
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=FIGURE_DPI, 
                   facecolor='none', transparent=True)
        plt.close(fig)
        
        temp_img_path.unlink()
