"""Small-world network visualization with four labeling schemes and dendrograms."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import networkx as nx
import numpy as np
from scipy.linalg import eigh
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from collections import defaultdict
from matplotlib.patches import FancyBboxPatch

try:
    from gudhi import SimplexTree
except ImportError:
    SimplexTree = None

try:
    import pyvista as pv
    try:
        pv.start_xvfb()
    except (OSError, AttributeError):
        pass
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

plt.rcParams.update({"font.family": "Times New Roman"})
try:
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}"
    })
except:
    plt.rcParams.update({"text.usetex": False})

FIGURE_DPI = 300
np.random.seed(14)


def compute_laplacian_spectrum(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute combinatorial Laplacian L = D - A and its eigendecomposition."""
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G, nodelist=range(n)).toarray()
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    eigenvalues, eigenvectors = eigh(L)
    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]
    for k in range(n):
        norm = np.linalg.norm(eigenvectors[:, k])
        if norm > 1e-10:
            eigenvectors[:, k] = eigenvectors[:, k] / norm
    return L, eigenvalues, eigenvectors


def create_graph() -> nx.Graph:
    """Create Watts-Strogatz small-world graph."""
    return nx.watts_strogatz_graph(n=30, k=4, p=0.3, seed=14)


def compute_3d_layout(G: nx.Graph) -> Dict[int, Tuple[float, float, float]]:
    """Compute 3D layout using Kamada-Kawai extended to 3D."""
    pos_2d = nx.kamada_kawai_layout(G, weight=None, dim=2)
    pos_3d = {}
    z_coords = nx.spring_layout(G, dim=1, k=0.5, seed=42)
    for i in range(G.number_of_nodes()):
        x, y = pos_2d[i]
        z = list(z_coords[i].values())[0] if z_coords[i] else 0.0
        pos_3d[i] = (x, y, z)
    return pos_3d


def find_closest_edge_to_camera(
    G: nx.Graph,
    pos_3d: Dict[int, Tuple[float, float, float]],
    camera_position: Tuple[float, float, float] = (5.0, 5.0, 5.0)
) -> Tuple[int, int]:
    """Find the edge closest to the camera position."""
    camera = np.array(camera_position)
    min_dist = float('inf')
    closest_edge = None
    for u, v in G.edges():
        x1, y1, z1 = pos_3d[u]
        x2, y2, z2 = pos_3d[v]
        mid_point = np.array([(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2])
        dist = np.linalg.norm(mid_point - camera)
        if dist < min_dist:
            min_dist = dist
            closest_edge = (u, v)
    return closest_edge if closest_edge else list(G.edges())[0]


def compute_h1_persistence_gudhi(
    G: nx.Graph,
    edge_labels: Dict[Tuple[int, int], Any],
    poset_order_func: Callable,
    scalarize_func: Callable[[Any], float]
) -> List[Tuple[float, float]]:
    """Compute H1 persistence using gudhi SimplexTree with poset-based filtration."""
    if SimplexTree is None:
        return []
    st = SimplexTree()
    n = G.number_of_nodes()
    for i in range(n):
        st.insert([i], 0.0)
    for u, v in G.edges():
        edge = tuple(sorted((u, v)))
        label = edge_labels[edge]
        filt_val = scalarize_func(label)
        st.insert([u, v], filt_val)
    for u in range(n):
        for v in range(u + 1, n):
            if G.has_edge(u, v):
                for w in range(v + 1, n):
                    if G.has_edge(u, w) and G.has_edge(v, w):
                        edges = [
                            tuple(sorted((u, v))),
                            tuple(sorted((u, w))),
                            tuple(sorted((v, w)))
                        ]
                        edge_labels_list = [edge_labels[e] for e in edges]
                        max_label = edge_labels_list[0]
                        for lab in edge_labels_list[1:]:
                            if poset_order_func(max_label, lab):
                                max_label = lab
                        filt_val = scalarize_func(max_label)
                        st.insert([u, v, w], filt_val)
    st.compute_persistence()
    persistence_result = st.persistence()
    h1_pairs = []
    max_filt = 0.0
    if edge_labels:
        all_filt_vals = [scalarize_func(lab) for lab in edge_labels.values()]
        max_filt = max(all_filt_vals) if all_filt_vals else 1.0
    if max_filt == 0.0:
        max_filt = 1.0
    try:
        persistence_pairs = st.persistence_pairs()
        for (birth_set, death_set) in persistence_pairs:
            if len(birth_set) == 2:
                birth_filt = st.filtration(birth_set)
                if len(death_set) == 0:
                    birth_val = float(birth_filt)
                    death_val = max_filt * 3.0 if max_filt > 0 else 10.0
                    h1_pairs.append((birth_val, death_val))
                elif len(death_set) == 3:
                    death_filt = st.filtration(death_set)
                    birth_val = float(birth_filt)
                    death_val = float(death_filt)
                    h1_pairs.append((birth_val, death_val))
    except:
        for (dim, (birth_filt, death_filt)) in persistence_result:
            if dim == 1:
                birth_val = float(birth_filt)
                if death_filt == float('inf') or np.isinf(death_filt):
                    death_val = max_filt * 3.0 if max_filt > 0 else 10.0
                else:
                    death_val = float(death_filt)
                h1_pairs.append((birth_val, death_val))
    return h1_pairs


def example_1_laplacian_scalar(G: nx.Graph) -> Tuple[Dict, List[Tuple[float, float]]]:
    """Example 1: Real scalar labels via Laplacian pseudoinverse diagonal."""
    n = G.number_of_nodes()
    L, eigenvalues, eigenvectors = compute_laplacian_spectrum(G)
    h = {}
    for i in range(n):
        h[i] = 0.0
        for k in range(1, n):
            if eigenvalues[k] > 1e-10:
                h[i] += (1.0 / eigenvalues[k]) * (eigenvectors[i, k] ** 2)
    edge_labels = {}
    for u, v in G.edges():
        edge = tuple(sorted((u, v)))
        edge_labels[edge] = max(h[u], h[v])
    def poset_order(p1, p2):
        return p1 <= p2
    def scalarize(p):
        return float(p)
    persistence_points = compute_h1_persistence_gudhi(G, edge_labels, poset_order, scalarize)
    return edge_labels, persistence_points


def example_2_heat_kernel_vector(G: nx.Graph) -> Tuple[Dict, List[Tuple[np.ndarray, np.ndarray]]]:
    """Example 2: Real vector labels via heat kernel diagonal at three time scales."""
    n = G.number_of_nodes()
    L, eigenvalues, eigenvectors = compute_laplacian_spectrum(G)
    lambda_1 = eigenvalues[1] if eigenvalues[1] > 1e-10 else 1.0
    c1, c2, c3 = 0.1, 0.5, 1.0
    t1 = c1 / lambda_1
    t2 = c2 / lambda_1
    t3 = c3 / lambda_1
    def heat_kernel_diagonal(t: float) -> np.ndarray:
        diag = np.zeros(n)
        for i in range(n):
            for k in range(n):
                diag[i] += np.exp(-t * eigenvalues[k]) * (eigenvectors[i, k] ** 2)
        return diag
    K_t1_diag = heat_kernel_diagonal(t1)
    K_t2_diag = heat_kernel_diagonal(t2)
    K_t3_diag = heat_kernel_diagonal(t3)
    d = {}
    for i in range(n):
        d[i] = np.array([K_t1_diag[i], K_t2_diag[i], K_t3_diag[i]], dtype=float)
    edge_labels = {}
    for u, v in G.edges():
        edge = tuple(sorted((u, v)))
        edge_labels[edge] = np.array([
            max(d[u][m], d[v][m]) for m in range(3)
        ])
    def poset_order(p1, p2):
        return np.all(p1 <= p2)
    def scalarize(p):
        return float(np.linalg.norm(p))
    persistence_scalar = compute_h1_persistence_gudhi(G, edge_labels, poset_order, scalarize)
    all_labels_list = list(edge_labels.values())
    persistence_points = []
    for birth_scalar, death_scalar in persistence_scalar:
        birth_idx = min(range(len(all_labels_list)),
                       key=lambda i: abs(np.linalg.norm(all_labels_list[i]) - birth_scalar))
        death_idx = min(range(len(all_labels_list)),
                       key=lambda i: abs(np.linalg.norm(all_labels_list[i]) - death_scalar))
        persistence_points.append((all_labels_list[birth_idx], all_labels_list[death_idx]))
    return edge_labels, persistence_points


def compute_laplacian_heat_profiles(
    G: nx.Graph,
    v0: int,
    T: float,
    n_samples: int = 101
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """Compute heat profiles using Laplacian heat flow (spectral)."""
    n = G.number_of_nodes()
    L, eigenvalues, eigenvectors = compute_laplacian_spectrum(G)
    t_samples = np.linspace(0, T, n_samples)
    profiles = {}
    for i in range(n):
        h_i = np.zeros(n_samples)
        for t_idx, t in enumerate(t_samples):
            for k in range(n):
                h_i[t_idx] += np.exp(-t * eigenvalues[k]) * eigenvectors[i, k] * eigenvectors[v0, k]
        profiles[i] = h_i
    return profiles, t_samples


def create_homeomorphism_from_difference(f_diff: np.ndarray, t_samples: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Create homeomorphism from difference function f_e(t) = h_u(t) - h_v(t)."""
    T = t_samples[-1]
    n_samples = len(t_samples)
    m_e = np.min(f_diff)
    g_e = f_diff - m_e + eps
    dt = t_samples[1] - t_samples[0] if n_samples > 1 else 1.0
    G_e = np.zeros(n_samples)
    for i in range(1, n_samples):
        G_e[i] = G_e[i-1] + 0.5 * dt * (g_e[i-1] + g_e[i])
    M_e = G_e[-1]
    if M_e > 1e-10:
        psi_e = G_e / M_e
    else:
        psi_e = t_samples / T
    s_samples = np.linspace(0, 1, n_samples)
    t_at_s = T * s_samples
    varphi_e = np.interp(t_at_s, t_samples, psi_e)
    return varphi_e


def example_3_laplacian_homeomorphism(G: nx.Graph) -> Tuple[Dict, List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Example 3: Homeomorphism-valued labels via Laplacian heat flow."""
    n = G.number_of_nodes()
    L, eigenvalues, eigenvectors = compute_laplacian_spectrum(G)
    i0 = 0
    lambda_1 = eigenvalues[1] if eigenvalues[1] > 1e-10 else 1.0
    c = 2.0
    T = c / lambda_1
    profiles, t_samples = compute_laplacian_heat_profiles(G, v0=i0, T=T, n_samples=101)
    eps = 1e-3
    edge_labels = {}
    for u, v in G.edges():
        edge = tuple(sorted((u, v)))
        f_e = profiles[u] - profiles[v]
        phi_e = create_homeomorphism_from_difference(f_e, t_samples, eps=eps)
        edge_labels[edge] = phi_e
    def poset_order(phi1, phi2):
        return np.all(phi1 <= phi2)
    def scalarize(phi):
        identity = t_samples / t_samples[-1]
        return float(np.max(np.abs(phi - identity)))
    persistence_scalar = compute_h1_persistence_gudhi(G, edge_labels, poset_order, scalarize)
    all_labels_list = list(edge_labels.values())
    persistence_points = []
    for birth_scalar, death_scalar in persistence_scalar:
        birth_idx = min(range(len(all_labels_list)),
                       key=lambda i: abs(scalarize(all_labels_list[i]) - birth_scalar))
        death_idx = min(range(len(all_labels_list)),
                       key=lambda i: abs(scalarize(all_labels_list[i]) - death_scalar))
        persistence_points.append((all_labels_list[birth_idx], all_labels_list[death_idx]))
    t_samples_normalized = np.linspace(0, 1, len(t_samples))
    return edge_labels, persistence_points, t_samples_normalized


def example_4_spectral_matrix(G: nx.Graph) -> Tuple[Dict, List[Tuple[np.ndarray, np.ndarray]]]:
    """Example 4: Matrix-valued labels via spectral 3D embedding and edge differences."""
    n = G.number_of_nodes()
    L, eigenvalues, eigenvectors = compute_laplacian_spectrum(G)
    lambda_1 = eigenvalues[1] if eigenvalues[1] > 1e-10 else 1.0
    c1, c2, c3 = 0.1, 0.5, 1.0
    t1 = c1 / lambda_1
    t2 = c2 / lambda_1
    t3 = c3 / lambda_1
    def heat_kernel_diagonal(t: float) -> np.ndarray:
        diag = np.zeros(n)
        for i in range(n):
            for k in range(n):
                diag[i] += np.exp(-t * eigenvalues[k]) * (eigenvectors[i, k] ** 2)
        return diag
    K_t1_diag = heat_kernel_diagonal(t1)
    K_t2_diag = heat_kernel_diagonal(t2)
    K_t3_diag = heat_kernel_diagonal(t3)
    d = {}
    for i in range(n):
        d[i] = np.array([K_t1_diag[i], K_t2_diag[i], K_t3_diag[i]], dtype=float)
    edge_labels = {}
    for u, v in G.edges():
        edge = tuple(sorted((u, v)))
        Delta_e = d[u] - d[v]
        M_e = np.outer(Delta_e, Delta_e)
        edge_labels[edge] = M_e
    def poset_order(M1, M2):
        diff = M2 - M1
        eigenvalues_diff = np.linalg.eigvalsh(diff)
        return np.all(eigenvalues_diff >= -1e-10)
    def scalarize(M):
        return float(np.trace(M))
    persistence_scalar = compute_h1_persistence_gudhi(G, edge_labels, poset_order, scalarize)
    all_labels_list = list(edge_labels.values())
    persistence_points = []
    for birth_scalar, death_scalar in persistence_scalar:
        birth_idx = min(range(len(all_labels_list)),
                       key=lambda i: abs(scalarize(all_labels_list[i]) - birth_scalar))
        death_idx = min(range(len(all_labels_list)),
                       key=lambda i: abs(scalarize(all_labels_list[i]) - death_scalar))
        persistence_points.append((all_labels_list[birth_idx], all_labels_list[death_idx]))
    return edge_labels, persistence_points


def visualize_graph_scalar(
    G: nx.Graph,
    pos: Dict[int, Tuple[float, float]],
    edge_labels: Dict[Tuple[int, int], float],
    output_path: Path
) -> None:
    """Visualize graph in 3D with PyVista, showing scalar labels on 4 maximally separated edges."""
    if not PYVISTA_AVAILABLE:
        raise ImportError("PyVista is required for 3D visualization")
    pos_3d = {}
    n = G.number_of_nodes()
    pos_z = nx.spring_layout(G, dim=2, k=0.5, seed=42)
    for i in range(n):
        x, y = pos[i]
        z = pos_z[i][1] if isinstance(pos_z[i], (list, tuple, np.ndarray)) and len(pos_z[i]) > 1 else 0.0
        pos_3d[i] = (x, y, z)
    from pyvista_helpers import visualize_graph_3d_pyvista
    visualize_graph_3d_pyvista(G, pos_3d, edge_labels, output_path, label_type="scalar")


def visualize_graph_vector(
    G: nx.Graph,
    pos: Dict[int, Tuple[float, float]],
    edge_labels: Dict[Tuple[int, int], np.ndarray],
    output_path: Path
) -> None:
    """Visualize graph in 3D with PyVista, showing vector label only on closest edge."""
    if not PYVISTA_AVAILABLE:
        raise ImportError("PyVista is required for 3D visualization")
    pos_3d = {}
    n = G.number_of_nodes()
    pos_z = nx.spring_layout(G, dim=2, k=0.5, seed=42)
    for i in range(n):
        x, y = pos[i]
        z = pos_z[i][1] if isinstance(pos_z[i], (list, tuple, np.ndarray)) and len(pos_z[i]) > 1 else 0.0
        pos_3d[i] = (x, y, z)
    from pyvista_helpers import visualize_graph_3d_pyvista
    visualize_graph_3d_pyvista(G, pos_3d, edge_labels, output_path, label_type="vector")


def visualize_graph_homeomorphism(
    G: nx.Graph,
    pos: Dict[int, Tuple[float, float]],
    edge_labels: Dict[Tuple[int, int], np.ndarray],
    t_samples: np.ndarray,
    output_path: Path
) -> None:
    """Visualize graph in 3D with PyVista, showing homeomorphism function curves on 4 maximally separated edges."""
    if not PYVISTA_AVAILABLE:
        raise ImportError("PyVista is required for 3D visualization")
    pos_3d = {}
    n = G.number_of_nodes()
    pos_z = nx.spring_layout(G, dim=2, k=0.5, seed=42)
    for i in range(n):
        x, y = pos[i]
        z = pos_z[i][1] if isinstance(pos_z[i], (list, tuple, np.ndarray)) and len(pos_z[i]) > 1 else 0.0
        pos_3d[i] = (x, y, z)
    from pyvista_helpers import visualize_graph_3d_pyvista
    visualize_graph_3d_pyvista(G, pos_3d, edge_labels, output_path, label_type="homeomorphism", t_samples=t_samples)


def visualize_graph_matrix(
    G: nx.Graph,
    pos: Dict[int, Tuple[float, float]],
    edge_labels: Dict[Tuple[int, int], np.ndarray],
    output_path: Path
) -> None:
    """Visualize graph in 3D with PyVista, showing matrix labels on 4 maximally separated edges."""
    if not PYVISTA_AVAILABLE:
        raise ImportError("PyVista is required for 3D visualization")
    pos_3d = {}
    n = G.number_of_nodes()
    pos_z = nx.spring_layout(G, dim=2, k=0.5, seed=42)
    for i in range(n):
        x, y = pos[i]
        z = pos_z[i][1] if isinstance(pos_z[i], (list, tuple, np.ndarray)) and len(pos_z[i]) > 1 else 0.0
        pos_3d[i] = (x, y, z)
    from pyvista_helpers import visualize_graph_3d_pyvista
    visualize_graph_3d_pyvista(G, pos_3d, edge_labels, output_path, label_type="matrix")


def visualize_pd_classical(
    persistence_points: List[Tuple[float, float]],
    output_path: Path
) -> None:
    """Visualize classical persistence diagram (R^2)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=FIGURE_DPI)
    if len(persistence_points) == 0:
        ax.text(0.5, 0.5, r"$\emptyset$",
               ha="center", va="center", transform=ax.transAxes, fontsize=20)
        ax.set_xlabel("Birth", fontsize=12)
        ax.set_ylabel("Death", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=FIGURE_DPI)
        plt.close(fig)
        return
    births = [p[0] for p in persistence_points]
    deaths = [p[1] for p in persistence_points]
    ax.scatter(births, deaths, s=150, alpha=0.8, color="#2c6df2",
              edgecolors="#000000", linewidths=1.5, zorder=3)
    max_val = max(max(births), max(deaths)) if births else 1.0
    ax.plot([0, max_val], [0, max_val], "r--", alpha=0.5, linewidth=2, zorder=1)
    ax.set_xlabel("Birth", fontsize=12)
    ax.set_ylabel("Death", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close(fig)


def compute_persistence_lifetime(birth: Any, death: Any, value_type: str) -> float:
    """Compute the persistence lifetime (distance to diagonal) for an interval [b, d]."""
    if value_type == "scalar":
        return abs(float(death) - float(birth)) / 2.0
    elif value_type == "vector":
        b_vec = np.array(birth)
        d_vec = np.array(death)
        return float(np.linalg.norm(d_vec - b_vec)) / 2.0
    elif value_type == "homeomorphism":
        phi_b = np.array(birth)
        phi_d = np.array(death)
        return float(np.max(np.abs(phi_d - phi_b))) / 2.0
    elif value_type == "matrix":
        M_b = np.array(birth)
        M_d = np.array(death)
        return float(np.linalg.norm(M_d - M_b, ord='fro')) / 2.0
    else:
        return abs(float(death) - float(birth)) / 2.0


def compute_interval_distance(birth1: Any, death1: Any, birth2: Any, death2: Any, value_type: str) -> float:
    """Compute d1 (1-strengthened metric) distance between two persistence intervals."""
    if value_type == "scalar":
        dist_birth = abs(float(birth1) - float(birth2))
        dist_death = abs(float(death1) - float(death2))
    elif value_type == "vector":
        dist_birth = float(np.linalg.norm(np.array(birth1) - np.array(birth2)))
        dist_death = float(np.linalg.norm(np.array(death1) - np.array(death2)))
    elif value_type == "homeomorphism":
        phi1_b = np.array(birth1)
        phi2_b = np.array(birth2)
        phi1_d = np.array(death1)
        phi2_d = np.array(death2)
        dist_birth = float(np.max(np.abs(phi1_b - phi2_b)))
        dist_death = float(np.max(np.abs(phi1_d - phi2_d)))
    elif value_type == "matrix":
        M1_b = np.array(birth1)
        M2_b = np.array(birth2)
        M1_d = np.array(death1)
        M2_d = np.array(death2)
        dist_birth = float(np.linalg.norm(M1_b - M2_b, ord='fro'))
        dist_death = float(np.linalg.norm(M1_d - M2_d, ord='fro'))
    else:
        dist_birth = abs(float(birth1) - float(birth2))
        dist_death = abs(float(death1) - float(death2))
    d_direct = max(dist_birth, dist_death)
    d_to_diag_1 = compute_persistence_lifetime(birth1, death1, value_type)
    d_to_diag_2 = compute_persistence_lifetime(birth2, death2, value_type)
    d_via_diagonal = d_to_diag_1 + d_to_diag_2
    return min(d_direct, d_via_diagonal)


def create_homeomorphism_visualization(
    phi_birth: np.ndarray,
    phi_death: np.ndarray,
    output_path: Path
) -> None:
    """Create visualization of two homeomorphism functions (birth and death) plotted on top of each other."""
    fig, ax = plt.subplots(figsize=(8, 2), dpi=FIGURE_DPI)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.axis('off')
    phi_birth = np.array(phi_birth)
    phi_death = np.array(phi_death)
    n_samples = len(phi_birth)
    s_samples = np.linspace(0, 1, n_samples)
    linewidth_val = 1.5
    ax.plot(s_samples, phi_birth, 'r-', linewidth=linewidth_val, transform=ax.transAxes)
    ax.plot(s_samples, phi_death, 'r--', linewidth=linewidth_val, transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 0], 'r-', linewidth=linewidth_val, transform=ax.transAxes)
    ax.plot([0, 0], [0, 1], 'r-', linewidth=linewidth_val, transform=ax.transAxes)
    tick_length = 0.01
    ax.plot([0, 0], [-tick_length, 0], 'r-', linewidth=linewidth_val, transform=ax.transAxes)
    ax.plot([1, 1], [-tick_length, 0], 'r-', linewidth=linewidth_val, transform=ax.transAxes)
    ax.plot([-tick_length, 0], [0, 0], 'r-', linewidth=linewidth_val, transform=ax.transAxes)
    ax.plot([-tick_length, 0], [1, 1], 'r-', linewidth=linewidth_val, transform=ax.transAxes)
    fig.canvas.draw()
    plot_x0, plot_y0 = 0.0, 0.0
    plot_x1, plot_y1 = 1.0, 1.0
    bbox_plot = ax.transAxes.transform([[plot_x0, plot_y0], [plot_x1, plot_y1]])
    plot_width_display = bbox_plot[1, 0] - bbox_plot[0, 0]
    plot_height_display = bbox_plot[1, 1] - bbox_plot[0, 1]
    pad_x_pixels = 25
    pad_y_pixels = 70
    pad_x = pad_x_pixels / (fig.get_dpi() * fig.get_size_inches()[0])
    pad_y = pad_y_pixels / (fig.get_dpi() * fig.get_size_inches()[1])
    box_width = (plot_x1 - plot_x0) + 2 * pad_x
    box_height = (plot_y1 - plot_y0) + 2 * pad_y
    box_x = plot_x0 - pad_x
    box_y = plot_y0 - pad_y
    box_face = FancyBboxPatch(
        (box_x, box_y), box_width, box_height,
        boxstyle="round,pad=0.01",
        facecolor='white',
        edgecolor='none',
        alpha=0.6,
        zorder=0,
        clip_on=False,
        transform=ax.transAxes
    )
    ax.add_patch(box_face)
    box_border = FancyBboxPatch(
        (box_x, box_y), box_width, box_height,
        boxstyle="round,pad=0.01",
        facecolor='none',
        edgecolor='black',
        linewidth=2.0,
        alpha=1.0,
        zorder=1,
        clip_on=False,
        transform=ax.transAxes
    )
    ax.add_patch(box_border)
    for line in ax.lines:
        line.set_zorder(2)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=FIGURE_DPI,
               facecolor='none', transparent=True)
    plt.close(fig)


def generate_dendrogram_assets(
    G: nx.Graph,
    edge_labels: Dict[Tuple[int, int], Any],
    persistence_points: List[Tuple],
    assets_dir: Path,
    subfigure_num: int,
    poset_order_func: Callable = None,
    scalarize_func: Callable[[Any], float] = None,
    value_type: str = "scalar",
    show_multiplicities: bool = True,
    title: str = ""
) -> None:
    """Generate separate assets for dendrogram visualization."""
    subfigure_dir = assets_dir / f"subfigure_{subfigure_num}"
    subfigure_dir.mkdir(parents=True, exist_ok=True)
    if len(persistence_points) == 0:
        fig, ax = plt.subplots(figsize=(10, 8), dpi=FIGURE_DPI)
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        ax.text(0.5, 0.5, r"$\emptyset$",
               ha="center", va="center", transform=ax.transAxes, fontsize=20)
        ax.axis('off')
        plt.savefig(subfigure_dir / "dendrogram.png", bbox_inches="tight", dpi=FIGURE_DPI,
                   facecolor='none', transparent=True)
        plt.close(fig)
        return
    interval_groups = defaultdict(list)
    for birth, death in persistence_points:
        if value_type == "scalar":
            key = (float(birth), float(death))
        elif value_type == "vector":
            b_arr = np.array(birth)
            d_arr = np.array(death)
            key = (tuple(np.round(b_arr, 6)), tuple(np.round(d_arr, 6)))
        elif value_type == "homeomorphism":
            b_arr = np.array(birth)
            d_arr = np.array(death)
            key = (tuple(np.round(b_arr, 6)), tuple(np.round(d_arr, 6)))
        elif value_type == "matrix":
            b_arr = np.array(birth)
            d_arr = np.array(death)
            key = (tuple(np.round(b_arr.flatten(), 6)), tuple(np.round(d_arr.flatten(), 6)))
        else:
            key = (birth, death)
        interval_groups[key].append((birth, death))
    unique_intervals = []
    multiplicities = []
    for key, group in interval_groups.items():
        unique_intervals.append(group[0])
        multiplicities.append(len(group))
    if value_type == "scalar":
        delta_birth = 0.0
        delta_death = 0.0
    elif value_type == "vector":
        delta_birth = np.zeros(3)
        delta_death = np.zeros(3)
    elif value_type == "homeomorphism":
        n_samples = 101
        s_samples = np.linspace(0, 1, n_samples)
        delta_birth = s_samples
        delta_death = s_samples
    elif value_type == "matrix":
        delta_birth = np.zeros((3, 3))
        delta_death = np.zeros((3, 3))
    else:
        delta_birth = 0.0
        delta_death = 0.0
    unique_intervals.append((delta_birth, delta_death))
    multiplicities.append(float('inf'))
    n_intervals = len(unique_intervals)
    if n_intervals == 0:
        fig, ax = plt.subplots(figsize=(10, 8), dpi=FIGURE_DPI)
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        ax.axis('off')
        plt.savefig(subfigure_dir / "dendrogram.png", bbox_inches="tight", dpi=FIGURE_DPI,
                   facecolor='none', transparent=True)
        plt.close(fig)
        return
    dist_matrix = np.zeros((n_intervals, n_intervals))
    for i in range(n_intervals):
        for j in range(n_intervals):
            if i == j:
                dist_matrix[i, j] = 0.0
            else:
                b1, d1 = unique_intervals[i]
                b2, d2 = unique_intervals[j]
                dist_matrix[i, j] = compute_interval_distance(b1, d1, b2, d2, value_type)
    condensed_dist = squareform(dist_matrix)
    linkage_matrix = linkage(condensed_dist, method='complete')
    if value_type == "vector":
        middle_idx = min(int(len(unique_intervals) * 0.75) + 1, len(unique_intervals) - 1)
    else:
        middle_idx = len(unique_intervals) // 2
    selected_birth, selected_death = unique_intervals[middle_idx]
    selected_mult = multiplicities[middle_idx]
    delta_idx = n_intervals - 1
    leaf_labels = [''] * n_intervals
    leaf_labels[delta_idx] = r"$[A] \times \infty$"
    if selected_mult is not None and not np.isinf(selected_mult):
        mult_text = f" \\times {int(selected_mult)}"
    else:
        mult_text = ""
    if value_type == "scalar":
        birth_str = f"{float(selected_birth):.2f}"
        death_str = f"{float(selected_death):.2f}"
        selected_label = f"$[{birth_str}, {death_str}]{mult_text}$"
    elif value_type == "vector":
        b_vec = np.array(selected_birth)
        d_vec = np.array(selected_death)
        b_str = f"({', '.join([f'{x:.2f}' for x in b_vec])})"
        d_str = f"({', '.join([f'{x:.2f}' for x in d_vec])})"
        selected_label = f"$[{b_str}, {d_str}]{mult_text}$"
    elif value_type == "matrix":
        M_b = np.array(selected_birth)
        M_d = np.array(selected_death)
        def matrix_to_str(M):
            rows = []
            for i in range(M.shape[0]):
                row_vals = [f"{M[i,j]:.2f}" for j in range(M.shape[1])]
                rows.append(" & ".join(row_vals))
            return r"\left(\begin{array}{@{}c@{\hspace{0.4em}}c@{\hspace{0.4em}}c@{}}" + "\\\\".join(rows) + r"\end{array}\right)"
        birth_matrix = matrix_to_str(M_b)
        death_matrix = matrix_to_str(M_d)
        selected_label = r"$\left[" + birth_matrix + r",\," + death_matrix + r"\right]" + mult_text + r"$"
    else:
        selected_label = ""
    leaf_labels[middle_idx] = selected_label
    if value_type == "homeomorphism":
        fig, ax = plt.subplots(figsize=(21, 15), dpi=FIGURE_DPI)
    else:
        fig, ax = plt.subplots(figsize=(14, 10), dpi=FIGURE_DPI)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    dn = dendrogram(linkage_matrix, ax=ax,
                   labels=leaf_labels,
                   leaf_rotation=90,
                   leaf_font_size=55,
                   color_threshold=0,
                   above_threshold_color='black',
                   count_sort=False,
                   distance_sort=False)
    icoord = dn['icoord']
    dcoord = dn['dcoord']
    for line in list(ax.lines):
        if (line.get_linewidth() <= 2.0 and
            str(line.get_color()).lower() in ['black', 'k', '#000000']):
            line.remove()
    for xs, ys in zip(icoord, dcoord):
        ax.plot(xs, ys, 'k-', linewidth=4.0, solid_capstyle='round')
    ax.set_ylabel("VPD distance", fontsize=55)
    ax.tick_params(axis='y', labelsize=55)
    ax.set_xticks([])
    ax.tick_params(axis='x', labelbottom=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_linewidth(4.0)
    leaf_order = dn['leaves']
    diagonal_leaf_pos = None
    selected_leaf_pos = None
    for i, leaf_idx in enumerate(leaf_order):
        if leaf_idx == delta_idx:
            diagonal_leaf_pos = i + 1
        if leaf_idx == middle_idx:
            selected_leaf_pos = i + 1
    plt.savefig(subfigure_dir / "dendrogram.png", bbox_inches="tight", dpi=FIGURE_DPI,
               facecolor='none', transparent=True)
    plt.close(fig)
    positions_file = subfigure_dir / "leaf_positions.txt"
    with open(positions_file, 'w') as f:
        f.write(f"Leaf Positions for Subfigure {subfigure_num}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Diagonal [A] label: Leaf position {diagonal_leaf_pos} (from left, 1-indexed)\n")
        f.write(f"Selected interval label: Leaf position {selected_leaf_pos} (from left, 1-indexed)\n")
        f.write(f"\nTotal number of leaves: {len(leaf_order)}\n")
        f.write(f"\nNote: Leaf positions are counted from left to right, starting at 1.\n")
        f.write(f"      The diagonal is always the last interval added (index {delta_idx} in unique_intervals).\n")
        f.write(f"      The selected interval is at index {middle_idx} in unique_intervals.\n")
    def create_label_image(label_text: str, output_path: Path):
        """Create a separate image for a label (text + box)"""
        fig, ax = plt.subplots(figsize=(8, 2), dpi=FIGURE_DPI)
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        ax.axis('off')
        text_obj = ax.text(0.5, 0.5, label_text,
                          ha='center', va='center', fontsize=int(55 * 0.85),
                          color='red', fontweight='bold',
                          transform=ax.transAxes)
        fig.canvas.draw()
        bbox = text_obj.get_window_extent(renderer=fig.canvas.renderer)
        bbox_data = bbox.transformed(ax.transAxes.inverted())
        text_center_x = (bbox_data.x0 + bbox_data.x1) / 2
        text_center_y = (bbox_data.y0 + bbox_data.y1) / 2
        pad_x_pixels = 25
        pad_y_pixels = 70
        pad_x = pad_x_pixels / (fig.get_dpi() * fig.get_size_inches()[0])
        pad_y = pad_y_pixels / (fig.get_size_inches()[1] * fig.get_dpi())
        box_width = bbox_data.width + 2 * pad_x
        box_height = bbox_data.height + 2 * pad_y
        box_x = text_center_x - box_width / 2
        box_y = text_center_y - box_height / 2
        box_face = FancyBboxPatch(
            (box_x, box_y), box_width, box_height,
            boxstyle="round,pad=0.01",
            facecolor='white',
            edgecolor='none',
            alpha=0.6,
            zorder=1,
            clip_on=False,
            transform=ax.transAxes
        )
        ax.add_patch(box_face)
        box_border = FancyBboxPatch(
            (box_x, box_y), box_width, box_height,
            boxstyle="round,pad=0.01",
            facecolor='none',
            edgecolor='black',
            linewidth=2.0,
            alpha=1.0,
            zorder=2,
            clip_on=False,
            transform=ax.transAxes
        )
        ax.add_patch(box_border)
        text_obj.set_zorder(3)
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=FIGURE_DPI,
                   facecolor='none', transparent=True)
        plt.close(fig)
    create_label_image(r"$[A] \times \infty$", subfigure_dir / "diagonal_label.png")
    create_label_image(selected_label, subfigure_dir / "selected_interval_label.png")
    if value_type == "homeomorphism":
        create_homeomorphism_visualization(selected_birth, selected_death,
                                          subfigure_dir / "selected_interval_functions.png")


def main() -> None:
    """Generate all visualizations and compute bounds."""
    networks_dir = Path("results/figures/networks")
    diagrams_dir = Path("results/figures/diagrams")
    dendrograms_dir = Path("results/figures/dendrograms")
    assets_dir = Path("results/figures/dendrograms/assets")
    networks_dir.mkdir(parents=True, exist_ok=True)
    diagrams_dir.mkdir(parents=True, exist_ok=True)
    dendrograms_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    G = create_graph()
    pos = nx.kamada_kawai_layout(G, weight=None)
    edge_labels_1, persistence_1 = example_1_laplacian_scalar(G)
    visualize_graph_scalar(G, pos, edge_labels_1,
                          networks_dir / "example_1_fiedler_graph.png")
    visualize_pd_classical(persistence_1,
                          diagrams_dir / "example_1_fiedler_diagram.png")
    def poset_order_1(p1, p2):
        return p1 <= p2
    def scalarize_1(p):
        return float(p)
    generate_dendrogram_assets(G, edge_labels_1, persistence_1, assets_dir, 1,
                              poset_order_1, scalarize_1, value_type="scalar")
    edge_labels_2, persistence_2 = example_2_heat_kernel_vector(G)
    visualize_graph_vector(G, pos, edge_labels_2,
                          networks_dir / "example_2_vector_3d_graph.png")
    def poset_order_2(p1, p2):
        return np.all(p1 <= p2)
    def scalarize_2(p):
        return float(np.linalg.norm(p))
    generate_dendrogram_assets(G, edge_labels_2, persistence_2, assets_dir, 2,
                              poset_order_2, scalarize_2, value_type="vector")
    edge_labels_3, persistence_3, t_samples = example_3_laplacian_homeomorphism(G)
    visualize_graph_homeomorphism(G, pos, edge_labels_3, t_samples,
                                 networks_dir / "example_3_homeomorphism_graph.png")
    def poset_order_3(phi1, phi2):
        return np.all(phi1 <= phi2)
    def scalarize_3(phi):
        identity = np.linspace(0, 1, len(phi))
        return float(np.max(np.abs(phi - identity)))
    generate_dendrogram_assets(G, edge_labels_3, persistence_3, assets_dir, 3,
                              poset_order_3, scalarize_3, value_type="homeomorphism")
    edge_labels_4, persistence_4 = example_4_spectral_matrix(G)
    visualize_graph_matrix(G, pos, edge_labels_4,
                          networks_dir / "example_4_matrix_graph.png")
    def poset_order_4(M1, M2):
        diff = M2 - M1
        eigenvalues_diff = np.linalg.eigvalsh(diff)
        return np.all(eigenvalues_diff >= -1e-10)
    def scalarize_4(M):
        return float(np.trace(M))
    generate_dendrogram_assets(G, edge_labels_4, persistence_4, assets_dir, 4,
                              poset_order_4, scalarize_4, value_type="matrix")


if __name__ == "__main__":
    main()
