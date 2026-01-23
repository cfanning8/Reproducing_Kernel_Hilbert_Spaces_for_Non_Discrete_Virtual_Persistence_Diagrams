"""
Persistence diagram computation from graphs and point clouds.
"""

import numpy as np
from typing import Optional, List, Tuple
from scipy.sparse import csr_matrix
import ripser
from gudhi import AlphaComplex

from .theoretical.heat_flow import (
    heat_vertex_function, 
    heat_edge_weights,
    lower_star_filtration_value
)
from .theoretical.virtual_diagrams import PersistenceDiagram


def graph_to_adjacency(graph_data) -> np.ndarray:
    """
    Convert graph data to adjacency matrix.
    
    Args:
        graph_data: PyTorch Geometric Data object or dict with 'edge_index'
    
    Returns:
        Adjacency matrix, shape (n, n)
    """
    if hasattr(graph_data, 'edge_index'):
        # PyTorch Geometric
        edge_index = graph_data.edge_index.cpu().numpy()
        num_nodes = graph_data.num_nodes if hasattr(graph_data, 'num_nodes') else edge_index.max() + 1
    elif isinstance(graph_data, dict) and 'edge_index' in graph_data:
        edge_index = graph_data['edge_index']
        num_nodes = graph_data.get('num_nodes', edge_index.max() + 1)
    else:
        raise ValueError("Graph data must have 'edge_index' attribute")
    
    # Build adjacency matrix
    adj = np.zeros((num_nodes, num_nodes))
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        adj[u, v] = 1
        adj[v, u] = 1  # Undirected
    
    return adj


def compute_clique_complex(adjacency: np.ndarray, max_dim: int = 2) -> List[List[Tuple]]:
    """
    Compute clique complex (flag complex) up to max_dim.
    
    Args:
        adjacency: Adjacency matrix
        max_dim: Maximum dimension of simplices
    
    Returns:
        List of simplices by dimension: [0-simplices, 1-simplices, 2-simplices, ...]
    """
    n = adjacency.shape[0]
    simplices = [[] for _ in range(max_dim + 1)]
    
    # 0-simplices (vertices)
    simplices[0] = [(i,) for i in range(n)]
    
    # 1-simplices (edges)
    simplices[1] = []
    for u in range(n):
        for v in range(u + 1, n):
            if adjacency[u, v] > 0:
                simplices[1].append((u, v))
    
    # Higher-dimensional simplices (cliques)
    if max_dim >= 2:
        # 2-simplices (triangles)
        simplices[2] = []
        for u in range(n):
            for v in range(u + 1, n):
                if adjacency[u, v] > 0:
                    for w in range(v + 1, n):
                        if adjacency[u, w] > 0 and adjacency[v, w] > 0:
                            simplices[2].append((u, v, w))
    
    # For dim > 2, would need recursive clique enumeration
    # For now, stop at triangles
    
    return simplices


def lower_star_filtration_graph(graph_data, 
                                tau: float = 1.0,
                                heat_method: str = 'content',
                                max_dim: int = 2) -> PersistenceDiagram:
    """
    Compute persistence diagram from graph using lower-star filtration on clique complex.
    
    Uses heat-derived vertex function: f_tau(v) from heat kernel H(tau).
    
    Args:
        graph_data: Graph data (PyTorch Geometric or dict)
        tau: Heat diffusion time
        heat_method: 'content' or 'diffusion' for vertex function
        max_dim: Maximum dimension of simplices (default: 2 for H₁ computation)
                 For H₁, 2-skeleton (up to triangles) is sufficient.
                 Higher dimensions increase computational cost (O(n³) for triangles).
    
    Returns:
        Persistence diagram (H1 only for now)
    """
    # Convert to adjacency
    adjacency = graph_to_adjacency(graph_data)
    
    # Compute heat-derived vertex function (with rank normalization)
    vertex_function = heat_vertex_function(adjacency, tau=tau, method=heat_method, normalize='rank')
    
    # Compute clique complex
    simplices = compute_clique_complex(adjacency, max_dim=max_dim)
    
    # Assign filtration values
    filtration_values = {}
    
    # 0-simplices: vertex function value
    for v in simplices[0]:
        filtration_values[v] = vertex_function[v[0]]
    
    # 1-simplices: max of endpoint vertex functions
    for edge in simplices[1]:
        filtration_values[edge] = max(vertex_function[edge[0]], vertex_function[edge[1]])
    
    # 2-simplices: max of triangle vertex functions
    for triangle in simplices[2]:
        filtration_values[triangle] = max(vertex_function[v] for v in triangle)
    
    # Build filtration and compute persistence
    # For now, use ripser on a distance matrix derived from filtration
    # This is a simplified approach; full implementation would use gudhi SimplexTree
    
    # Create distance matrix: d(u,v) = max(f(u), f(v)) if edge exists, else inf
    n = adjacency.shape[0]
    dist_matrix = np.full((n, n), np.inf)
    np.fill_diagonal(dist_matrix, vertex_function)
    
    for edge in simplices[1]:
        u, v = edge
        dist_matrix[u, v] = filtration_values[edge]
        dist_matrix[v, u] = filtration_values[edge]
    
    # Use ripser on distance matrix (suboptimal but works)
    # Filter out infinite distances for ripser
    max_finite = dist_matrix[dist_matrix < np.inf].max() if np.any(dist_matrix < np.inf) else 1.0
    dist_matrix[dist_matrix == np.inf] = max_finite * 2
    
    # Compute persistence
    result = ripser.ripser(dist_matrix, maxdim=1, metric='precomputed')
    diagrams = result['dgms']
    
    # Extract H1 diagram
    if len(diagrams) > 1 and len(diagrams[1]) > 0:
        h1_points = diagrams[1]
        # Filter out infinite persistence
        h1_points = h1_points[np.isfinite(h1_points[:, 1]), :]
        return PersistenceDiagram(h1_points)
    else:
        return PersistenceDiagram(np.empty((0, 2)))


def compute_persistence_diagram_alpha(X: np.ndarray, 
                                     max_points: Optional[int] = None) -> PersistenceDiagram:
    """
    Compute persistence diagram from point cloud using alpha complex.
    
    Args:
        X: Point cloud, shape (n, dim)
        max_points: Maximum number of points (subsample if needed)
    
    Returns:
        Persistence diagram
    """
    if max_points is not None and len(X) > max_points:
        # Subsample
        rng = np.random.RandomState(14)
        indices = rng.choice(len(X), max_points, replace=False)
        X = X[indices]
    
    # Use gudhi AlphaComplex
    alpha_complex = AlphaComplex(points=X)
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=np.inf)
    
    # Get persistence
    persistence = simplex_tree.persistence()
    
    # Extract H1 diagram
    h1_points = []
    for dim, (birth, death) in persistence:
        if dim == 1 and np.isfinite(death):
            h1_points.append([birth, death])
    
    if len(h1_points) > 0:
        return PersistenceDiagram(np.array(h1_points))
    else:
        return PersistenceDiagram(np.empty((0, 2)))


def compute_persistence_diagram(X, 
                                method: str = 'auto',
                                max_points: Optional[int] = None,
                                is_graph: bool = False,
                                tau: float = 1.0,
                                heat_method: str = 'content',
                                max_dim: int = 2) -> np.ndarray:
    """
    Compute persistence diagram from input data.
    
    Args:
        X: Input data (point cloud array or graph data)
        method: 'ripser', 'gudhi', 'alpha', 'auto', or 'graph'
        max_points: Maximum points for subsampling
        is_graph: Whether input is a graph
        tau: Heat diffusion time (for graphs)
        heat_method: Heat method for graphs ('content' or 'diffusion')
        max_dim: Maximum dimension of simplices (default: 2 for H₁)
                 For H₁, 2-skeleton is sufficient. Higher dimensions increase cost.
    
    Returns:
        Array of (birth, death) pairs, shape (n, 2)
    """
    if is_graph or (hasattr(X, 'edge_index') or (isinstance(X, dict) and 'edge_index' in X)):
        # Graph data: use lower-star filtration
        diagram = lower_star_filtration_graph(X, tau=tau, heat_method=heat_method, max_dim=max_dim)
        return diagram.points
    
    # Point cloud data
    if method == 'alpha' or method == 'auto':
        diagram = compute_persistence_diagram_alpha(X, max_points=max_points)
        return diagram.points
    else:
        # Use ripser
        if max_points is not None and len(X) > max_points:
            rng = np.random.RandomState(14)
            indices = rng.choice(len(X), max_points, replace=False)
            X = X[indices]
        
        result = ripser.ripser(X, maxdim=1)
        diagrams = result['dgms']
        
        if len(diagrams) > 1 and len(diagrams[1]) > 0:
            h1_points = diagrams[1]
            h1_points = h1_points[np.isfinite(h1_points[:, 1]), :]
            return h1_points
        else:
            return np.empty((0, 2))
