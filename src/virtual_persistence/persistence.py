"""
Persistence diagram computation from graphs and point clouds.
"""

import numpy as np
from typing import Optional, List, Tuple
from scipy.sparse import csr_matrix
import ripser
from gudhi import AlphaComplex

from .heat_flow import (
    heat_vertex_function, 
    heat_edge_weights,
    lower_star_filtration_value
)
from .virtual_diagrams import PersistenceDiagram


def graph_to_adjacency(graph_data) -> np.ndarray:
    """Convert graph data to adjacency matrix."""
    if hasattr(graph_data, 'edge_index'):
        # PyTorch Geometric
        edge_index = graph_data.edge_index.cpu().numpy()
        num_nodes = graph_data.num_nodes if hasattr(graph_data, 'num_nodes') else edge_index.max() + 1
    elif isinstance(graph_data, dict) and 'edge_index' in graph_data:
        edge_index = graph_data['edge_index']
        num_nodes = graph_data.get('num_nodes', edge_index.max() + 1)
    else:
        raise ValueError("Graph data must have 'edge_index' attribute")
    
    adj = np.zeros((num_nodes, num_nodes))
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        adj[u, v] = 1
        adj[v, u] = 1  # Undirected
    
    return adj


def compute_clique_complex(adjacency: np.ndarray, max_dim: int = 2) -> List[List[Tuple]]:
    """Compute clique complex up to max_dim."""
    n = adjacency.shape[0]
    simplices = [[] for _ in range(max_dim + 1)]
    
    simplices[0] = [(i,) for i in range(n)]
    simplices[1] = []
    for u in range(n):
        for v in range(u + 1, n):
            if adjacency[u, v] > 0:
                simplices[1].append((u, v))
    
    if max_dim >= 2:
        simplices[2] = []
        for u in range(n):
            for v in range(u + 1, n):
                if adjacency[u, v] > 0:
                    for w in range(v + 1, n):
                        if adjacency[u, w] > 0 and adjacency[v, w] > 0:
                            simplices[2].append((u, v, w))
    
    return simplices


def lower_star_filtration_graph(graph_data, 
                                tau: float = 1.0,
                                heat_method: str = 'content',
                                max_dim: int = 2) -> PersistenceDiagram:
    """Compute persistence diagram from graph using lower-star filtration."""
    adjacency = graph_to_adjacency(graph_data)
    vertex_function = heat_vertex_function(adjacency, tau=tau, method=heat_method, normalize='rank')
    simplices = compute_clique_complex(adjacency, max_dim=max_dim)
    filtration_values = {}
    
    for v in simplices[0]:
        filtration_values[v] = vertex_function[v[0]]
    
    for edge in simplices[1]:
        filtration_values[edge] = max(vertex_function[edge[0]], vertex_function[edge[1]])
    
    for triangle in simplices[2]:
        filtration_values[triangle] = max(vertex_function[v] for v in triangle)
    n = adjacency.shape[0]
    dist_matrix = np.full((n, n), np.inf)
    np.fill_diagonal(dist_matrix, vertex_function)
    
    for edge in simplices[1]:
        u, v = edge
        dist_matrix[u, v] = filtration_values[edge]
        dist_matrix[v, u] = filtration_values[edge]
    
    max_finite = dist_matrix[dist_matrix < np.inf].max() if np.any(dist_matrix < np.inf) else 1.0
    dist_matrix[dist_matrix == np.inf] = max_finite * 2
    
    result = ripser.ripser(dist_matrix, maxdim=1, metric='precomputed')
    diagrams = result['dgms']
    if len(diagrams) > 1 and len(diagrams[1]) > 0:
        h1_points = diagrams[1]
        h1_points = h1_points[np.isfinite(h1_points[:, 1]), :]
        return PersistenceDiagram(h1_points)
    else:
        return PersistenceDiagram(np.empty((0, 2)))


def compute_persistence_diagram_alpha(X: np.ndarray, 
                                     max_points: Optional[int] = None) -> PersistenceDiagram:
    """Compute persistence diagram using alpha complex."""
    if max_points is not None and len(X) > max_points:
        rng = np.random.RandomState(14)
        indices = rng.choice(len(X), max_points, replace=False)
        X = X[indices]
    
    alpha_complex = AlphaComplex(points=X)
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=np.inf)
    persistence = simplex_tree.persistence()
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
    """Compute persistence diagram from input data."""
    if is_graph or (hasattr(X, 'edge_index') or (isinstance(X, dict) and 'edge_index' in X)):
        diagram = lower_star_filtration_graph(X, tau=tau, heat_method=heat_method, max_dim=max_dim)
        return diagram.points
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
