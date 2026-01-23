"""
Heat flow on graphs for filtration and weighting.
"""

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import expm_multiply
from typing import Optional, Tuple


def graph_laplacian(adjacency: np.ndarray, normalized: bool = True) -> csr_matrix:
    """
    Compute graph Laplacian.
    
    Args:
        adjacency: Adjacency matrix (sparse or dense), shape (n, n)
        normalized: If True, use normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
                    If False, use combinatorial Laplacian L = D - A
    
    Returns:
        Laplacian matrix (sparse)
    """
    if not isinstance(adjacency, csr_matrix):
        adjacency = csr_matrix(adjacency)
    
    n = adjacency.shape[0]
    degrees = np.array(adjacency.sum(axis=1)).flatten()
    
    if normalized:
        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        deg_sqrt_inv = np.zeros(n)
        deg_sqrt_inv[degrees > 0] = 1.0 / np.sqrt(degrees[degrees > 0])
        D_sqrt_inv = csr_matrix((deg_sqrt_inv, (np.arange(n), np.arange(n))))
        L = eye(n) - D_sqrt_inv @ adjacency @ D_sqrt_inv
    else:
        # Combinatorial Laplacian: L = D - A
        D = csr_matrix((degrees, (np.arange(n), np.arange(n))))
        L = D - adjacency
    
    return L


def heat_kernel(L: csr_matrix, tau: float) -> csr_matrix:
    """
    Compute heat kernel H(tau) = exp(-tau * L).
    
    Args:
        L: Graph Laplacian (sparse)
        tau: Time parameter (heat diffusion time)
    
    Returns:
        Heat kernel matrix (sparse)
    """
    # Use sparse matrix exponential
    # For large matrices, expm_multiply is more efficient
    n = L.shape[0]
    I = eye(n)
    
    # For small matrices, can use dense expm
    if n < 1000:
        L_dense = L.toarray()
        H_dense = np.linalg.matrix_power(I.toarray() - tau * L_dense / 10, 10)
        # More accurate: use scipy.linalg.expm
        from scipy.linalg import expm
        H_dense = expm(-tau * L_dense)
        return csr_matrix(H_dense)
    else:
        # For large matrices, use iterative method
        # Approximate: H(tau) ≈ (I - tau*L/k)^k for large k
        k = 20
        H = I - (tau / k) * L
        for _ in range(k - 1):
            H = H @ (I - (tau / k) * L)
        return H


def heat_edge_weights(adjacency: np.ndarray, 
                      tau: float = 1.0,
                      normalized: bool = True) -> np.ndarray:
    """
    Compute heat-based edge weights.
    
    Weights are computed as w_tau(u,v) = H(tau)_{uv} for (u,v) in E only.
    Non-edges remain zero (no new edges created).
    
    Args:
        adjacency: Adjacency matrix, shape (n, n)
        tau: Heat diffusion time
        normalized: Whether to use normalized Laplacian
    
    Returns:
        Weighted adjacency matrix (same sparsity pattern as input)
    """
    L = graph_laplacian(adjacency, normalized=normalized)
    H = heat_kernel(L, tau)
    
    # Extract weights only for existing edges
    # H is dense, but we only keep weights where adjacency > 0
    H_dense = H.toarray() if isinstance(H, csr_matrix) else H
    adj_dense = adjacency.toarray() if isinstance(adjacency, csr_matrix) else adjacency
    
    # Set weights to zero for non-edges
    weights = H_dense * (adj_dense > 0)
    
    return weights


def heat_vertex_function(adjacency: np.ndarray,
                         tau: float = 1.0,
                         source: Optional[np.ndarray] = None,
                         method: str = 'content',
                         normalize: str = 'rank') -> np.ndarray:
    """
    Compute heat-derived vertex function for lower-star filtration.
    
    Options:
    - 'content': f_tau(v) = H(tau)_{vv} (heat content at node)
    - 'diffusion': f_tau(v) = u(tau)_v where u(tau) = H(tau) * s (diffusion from source)
    
    Args:
        adjacency: Adjacency matrix, shape (n, n)
        tau: Heat diffusion time
        source: Source distribution for diffusion method (if None, uses uniform)
        method: 'content' or 'diffusion'
        normalize: Normalization method: 'rank' (default, recommended), 'minmax', 'zscore', or None
    
    Returns:
        Vertex function values, shape (n,), normalized per graph
    """
    L = graph_laplacian(adjacency, normalized=True)
    H = heat_kernel(L, tau)
    
    if method == 'content':
        # Heat content: diagonal of heat kernel
        H_dense = H.toarray() if isinstance(H, csr_matrix) else H
        f = np.diag(H_dense)
    elif method == 'diffusion':
        # Diffusion from source
        n = adjacency.shape[0]
        if source is None:
            # Uniform source
            source = np.ones(n) / n
        else:
            source = np.asarray(source)
            source = source / source.sum()
        
        # u(tau) = H(tau) * s
        if isinstance(H, csr_matrix):
            u = H @ source
        else:
            u = H @ source
        f = u
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize per graph to ensure comparability across graphs
    if normalize == 'rank':
        # Rank normalization: f(v) <- rank(f(v)) / |V|
        # Very robust, scale-free, recommended
        ranks = np.argsort(np.argsort(f))  # Get ranks (0-indexed)
        f = ranks / len(f)
    elif normalize == 'minmax':
        # Min-max normalization to [0, 1]
        f_min, f_max = f.min(), f.max()
        if f_max > f_min:
            f = (f - f_min) / (f_max - f_min)
        else:
            f = np.zeros_like(f)  # All values equal
    elif normalize == 'zscore':
        # Z-score normalization
        f_mean, f_std = f.mean(), f.std()
        if f_std > 0:
            f = (f - f_mean) / f_std
        else:
            f = np.zeros_like(f)  # All values equal
    elif normalize is None or normalize == 'none':
        # No normalization (not recommended for cross-graph comparability)
        pass
    else:
        raise ValueError(f"Unknown normalization method: {normalize}")
    
    return f


def lower_star_filtration_value(clique_vertices: np.ndarray, 
                               vertex_function: np.ndarray) -> float:
    """
    Compute lower-star filtration value for a clique.
    
    Lower-star filtration: f(sigma) = max_{v in sigma} f(v)
    
    Args:
        clique_vertices: Indices of vertices in clique
        vertex_function: Vertex function values
    
    Returns:
        Filtration value
    """
    return np.max(vertex_function[clique_vertices])
