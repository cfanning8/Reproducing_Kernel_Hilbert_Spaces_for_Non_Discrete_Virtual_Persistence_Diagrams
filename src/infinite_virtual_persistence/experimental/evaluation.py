"""
Evaluation metrics: Lipschitz bounds, kernel approximation, robustness tests.
"""

import numpy as np
from typing import Tuple, Optional, Callable
from scipy.spatial.distance import pdist, squareform


def estimate_lipschitz_constant(f: Callable, X: np.ndarray, 
                                 rho: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Estimate Lipschitz constant from data.
    
    hat{Lip}(f) = max_{(i,j) in P} |f(x_i) - f(x_j)| / rho(x_i, x_j)
    
    Args:
        f: Function to evaluate
        X: Points, shape (n, dim)
        rho: Distance matrix, shape (n, n)
    
    Returns:
        (estimated_lipschitz, pair_ratios)
    """
    n = len(X)
    f_values = np.array([f(x) for x in X])
    
    ratios = []
    for i in range(n):
        for j in range(i + 1, n):
            if rho[i, j] > 0:
                ratio = np.abs(f_values[i] - f_values[j]) / rho[i, j]
                ratios.append(ratio)
    
    ratios = np.array(ratios)
    return np.max(ratios), ratios


def kernel_approximation_error(kernel_true: Callable, 
                               kernel_approx: Callable,
                               X: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute kernel approximation error.
    
    max_{i,j} |k_hat_R(x_i, x_j) - k_t(x_i, x_j)|
    
    Args:
        kernel_true: True kernel function
        kernel_approx: Approximate kernel function
        X: Points, shape (n, dim)
    
    Returns:
        (max_error, error_matrix)
    """
    n = len(X)
    error_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            k_true = kernel_true(X[i], X[j])
            k_approx = kernel_approx(X[i], X[j])
            error_matrix[i, j] = np.abs(k_true - k_approx)
    
    return np.max(error_matrix), error_matrix


def robustness_test(f: Callable, X_original: np.ndarray, 
                   X_perturbed: np.ndarray,
                   rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Test robustness by correlating output drift with metric distance.
    
    Args:
        f: Function to test
        X_original: Original data points
        X_perturbed: Perturbed data points
        rho: Metric distance matrix
    
    Returns:
        (output_drifts, correlations)
    """
    n = len(X_original)
    f_original = np.array([f(x) for x in X_original])
    f_perturbed = np.array([f(x) for x in X_perturbed])
    
    output_drifts = np.abs(f_original - f_perturbed)
    
    # Compute correlation with rho distances
    # For each point, compute drift vs distance to original
    distances = np.array([rho[i, i] if i < len(X_perturbed) else 0 
                         for i in range(n)])
    
    # Filter out zero distances
    mask = distances > 0
    if np.sum(mask) > 0:
        correlation = np.corrcoef(output_drifts[mask], distances[mask])[0, 1]
    else:
        correlation = 0.0
    
    return output_drifts, correlation


def apply_point_jitter(X: np.ndarray, noise_level: float, 
                      seed: Optional[int] = None) -> np.ndarray:
    """Apply Gaussian jitter to point cloud."""
    rng = np.random.RandomState(seed)
    return X + rng.randn(*X.shape) * noise_level


def apply_sparsification(X: np.ndarray, sparsity: float, 
                        seed: Optional[int] = None) -> np.ndarray:
    """Randomly remove points."""
    rng = np.random.RandomState(seed)
    n = len(X)
    keep = rng.choice(n, size=int(n * (1 - sparsity)), replace=False)
    return X[keep]


def apply_edge_rewiring(adjacency: np.ndarray, noise_level: float,
                       seed: Optional[int] = None) -> np.ndarray:
    """Apply noise to edge weights in graph."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(*adjacency.shape) * noise_level
    return np.maximum(0, adjacency + noise)  # Keep non-negative


def apply_attribute_noise(X: np.ndarray, noise_level: float,
                         seed: Optional[int] = None) -> np.ndarray:
    """Apply noise to node/point attributes."""
    rng = np.random.RandomState(seed)
    return X + rng.randn(*X.shape) * noise_level

