"""
Virtual persistence diagrams and Grothendieck metric construction.
"""

import numpy as np
from typing import Dict, Tuple, Set, Optional
from collections import defaultdict
from scipy.spatial.distance import cdist


class MetricPair:
    """Represents a metric pair (X, d, A) with 1-strengthened metric."""
    
    def __init__(self, X: np.ndarray, d: Optional[np.ndarray] = None, A: Optional[np.ndarray] = None, 
                 max_points: int = 1000, lazy: bool = True):
        """Initialize metric pair."""
        self.X = np.asarray(X)
        self.n = len(self.X)
        
        if self.n > max_points:
            import warnings
            warnings.warn(f"MetricPair: Sampling {max_points} points from {self.n} to prevent memory issues")
            rng = np.random.RandomState(14)
            sample_idx = rng.choice(self.n, size=max_points, replace=False)
            self.X = self.X[sample_idx]
            self.n = len(self.X)
        
        self._d = d
        self._d_computed = (d is not None)
        
        if A is None:
            self.A = set()
        else:
            self.A = set(A) if isinstance(A, (list, np.ndarray)) else A
        
        self._d1 = None
        self._X_A = None
        self.lazy = lazy
        
        if not lazy and d is None:
            self._compute_d()
    
    def _compute_d(self):
        """Compute distance matrix if not already computed."""
        if not self._d_computed:
            if self.n > 10000:
                import warnings
                warnings.warn(f"Computing {self.n}x{self.n} distance matrix - this may be slow and memory-intensive")
            self._d = cdist(self.X, self.X, metric='euclidean')
            self._d_computed = True
    
    @property
    def d(self):
        """Lazy access to distance matrix."""
        if not self._d_computed:
            self._compute_d()
        return self._d
    
    def d_to_A(self, i: int) -> float:
        """Distance from point i to set A."""
        if not self.A:
            if self.X.shape[1] == 2:
                return abs(self.X[i, 1] - self.X[i, 0]) / 2.0
            else:
                return np.linalg.norm(self.X[i])
        if not self._d_computed:
            self._compute_d()
        return min(self.d[i, a] for a in self.A)
    
    def d1(self, i: int, j: int) -> float:
        """1-strengthened metric d1(x,y) = min(d(x,y), d(x,A) + d(y,A))."""
        if self._d1 is None:
            self._compute_d1()
        return self._d1[i, j]
    
    def _compute_d1(self):
        """Precompute d1 matrix (lazy computation)."""
        if self._d1 is not None:
            return
        
        n = self.n
        
        if n > 10000:
            import warnings
            warnings.warn(f"Computing {n}x{n} d1 matrix - this may be slow and memory-intensive")
        
        if not self._d_computed:
            self._compute_d()
        
        self._d1 = np.zeros((n, n))
        d_to_A = np.array([self.d_to_A(i) for i in range(n)])
        
        for i in range(n):
            for j in range(n):
                self._d1[i, j] = min(
                    self.d[i, j],
                    d_to_A[i] + d_to_A[j]
                )
    
    def quotient_space(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return quotient space X/A with collapsed basepoint.
        Returns (points, d1_quotient).
        """
        if self._X_A is not None:
            return self._X_A
        
        if not self.A:
            self._X_A = (self.X, self._d1 if self._d1 is not None else self._compute_d1())
            return self._X_A
        
        A_list = sorted(self.A)
        non_A = [i for i in range(self.n) if i not in self.A]
        
        X_quotient = np.vstack([self.X[non_A], np.zeros((1, self.X.shape[1]))])
        n_quotient = len(X_quotient)
        d1_quotient = np.zeros((n_quotient, n_quotient))
        
        idx_map = {i: j for j, i in enumerate(non_A)}
        basepoint_idx = n_quotient - 1
        
        for i_orig in range(self.n):
            for j_orig in range(self.n):
                if i_orig in self.A:
                    i_q = basepoint_idx
                else:
                    i_q = idx_map[i_orig]
                
                if j_orig in self.A:
                    j_q = basepoint_idx
                else:
                    j_q = idx_map[j_orig]
                
                d1_quotient[i_q, j_q] = self.d1(i_orig, j_orig)
        
        self._X_A = (X_quotient, d1_quotient)
        return self._X_A


class PersistenceDiagram:
    """Represents a finite persistence diagram as a multiset."""
    
    def __init__(self, points: np.ndarray):
        """Initialize persistence diagram from (birth, death) pairs."""
        self.points = np.asarray(points)
        if len(self.points) > 0:
            assert self.points.shape[1] == 2
            assert np.all(self.points[:, 1] >= self.points[:, 0])
    
    def __len__(self):
        return len(self.points)
    
    def __repr__(self):
        return f"PersistenceDiagram({len(self.points)} points)"
    
    def __getstate__(self):
        """Support pickle serialization."""
        return (self.points,)
    
    def __setstate__(self, state):
        """Support pickle deserialization (backwards compatible with any tuple length)."""
        if isinstance(state, tuple) and len(state) > 0:
            try:
                self.points = np.asarray(state[0])
                if len(self.points) > 0 and self.points.shape[1] == 2:
                    return
            except:
                pass
            
            for item in state:
                try:
                    arr = np.asarray(item)
                    if len(arr.shape) == 2 and arr.shape[1] == 2:
                        self.points = arr
                        return
                except:
                    continue
            
            self.points = np.empty((0, 2))
        elif isinstance(state, dict):
            self.points = np.asarray(state.get('points', np.empty((0, 2))))
        elif isinstance(state, np.ndarray):
            self.points = np.asarray(state)
        else:
            self.points = np.empty((0, 2))


class VirtualDiagram:
    """Represents an element of K(X,A) as a formal difference alpha - beta."""
    
    def __init__(self, alpha: Dict[Tuple[float, float], int], 
                 beta: Optional[Dict[Tuple[float, float], int]] = None):
        """Initialize virtual diagram from alpha and beta multisets."""
        self.alpha = defaultdict(int, {k: max(0, v) for k, v in alpha.items()})
        self.beta = defaultdict(int, {k: max(0, v) for k, v in (beta or {}).items()})
    
    def __neg__(self):
        """Negation: swap alpha and beta."""
        return VirtualDiagram(self.beta, self.alpha)
    
    def __add__(self, other):
        """Addition in K(X,A)."""
        if not isinstance(other, VirtualDiagram):
            raise TypeError("Can only add VirtualDiagram to VirtualDiagram")
        
        alpha_new = defaultdict(int, self.alpha)
        beta_new = defaultdict(int, self.beta)
        
        for k, v in other.alpha.items():
            alpha_new[k] += v
        for k, v in other.beta.items():
            beta_new[k] += v
        
        return VirtualDiagram(alpha_new, beta_new)
    
    def __sub__(self, other):
        """Subtraction: self + (-other)."""
        return self + (-other)
    
    def to_diagram(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to pair of diagrams (alpha, beta) as arrays."""
        alpha_points = []
        beta_points = []
        
        for (b, d), mult in self.alpha.items():
            alpha_points.extend([(b, d)] * mult)
        
        for (b, d), mult in self.beta.items():
            beta_points.extend([(b, d)] * mult)
        
        return (np.array(alpha_points) if alpha_points else np.empty((0, 2)),
                np.array(beta_points) if beta_points else np.empty((0, 2)))


def wasserstein_1(alpha: np.ndarray, beta: np.ndarray, d1: np.ndarray) -> float:
    """Compute 1-Wasserstein distance between persistence diagrams."""
    from scipy.optimize import linear_sum_assignment
    
    n, m = len(alpha), len(beta)
    
    if n == 0 and m == 0:
        return 0.0
    if n == 0:
        return np.sum([d1_to_basepoint(p, d1) for p in beta])
    if m == 0:
        return np.sum([d1_to_basepoint(p, d1) for p in alpha])
    
    cost_matrix = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            cost_matrix[i, j] = np.abs(alpha[i, 0] - beta[j, 0]) + np.abs(alpha[i, 1] - beta[j, 1])
    
    basepoint_costs_alpha = np.array([d1_to_basepoint(alpha[i], d1) for i in range(n)])
    basepoint_costs_beta = np.array([d1_to_basepoint(beta[j], d1) for j in range(m)])
    
    extended_cost = np.zeros((n + m, n + m))
    extended_cost[:n, :m] = cost_matrix
    extended_cost[:n, m:] = np.diag(basepoint_costs_alpha)
    extended_cost[n:, :m] = np.diag(basepoint_costs_beta)
    extended_cost[n:, m:] = 0
    
    row_ind, col_ind = linear_sum_assignment(extended_cost)
    return extended_cost[row_ind, col_ind].sum()


def d1_to_basepoint(point: np.ndarray, d1: np.ndarray) -> float:
    """Distance from persistence point to basepoint (diagonal)."""
    return np.linalg.norm(point)


def grothendieck_metric(g1: VirtualDiagram, g2: VirtualDiagram, 
                        d1: np.ndarray) -> float:
    """Compute Grothendieck metric rho(g1, g2) = W1(alpha1 + beta2, alpha2 + beta1)."""
    alpha1, beta1 = g1.to_diagram()
    alpha2, beta2 = g2.to_diagram()
    
    combined1 = np.vstack([alpha1, beta2]) if len(beta2) > 0 else alpha1
    combined2 = np.vstack([alpha2, beta1]) if len(beta1) > 0 else alpha2
    
    return wasserstein_1(combined1, combined2, d1)

