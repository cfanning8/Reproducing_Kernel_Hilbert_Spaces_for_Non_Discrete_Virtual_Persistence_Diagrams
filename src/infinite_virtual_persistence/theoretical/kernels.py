"""
Gaussian/RKHS kernels for virtual persistence diagrams.
"""

import numpy as np
from typing import Callable, Optional, Tuple
from scipy.linalg import sqrtm
from scipy.sparse.linalg import LinearOperator


class HilbertEmbedding:
    """
    Hilbert embedding phi: (X/A, d1, [A]) -> H with phi([A]) = 0.
    Extends to bounded linear operator J: B -> H via Lipschitz-free property.
    """
    
    def __init__(self, phi: Callable[[np.ndarray], np.ndarray], 
                 L: float, basepoint: Optional[np.ndarray] = None):
        """
        Initialize Hilbert embedding.
        
        Args:
            phi: Function mapping points in X/A to H, with phi([A]) = 0
            L: Lipschitz constant Lip_{d1}(phi)
            basepoint: Basepoint [A] in X/A
        """
        self.phi = phi
        self.L = L
        self.basepoint = basepoint
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate phi(x)."""
        if self.basepoint is not None:
            # Ensure basepoint maps to zero
            if np.allclose(x, self.basepoint):
                return np.zeros_like(self.phi(x))
        return self.phi(x)
    
    def extend_to_B(self, delta_points: np.ndarray, 
                    coefficients: np.ndarray) -> np.ndarray:
        """
        Extend to linear operator J on B.
        
        For x = sum_i n_i delta_{u_i} in K(X,A) subset B,
        J(x) = sum_i n_i phi(u_i).
        
        Args:
            delta_points: Points u_i in X/A, shape (n, dim)
            coefficients: Coefficients n_i, shape (n,)
        
        Returns:
            J(x) in H
        """
        result = np.zeros_like(self.phi(delta_points[0]))
        for i, (point, coeff) in enumerate(zip(delta_points, coefficients)):
            result += coeff * self(point)
        return result


class GaussianRKHSKernel:
    """
    Translation-invariant positive definite kernel on B (and K(X,A))
    via Gaussian measure on dual space B*.
    
    k_t(x,y) = exp(-t/2 ||Sigma^{1/2} J(x-y)||^2)
    """
    
    def __init__(self, J: HilbertEmbedding, Sigma: np.ndarray, t: float = 1.0, 
                 embedding_dim: Optional[int] = None):
        """
        Initialize Gaussian RKHS kernel.
        
        Args:
            J: Hilbert embedding extended to B
            Sigma: Covariance operator on ran(J), shape (dim_H, dim_H)
            t: Scale parameter
            embedding_dim: Expected embedding dimension. If None, inferred from Sigma.
                          If embeddings are concatenated across classes, this should be
                          n_classes * dim_H. The kernel will create a block-diagonal Sigma.
        """
        self.J = J
        self.Sigma_base = np.asarray(Sigma)  # Base Sigma (dim_H x dim_H)
        self.t = t
        self.embedding_dim = embedding_dim
        
        # If embedding_dim is provided and different from Sigma size, create block-diagonal
        dim_H = self.Sigma_base.shape[0]
        if embedding_dim is not None and embedding_dim != dim_H:
            if embedding_dim % dim_H != 0:
                raise ValueError(f"embedding_dim ({embedding_dim}) must be a multiple of dim_H ({dim_H})")
            n_blocks = embedding_dim // dim_H
            # Create block-diagonal Sigma: [Sigma, 0, ...; 0, Sigma, ...; ...]
            self.Sigma = np.kron(np.eye(n_blocks), self.Sigma_base)
        else:
            self.Sigma = self.Sigma_base
        
        # Precompute Sigma^{1/2}
        self.Sigma_sqrt = sqrtm(self.Sigma)
        self.Q = None  # Covariance operator Q = J* Sigma J: B -> B*
    
    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate kernel k_t(x, y).
        
        Args:
            x, y: Elements of B represented as (delta_points, coefficients)
        
        Returns:
            k_t(x, y)
        """
        # Compute J(x - y)
        x_points, x_coeffs = x if isinstance(x, tuple) else (x, np.ones(len(x)))
        y_points, y_coeffs = y if isinstance(y, tuple) else (y, np.ones(len(y)))
        
        # For simplicity, assume x and y are already in H via J
        # In full implementation, need to compute J(x) and J(y)
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            # Assume x, y are already in H
            diff = x - y
        else:
            Jx = self.J.extend_to_B(x_points, x_coeffs)
            Jy = self.J.extend_to_B(y_points, y_coeffs)
            diff = Jx - Jy
        
        # Auto-detect dimension and adjust Sigma if needed
        actual_dim = len(diff)
        sigma_dim = self.Sigma.shape[0]
        
        if actual_dim != sigma_dim:
            # Embeddings are concatenated - use block-diagonal Sigma
            if actual_dim % sigma_dim != 0:
                raise ValueError(f"Embedding dimension {actual_dim} is not a multiple of Sigma dimension {sigma_dim}")
            n_blocks = actual_dim // sigma_dim
            Sigma_actual = np.kron(np.eye(n_blocks), self.Sigma_base)
            Sigma_sqrt_actual = sqrtm(Sigma_actual)
        else:
            Sigma_sqrt_actual = self.Sigma_sqrt
        
        # Compute ||Sigma^{1/2} (Jx - Jy)||^2
        Sigma_sqrt_diff = Sigma_sqrt_actual @ diff
        norm_sq = np.dot(Sigma_sqrt_diff, Sigma_sqrt_diff)
        
        return np.exp(-self.t / 2 * norm_sq)
    
    def gram_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Gram matrix K_ij = k_t(x_i, x_j).
        
        Args:
            X: Array of points, shape (n, dim) (already embedded via J)
               If dim != Sigma.shape[0], automatically adjusts Sigma to block-diagonal
        
        Returns:
            Gram matrix, shape (n, n)
        """
        n = len(X)
        K = np.zeros((n, n))
        
        # Auto-detect embedding dimension and adjust Sigma if needed
        actual_dim = X.shape[1] if len(X.shape) > 1 else len(X[0])
        sigma_dim = self.Sigma.shape[0]
        
        if actual_dim != sigma_dim:
            # Embeddings are concatenated - use block-diagonal Sigma
            if actual_dim % sigma_dim != 0:
                raise ValueError(f"Embedding dimension {actual_dim} is not a multiple of Sigma dimension {sigma_dim}")
            n_blocks = actual_dim // sigma_dim
            Sigma_actual = np.kron(np.eye(n_blocks), self.Sigma_base)
            Sigma_sqrt_actual = sqrtm(Sigma_actual)
        else:
            Sigma_sqrt_actual = self.Sigma_sqrt
        
        for i in range(n):
            for j in range(n):
                diff = X[i] - X[j]
                Sigma_sqrt_diff = Sigma_sqrt_actual @ diff
                norm_sq = np.dot(Sigma_sqrt_diff, Sigma_sqrt_diff)
                K[i, j] = np.exp(-self.t / 2 * norm_sq)
        
        return K
    
    def lipschitz_bound(self, f_norm: float) -> float:
        """
        Compute Lipschitz bound: Lip_rho(f|_K) <= sqrt(t) ||Sigma^{1/2} J|| ||f||_H
        
        Args:
            f_norm: ||f||_{H_{J,Sigma,t}}
        
        Returns:
            Lipschitz bound
        """
        # ||Sigma^{1/2} J|| = ||Sigma^{1/2}|| ||J|| = ||Sigma^{1/2}|| L
        Sigma_sqrt_norm = np.linalg.norm(self.Sigma_sqrt, ord=2)
        J_norm = self.J.L
        
        return np.sqrt(self.t) * Sigma_sqrt_norm * J_norm * f_norm


class RandomFourierFeatures:
    """
    Random Fourier feature approximation of Gaussian RKHS kernel.
    """
    
    def __init__(self, kernel: GaussianRKHSKernel, R: int, seed: Optional[int] = 14):
        """
        Initialize random Fourier features.
        
        Args:
            kernel: Gaussian RKHS kernel
            R: Number of random features
            seed: Random seed (default: 14)
        """
        self.kernel = kernel
        self.R = R
        self.rng = np.random.RandomState(seed)
        
        # Sample u_r ~ N(0, Sigma) in H
        dim_H = self.kernel.Sigma.shape[0]
        z_r = self.rng.randn(R, dim_H)  # z_r ~ N(0, I)
        self.u_r = (self.kernel.Sigma_sqrt @ z_r.T).T  # u_r = Sigma^{1/2} z_r
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Compute random feature map Phi_R(x).
        
        Args:
            x: Point in H (already embedded via J)
        
        Returns:
            Phi_R(x) = (1/sqrt(R)) (e^{i sqrt(t) <Jx, u_1>}, ..., e^{i sqrt(t) <Jx, u_R>})
        """
        features = np.zeros(self.R, dtype=complex)
        
        for r in range(self.R):
            inner_prod = np.dot(x, self.u_r[r])
            features[r] = np.exp(1j * np.sqrt(self.kernel.t) * inner_prod)
        
        return features / np.sqrt(self.R)
    
    def real_features(self, x: np.ndarray) -> np.ndarray:
        """
        Real-valued feature map using cos/sin.
        
        Returns:
            [cos(sqrt(t) <Jx, u_r>), sin(sqrt(t) <Jx, u_r>)] for r=1,...,R
        """
        features = []
        
        for r in range(self.R):
            inner_prod = np.dot(x, self.u_r[r])
            scaled = np.sqrt(self.kernel.t) * inner_prod
            features.extend([np.cos(scaled), np.sin(scaled)])
        
        return np.array(features) / np.sqrt(self.R)
    
    def kernel_approx(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Approximate kernel k_t(x,y) via random features.
        
        k_hat_R(x,y) = <Phi_R(x), Phi_R(y)>
        """
        phi_x = self.real_features(x)
        phi_y = self.real_features(y)
        return np.dot(phi_x, phi_y)
    
    def concentration_bound(self, epsilon: float) -> float:
        """
        Hoeffding concentration bound.
        
        P(|k_hat_R(x,y) - k_t(x,y)| > epsilon) <= 4 exp(-R epsilon^2 / 4)
        """
        return 4 * np.exp(-self.R * epsilon**2 / 4)
    
    def finite_sample_bound(self, N: int, epsilon: float, delta: float) -> int:
        """
        Compute required R for finite sample guarantee.
        
        If R >= (4/epsilon^2) log(4N^2/delta), then with probability >= 1-delta,
        max_{i,j} |k_hat_R(x_i, x_j) - k_t(x_i, x_j)| <= epsilon
        """
        return int(np.ceil(4 / epsilon**2 * np.log(4 * N**2 / delta)))

