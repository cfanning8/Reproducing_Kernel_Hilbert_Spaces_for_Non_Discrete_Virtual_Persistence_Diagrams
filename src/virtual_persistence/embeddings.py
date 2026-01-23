"""Hilbert embeddings: learned, graph Laplacian-based, spectral."""

import numpy as np
from typing import Callable, Optional, Tuple
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigs
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
import warnings


class GraphLaplacianEmbedding:
    """Hilbert embedding based on graph Laplacian eigenvectors."""
    
    def __init__(self, dim_H: int = 128, normalized: bool = True, 
                 k_neighbors: int = 10):
        """Initialize graph Laplacian embedding."""
        self.dim_H = dim_H
        self.normalized = normalized
        self.k_neighbors = k_neighbors
        self.eigenvectors = None
        self.eigenvalues = None
        self.L = None  # Lipschitz constant
    
    def fit(self, X: np.ndarray, d1: np.ndarray):
        """Fit embedding to data."""
        n = len(X)
        
        from sklearn.neighbors import kneighbors_graph
        knn_graph = kneighbors_graph(d1, n_neighbors=self.k_neighbors, 
                                     mode='distance', metric='precomputed')
        
        if self.normalized:
            L = csgraph.laplacian(knn_graph, normed=True)
        else:
            L = csgraph.laplacian(knn_graph, normed=False)
        
        try:
            eigenvals, eigenvecs = eigs(L, k=min(self.dim_H + 1, n - 1), 
                                        which='SM', return_eigenvectors=True)
            eigenvals = np.real(eigenvals)
            eigenvecs = np.real(eigenvecs)
            
            idx = np.argsort(eigenvals)
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            self.eigenvalues = eigenvals[1:]
            self.eigenvectors = eigenvecs[:, 1:]
            
            self.L = np.max(self.eigenvalues) if len(self.eigenvalues) > 0 else 1.0
        except Exception as e:
            warnings.warn(f"Error computing Laplacian eigenvectors: {e}")
            raise RuntimeError(f"Error computing Laplacian eigenvectors: {e}") from e
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Embed point x."""
        if self.eigenvectors is None:
            raise ValueError("Embedding not fitted")
        
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        if x.shape[1] != self.eigenvectors.shape[0]:
            raise ValueError(
                f"Input dimension {x.shape[1]} does not match "
                f"embedding dimension {self.eigenvectors.shape[0]}. "
                f"Must fit embedding on data with same dimension."
            )
        
        return (self.eigenvectors.T @ x.T).T.flatten()[:self.dim_H]


class LearnedEmbedding:
    """Learned Hilbert embedding using neural network."""
    
    def __init__(self, dim_H: int = 128, hidden_dims: list = [256, 128],
                 learning_rate: float = 0.001):
        """Initialize learned embedding."""
        self.dim_H = dim_H
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.model = None
        self.L = 1.0  # Will be estimated during training
    
    def fit(self, X: np.ndarray, d1: np.ndarray, max_epochs: int = 100):
        """Train embedding to preserve d1 distances."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            
            n, input_dim = X.shape
            
            layers = []
            dims = [input_dim] + self.hidden_dims + [self.dim_H]
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.ReLU())
            
            self.model = nn.Sequential(*layers)
            
            X_tensor = torch.FloatTensor(X)
            d1_tensor = torch.FloatTensor(d1)
            
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            for epoch in range(max_epochs):
                optimizer.zero_grad()
                X_embedded = self.model(X_tensor)
                dists_embedded = torch.cdist(X_embedded, X_embedded)
                loss = torch.mean((dists_embedded - d1_tensor) ** 2)
                loss.backward()
                optimizer.step()

            X_tensor.requires_grad = True
            X_embedded = self.model(X_tensor)
            grad_norm = torch.autograd.grad(X_embedded.sum(), X_tensor, 
                                           create_graph=True)[0]
            self.L = torch.norm(grad_norm, dim=1).max().item()
            
        except ImportError:
            raise ImportError(
                "PyTorch not available. Install with: pip install torch"
            )
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Embed point x."""
        if self.model is None:
            if hasattr(self, 'eigenvectors'):
                # Use PCA
                if len(x.shape) == 1:
                    x = x.reshape(1, -1)
                return (self.eigenvectors.T @ x.T).T.flatten()
            else:
                return np.zeros(self.dim_H)
        
        try:
            import torch
            x_tensor = torch.FloatTensor(x)
            if len(x_tensor.shape) == 1:
                x_tensor = x_tensor.unsqueeze(0)
            with torch.no_grad():
                embedded = self.model(x_tensor)
            return embedded.numpy().flatten()
        except:
            return np.zeros(self.dim_H)


class SpectralEmbeddingWrapper:
    """Wrapper for sklearn SpectralEmbedding."""
    
    def __init__(self, dim_H: int = 128, affinity: str = 'nearest_neighbors',
                 n_neighbors: int = 10):
        """Initialize spectral embedding."""
        self.dim_H = dim_H
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.embedding = SpectralEmbedding(n_components=dim_H, 
                                          affinity=affinity,
                                          n_neighbors=n_neighbors)
        self.L = None
    
    def fit(self, X: np.ndarray, d1: Optional[np.ndarray] = None):
        """Fit embedding."""
        if d1 is not None:
            sigma = np.median(d1[d1 > 0])
            affinity = np.exp(-d1**2 / (2 * sigma**2))
            pass
        
        self.embedding.fit(X)
        if hasattr(self.embedding, 'embedding_'):
            emb_matrix = self.embedding.embedding_
            self.L = np.linalg.norm(emb_matrix, ord=2)
        else:
            raise RuntimeError("SpectralEmbedding fit failed - embedding_ not available")
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Embed point x."""
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        return self.embedding.transform(x).flatten()


def create_embedding(embedding_type: str = 'default', **kwargs) -> Tuple[Callable, float]:
    """Factory function to create embeddings. Returns (phi, L)."""
    dim_H = kwargs.get('dim_H', 128)
    
    if embedding_type == 'laplacian':
        emb = GraphLaplacianEmbedding(dim_H=dim_H, **kwargs)
        # Need to fit on data first
        def phi(x):
            return emb(x)
        return phi, emb.L
    
    elif embedding_type == 'learned':
        emb = LearnedEmbedding(dim_H=dim_H, **kwargs)
        def phi(x):
            return emb(x)
        return phi, emb.L
    
    elif embedding_type == 'spectral':
        emb = SpectralEmbeddingWrapper(dim_H=dim_H, **kwargs)
        def phi(x):
            return emb(x)
        return phi, emb.L
    
    elif embedding_type == 'pca':
        pca = PCA(n_components=dim_H)
        def phi(x):
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            return pca.transform(x).flatten()
        L = 1.0  # Will be set after fit
        return phi, L
    
    else:
        emb = GraphLaplacianEmbedding(dim_H=dim_H)
        def phi(x):
            return emb(x)
        return phi, emb.L

