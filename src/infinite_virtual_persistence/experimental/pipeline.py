"""
End-to-end computational pipeline for virtual persistence diagram learning.
"""

import numpy as np
from typing import Tuple, Optional, Callable, Dict, List, TYPE_CHECKING
from ..theoretical.virtual_diagrams import MetricPair, PersistenceDiagram, VirtualDiagram
from ..theoretical.kernels import HilbertEmbedding, GaussianRKHSKernel, RandomFourierFeatures
from ..theoretical.barycenter import compute_class_barycenters
from .learning import KernelRidge, KernelSVM
from ..persistence import compute_persistence_diagram

if TYPE_CHECKING:
    from tqdm import tqdm


class VirtualPersistencePipeline:
    """
    Complete pipeline: raw data -> filtration -> persistence diagram -> 
    virtual diagram (D(x) - R_c) -> concatenated embeddings -> kernel -> learning.
    """
    
    def __init__(self, 
                 phi: Callable[[np.ndarray], np.ndarray],
                 L: float,
                 Sigma: np.ndarray,
                 t: float = 1.0,
                 lambda_reg: float = 1.0,
                 R: Optional[int] = None,
                 aggregation: str = 'concatenate'):
        """
        Initialize pipeline.
        
        Args:
            phi: Hilbert embedding phi: (X/A, d1, [A]) -> H
            L: Lipschitz constant of phi
            Sigma: Covariance operator on ran(J)
            t: Kernel scale parameter
            lambda_reg: Regularization for kernel ridge
            R: Number of random Fourier features (None = use exact kernel)
            aggregation: 'sum' (default, recommended) or 'mean' for pooling point embeddings.
                        'sum' gives J(D) = sum φ(p), which is mathematically correct.
                        Both yield fixed-dimensional embeddings.
        """
        self.phi = phi
        self.L = L
        self.Sigma = Sigma
        self.t = t
        self.lambda_reg = lambda_reg
        self.R = R
        self.aggregation = aggregation
        
        # Initialize components
        self.embedding = HilbertEmbedding(phi, L)
        self.kernel = GaussianRKHSKernel(self.embedding, Sigma, t)
        
        if R is not None:
            self.rff = RandomFourierFeatures(self.kernel, R)
        else:
            self.rff = None
        
        # Reference diagrams (computed per class from training data)
        # These are fold-scoped to prevent leakage
        self.reference_diagrams = {}
        self.classes = None
        self.metric_pair = None
        self.fold_id = None  # Track which fold these references belong to
        self._fit_fold_id = None  # Internal: fold_id used during fit
    
    def compute_persistence_diagram(self, X, 
                                   is_graph: bool = False,
                                   tau: float = 1.0,
                                   heat_method: str = 'content',
                                   method: str = 'auto',
                                   max_dim: int = 2) -> PersistenceDiagram:
        """
        Compute persistence diagram from input data.
        
        Args:
            X: Input data (point cloud array or graph data)
            is_graph: Whether input is a graph
            tau: Heat diffusion time (for graphs)
            heat_method: Heat method for graphs
            method: 'ripser', 'gudhi', 'alpha', or 'auto'
            max_dim: Maximum dimension of simplices (default: 2 for H₁)
        
        Returns:
            Persistence diagram
        """
        points = compute_persistence_diagram(
            X, 
            method=method, 
            max_points=500,
            is_graph=is_graph,
            tau=tau,
            heat_method=heat_method,
            max_dim=max_dim
        )
        return PersistenceDiagram(points)
    
    def virtual_diagram_to_embedding(self, virtual: VirtualDiagram,
                                    embedded_points: np.ndarray,
                                    persistence_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert virtual diagram to embedding vector using sum-pooled J(D).
        
        Implements J(D) = sum_{p in D} φ(p) (optionally weighted by persistence).
        This gives a fixed-dimensional embedding regardless of diagram size.
        
        Args:
            virtual: Virtual diagram
            embedded_points: Persistence points (already in persistence space)
            persistence_weights: Optional weights (e.g., persistence = death - birth)
                                 If None, uses uniform weights
        
        Returns:
            Embedding vector in H (fixed dimension)
        """
        if len(embedded_points) == 0:
            return np.zeros(self.Sigma.shape[0])
        
        # Embed each persistence point via phi
        embedded = np.array([self.embedding(p) for p in embedded_points])
        
        # Compute weights if not provided
        if persistence_weights is None:
            # Option 1: Uniform weights (unweighted sum)
            weights = np.ones(len(embedded_points))
        else:
            weights = np.asarray(persistence_weights)
            if len(weights) != len(embedded_points):
                raise ValueError("Weights length must match embedded_points length")
        
        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        # Sum-pooled embedding: J(D) = sum_{p in D} w(p) * φ(p)
        # This is the canonical linear extension to K(X,A)
        if self.aggregation == 'sum' or self.aggregation == 'concatenate':
            # Sum-pooled (fixed dimension, mathematically correct)
            return np.sum(weights.reshape(-1, 1) * embedded, axis=0)
        elif self.aggregation == 'mean':
            # Mean-pooled (also fixed dimension, normalized by count)
            return np.mean(embedded, axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}. Use 'sum' or 'mean'.")
    
    def diagram_to_virtual_embeddings(self, diagram: PersistenceDiagram,
                                      reference_diagrams: Dict[int, PersistenceDiagram],
                                      metric_pair: MetricPair,
                                      use_persistence_weights: bool = True) -> List[np.ndarray]:
        """
        Convert persistence diagram to list of virtual diagram embeddings.
        
        Creates D(x) - R_c for each class c, then embeds using J(D-R) = J(D) - J(R).
        This is the mathematically correct embedding of the virtual diagram.
        
        Args:
            diagram: Observed persistence diagram
            reference_diagrams: Dictionary mapping class -> reference diagram
            metric_pair: Metric pair for computing differences
            use_persistence_weights: If True, weight by persistence (death - birth)
        
        Returns:
            List of embedding vectors (one per class), each of fixed dimension
        """
        embeddings = []
        
        # Compute persistence weights for diagram if requested
        diagram_weights = None
        if use_persistence_weights and len(diagram.points) > 0:
            diagram_weights = diagram.points[:, 1] - diagram.points[:, 0]  # death - birth
        
        for class_label, ref_diagram in reference_diagrams.items():
            # Compute J(D(x)) = sum_{p in D(x)} w(p) * φ(p)
            diagram_embedded = self.virtual_diagram_to_embedding(
                VirtualDiagram({tuple(p): 1 for p in diagram.points}), 
                diagram.points,
                persistence_weights=diagram_weights
            )
            
            # Compute J(R_c) = sum_{q in R_c} w(q) * φ(q)
            ref_weights = None
            if use_persistence_weights and len(ref_diagram.points) > 0:
                ref_weights = ref_diagram.points[:, 1] - ref_diagram.points[:, 0]
            
            ref_embedded = self.virtual_diagram_to_embedding(
                VirtualDiagram({tuple(q): 1 for q in ref_diagram.points}),
                ref_diagram.points,
                persistence_weights=ref_weights
            )
            
            # Virtual diagram embedding: J(D(x) - R_c) = J(D(x)) - J(R_c)
            # This is the canonical linear extension to K(X,A)
            virtual_embedding = diagram_embedded - ref_embedded
            embeddings.append(virtual_embedding)
        
        return embeddings
    
    def fit(self, X_train: List, y_train: np.ndarray, 
            task: str = 'classification',
            is_graph: bool = False,
            tau: float = 1.0,
            heat_method: str = 'content',
            chunk_size: int = 10, 
            start_idx: int = 0, 
            embedded_train: Optional[List[np.ndarray]] = None,
            pbar: Optional['tqdm'] = None,
            fold_id: Optional[str] = None):
        """
        Fit model on training data.
        
        Computes reference diagrams (barycenters) per class from training data only.
        All training-derived artifacts (references, metric_pair) are fold-scoped.
        
        Args:
            X_train: Training data (list of graphs or point clouds)
            y_train: Training labels
            task: 'regression' or 'classification'
            is_graph: Whether input is graph data
            tau: Heat diffusion time (for graphs)
            heat_method: Heat method for graphs
            chunk_size: Number of samples to process before saving checkpoint
            start_idx: Starting index (for resuming)
            embedded_train: Pre-computed embeddings (for resuming)
            pbar: Optional tqdm progress bar to update
            fold_id: Optional fold identifier for leakage prevention.
                    If provided, must match when calling predict().
                    Recommended format: "fold_{fold_num}" or "train_fold_{fold_num}"
        """
        # Step 1: Compute persistence diagrams for all training data
        if start_idx == 0:
            # Set fold_id for this fit (leakage prevention)
            if fold_id is not None:
                self._fit_fold_id = fold_id
                self.fold_id = fold_id
                print(f"Fitting with fold_id={fold_id} (fold-scoped references)")
            else:
                self._fit_fold_id = "default"
                self.fold_id = "default"
                print("Warning: No fold_id provided. Using 'default'. "
                      "For cross-validation, provide explicit fold_id to prevent leakage.")
            
            print("Computing persistence diagrams for training data...")
            train_diagrams = []
            for idx, X in enumerate(X_train):
                if pbar is not None:
                    pbar.set_postfix_str(f"Computing PD {idx+1}/{len(X_train)}")
                diagram = self.compute_persistence_diagram(
                    X, is_graph=is_graph, tau=tau, heat_method=heat_method
                )
                train_diagrams.append(diagram)
                if pbar is not None:
                    pbar.update(1)
            
            # Step 2: Create metric pair from all training persistence points
            # FOLD-SCOPED: Only uses training data from this fold
            all_points = []
            for diagram in train_diagrams:
                all_points.extend(diagram.points.tolist())
            all_points = np.array(all_points)
            self.metric_pair = MetricPair(all_points)
            
            # Step 3: Compute reference diagrams (barycenters) per class
            # FOLD-SCOPED: Only uses training data from this fold
            print("Computing barycenter reference diagrams per class...")
            self.classes = np.unique(y_train)
            self.reference_diagrams = compute_class_barycenters(
                train_diagrams, y_train, self.metric_pair, classes=self.classes
            )
            print(f"Computed {len(self.reference_diagrams)} reference diagrams "
                  f"(fold_id={self.fold_id})")
        
        # Step 4: Compute virtual diagram embeddings
        if embedded_train is None:
            embedded_train = []
        else:
            embedded_train = list(embedded_train)
        
        n_total = len(X_train)
        end_idx = min(start_idx + chunk_size, n_total)
        
        for idx in range(start_idx, end_idx):
            X = X_train[idx]
            
            # Compute persistence diagram
            diagram = self.compute_persistence_diagram(
                X, is_graph=is_graph, tau=tau, heat_method=heat_method
            )
            
            # Convert to list of virtual diagram embeddings (one per class)
            # Each embedding is fixed-dimensional (dim_H)
            virtual_embeddings = self.diagram_to_virtual_embeddings(
                diagram, self.reference_diagrams, self.metric_pair
            )
            
            # Combine embeddings from all classes
            # Each virtual_embeddings[i] is of dimension dim_H
            if self.aggregation == 'concatenate' or self.aggregation == 'sum':
                # Concatenate across classes: [J(D-R_1), ..., J(D-R_C)]
                # Dimension: C × dim_H (fixed for fixed number of classes)
                final_embedding = np.concatenate(virtual_embeddings)
            else:
                # Mean aggregation: take mean across classes
                final_embedding = np.mean(virtual_embeddings, axis=0)
            
            embedded_train.append(final_embedding)
            
            if pbar is not None:
                pbar.update(1)
        
        # Step 5: Fit model if all training samples processed
        if end_idx >= n_total:
            embedded_train_array = np.array(embedded_train)
            
            # Fit model
            if task == 'regression':
                self.model = KernelRidge(self.kernel, self.lambda_reg)
            else:
                self.model = KernelSVM(self.kernel, task='classification')
            
            self.model.fit(embedded_train_array, y_train)
            self.X_train_embedded = embedded_train_array
        
        return embedded_train
    
    def predict(self, X_test: List, 
                is_graph: bool = False,
                tau: float = 1.0,
                heat_method: str = 'content',
                chunk_size: int = 10, 
                start_idx: int = 0, 
                embedded_test: Optional[List[np.ndarray]] = None,
                y_pred: Optional[List] = None, 
                pbar: Optional['tqdm'] = None,
                fold_id: Optional[str] = None) -> np.ndarray:
        """
        Predict on test data.
        
        Args:
            X_test: Test data
            is_graph: Whether input is graph data
            tau: Heat diffusion time (for graphs)
            heat_method: Heat method for graphs
            chunk_size: Number of samples to process before saving checkpoint
            start_idx: Starting index (for resuming)
            embedded_test: Pre-computed embeddings (for resuming)
            y_pred: Pre-computed predictions (for resuming)
            pbar: Optional tqdm progress bar to update
            fold_id: Optional fold identifier. If provided, must match the fold_id
                    used during fit() to ensure no leakage.
        
        Returns:
            Predictions array
        """
        if self.reference_diagrams is None or self.metric_pair is None:
            raise ValueError("Must call fit() before predict()")
        
        # Leakage prevention: verify fold_id matches
        if fold_id is not None and self.fold_id is not None:
            if fold_id != self.fold_id:
                raise ValueError(
                    f"fold_id mismatch: fit() used fold_id='{self.fold_id}', "
                    f"but predict() called with fold_id='{fold_id}'. "
                    f"This may indicate leakage. Ensure test data uses references "
                    f"from the same fold as training."
                )
        elif fold_id is not None and self.fold_id is None:
            # fit() didn't use fold_id, but predict() does - warn
            import warnings
            warnings.warn(
                f"predict() called with fold_id='{fold_id}', but fit() did not use fold_id. "
                f"Cannot verify fold consistency. Consider using fold_id in fit() as well."
            )
        
        if embedded_test is None:
            embedded_test = []
        else:
            embedded_test = list(embedded_test)
        
        if y_pred is None:
            y_pred = []
        else:
            y_pred = list(y_pred)
        
        n_total = len(X_test)
        end_idx = min(start_idx + chunk_size, n_total)
        new_embedded = []
        
        for idx in range(start_idx, end_idx):
            X = X_test[idx]
            
            # Compute persistence diagram
            diagram = self.compute_persistence_diagram(
                X, is_graph=is_graph, tau=tau, heat_method=heat_method
            )
            
            # Convert to virtual diagram embeddings
            virtual_embeddings = self.diagram_to_virtual_embeddings(
                diagram, self.reference_diagrams, self.metric_pair
            )
            
            # Combine embeddings from all classes
            if self.aggregation == 'concatenate' or self.aggregation == 'sum':
                final_embedding = np.concatenate(virtual_embeddings)
            else:
                final_embedding = np.mean(virtual_embeddings, axis=0)
            
            new_embedded.append(final_embedding)
            
            if pbar is not None:
                pbar.update(1)
        
        # Predict
        if len(new_embedded) > 0:
            new_embedded_array = np.array(new_embedded)
            new_predictions = self.model.predict(new_embedded_array)
            y_pred.extend(new_predictions)
            if embedded_test is not None:
                embedded_test.extend(new_embedded)
        
        return np.array(y_pred)


def create_default_embedding(dim_H: int = 128) -> Tuple[Callable, float]:
    """
    Create default Hilbert embedding.
    
    Returns:
        (phi, L) where phi maps to R^dim_H and L is Lipschitz constant
    """
    rng = np.random.RandomState(14)
    W = rng.randn(dim_H, 2)  # Assuming 2D persistence space
    
    def phi(x: np.ndarray) -> np.ndarray:
        """Embed persistence point to H."""
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        return (W @ x.T).T.flatten()
    
    L = np.linalg.norm(W, ord=2)
    return phi, L


def create_default_covariance(dim_H: int = 128, scale: float = 1.0) -> np.ndarray:
    """
    Create default covariance operator.
    
    Args:
        dim_H: Dimension of Hilbert space
        scale: Scale parameter
    
    Returns:
        Covariance matrix Sigma
    """
    return scale**2 * np.eye(dim_H)
