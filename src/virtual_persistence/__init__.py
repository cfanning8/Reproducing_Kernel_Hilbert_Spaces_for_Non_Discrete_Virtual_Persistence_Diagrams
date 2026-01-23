"""
Virtual Persistence: RKHS for Non-Discrete Virtual Persistence Diagrams

This package implements the theoretical framework for reproducing kernel Hilbert
spaces on virtual persistence diagrams in the non-discrete case, as described in:

Fanning, C. & Aktas, M. E. (2025). Reproducing Kernel Hilbert Spaces for 
Non-Discrete Virtual Persistence Diagrams.
"""

__version__ = "0.1.0"

from .virtual_diagrams import (
    MetricPair,
    PersistenceDiagram,
    VirtualDiagram,
    grothendieck_metric
)

from .kernels import (
    HilbertEmbedding,
    GaussianRKHSKernel,
    RandomFourierFeatures
)

from .embeddings import (
    GraphLaplacianEmbedding,
    LearnedEmbedding,
    SpectralEmbeddingWrapper,
    create_embedding
)

# Persistence computation
from .persistence import (
    compute_persistence_diagram,
    compute_persistence_diagram_alpha,
    lower_star_filtration_graph
)

__all__ = [
    # Virtual diagrams
    'MetricPair',
    'PersistenceDiagram',
    'VirtualDiagram',
    'grothendieck_metric',
    # Kernels
    'HilbertEmbedding',
    'GaussianRKHSKernel',
    'RandomFourierFeatures',
    # Embeddings
    'GraphLaplacianEmbedding',
    'LearnedEmbedding',
    'SpectralEmbeddingWrapper',
    'create_embedding',
    # Persistence computation
    'compute_persistence_diagram',
    'compute_persistence_diagram_alpha',
    'lower_star_filtration_graph',
]
