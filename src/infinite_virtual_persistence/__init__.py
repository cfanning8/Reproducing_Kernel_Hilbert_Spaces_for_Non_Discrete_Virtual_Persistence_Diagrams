"""
Infinite Virtual Persistence: RKHS for Virtual Persistence Diagrams
"""

__version__ = "0.1.0"

from .theoretical.virtual_diagrams import (
    MetricPair,
    PersistenceDiagram,
    VirtualDiagram,
    grothendieck_metric
)

from .theoretical.kernels import (
    HilbertEmbedding,
    GaussianRKHSKernel,
    RandomFourierFeatures
)

from .experimental.learning import (
    KernelRidge,
    KernelSVM,
    GaussianProcessWrapper,
    radev_complexity_bound
)

from .experimental.pipeline import (
    VirtualPersistencePipeline,
    create_default_embedding,
    create_default_covariance
)

from .theoretical.embeddings import (
    GraphLaplacianEmbedding,
    LearnedEmbedding,
    SpectralEmbeddingWrapper,
    create_embedding
)

from .experimental.evaluation import (
    estimate_lipschitz_constant,
    kernel_approximation_error,
    robustness_test
)

from .experimental.kernel_testing import (
    KernelApproximationTester,
    test_kernel_approximation
)

from .visualization import (
    plot_persistence_diagram,
    plot_persistence_landscape,
    plot_embedding_2d,
    plot_embedding_3d,
    create_summary_visualization
)

__all__ = [
    'MetricPair',
    'PersistenceDiagram',
    'VirtualDiagram',
    'grothendieck_metric',
    'HilbertEmbedding',
    'GaussianRKHSKernel',
    'RandomFourierFeatures',
    'KernelRidge',
    'KernelSVM',
    'GaussianProcessWrapper',
    'radev_complexity_bound',
    'VirtualPersistencePipeline',
    'create_default_embedding',
    'create_default_covariance',
    'GraphLaplacianEmbedding',
    'LearnedEmbedding',
    'SpectralEmbeddingWrapper',
    'create_embedding',
    'estimate_lipschitz_constant',
    'kernel_approximation_error',
    'robustness_test',
    'KernelApproximationTester',
    'test_kernel_approximation',
    'plot_persistence_diagram',
    'plot_persistence_landscape',
    'plot_embedding_2d',
    'plot_embedding_3d',
    'create_summary_visualization'
]
