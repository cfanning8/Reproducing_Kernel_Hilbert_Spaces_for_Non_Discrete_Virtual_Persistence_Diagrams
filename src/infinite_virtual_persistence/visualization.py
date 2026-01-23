"""
Visualization tools for persistence diagrams and embeddings.
"""

import numpy as np
from typing import Optional, List, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings


def plot_persistence_diagram(diagram: np.ndarray, 
                           max_persistence: Optional[float] = None,
                           title: str = 'Persistence Diagram',
                           save_path: Optional[str] = None,
                           show: bool = False) -> plt.Figure:
    """
    Plot persistence diagram.
    
    Args:
        diagram: Array of (birth, death) pairs, shape (n, 2)
        max_persistence: Maximum persistence to display
        title: Plot title
        save_path: Path to save figure
        show: Whether to display figure
    
    Returns:
        matplotlib Figure
    """
    if len(diagram) == 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(0.5, 0.5, 'Empty Persistence Diagram', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')
        ax.set_title(title)
        return fig
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    persistence = deaths - births
    
    # Color by persistence
    scatter = ax.scatter(births, deaths, c=persistence, 
                        cmap='viridis', s=50, alpha=0.6, edgecolors='black')
    
    # Diagonal line
    min_val = min(births.min(), deaths.min())
    max_val = max(births.max(), deaths.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
           'r--', linewidth=2, label='Diagonal')
    
    ax.set_xlabel('Birth', fontsize=12)
    ax.set_ylabel('Death', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.colorbar(scatter, ax=ax, label='Persistence')
    
    if max_persistence:
        ax.set_xlim(0, max_persistence)
        ax.set_ylim(0, max_persistence)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_persistence_landscape(diagram: np.ndarray,
                               num_levels: int = 10,
                               resolution: int = 100,
                               title: str = 'Persistence Landscape',
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot persistence landscape.
    
    Args:
        diagram: Persistence diagram
        num_levels: Number of landscape levels
        resolution: Resolution for landscape computation
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    if len(diagram) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Empty Persistence Diagram', 
               ha='center', va='center', transform=ax.transAxes)
        return fig
    
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    
    # Compute landscape
    t_min = births.min()
    t_max = deaths.max()
    t = np.linspace(t_min, t_max, resolution)
    
    landscapes = []
    for level in range(num_levels):
        landscape = np.zeros(resolution)
        for b, d in zip(births, deaths):
            # Lambda_k(t) = max(0, min(t - b, d - t))
            lambda_t = np.maximum(0, np.minimum(t - b, d - t))
            landscape = np.maximum(landscape, lambda_t)
        landscapes.append(landscape)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for level, landscape in enumerate(landscapes):
        ax.plot(t, landscape, label=f'Level {level + 1}', linewidth=2)
    
    ax.set_xlabel('t', fontsize=12)
    ax.set_ylabel('Lambda(t)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    return fig


def plot_embedding_2d(embeddings: np.ndarray,
                     labels: Optional[np.ndarray] = None,
                     title: str = '2D Embedding',
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 2D projection of embeddings.
    
    Args:
        embeddings: Embeddings, shape (n, dim_H)
        labels: Optional labels for coloring
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    from sklearn.decomposition import PCA
    
    # Project to 2D if needed
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        explained_var = pca.explained_variance_ratio_.sum()
        title += f' (PCA, {explained_var:.2%} variance)'
    else:
        embeddings_2d = embeddings
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None:
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                           c=labels, cmap='tab10', s=50, alpha=0.6, 
                           edgecolors='black')
        plt.colorbar(scatter, ax=ax, label='Class')
    else:
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                  s=50, alpha=0.6, edgecolors='black')
    
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    return fig


def plot_embedding_3d(embeddings: np.ndarray,
                     labels: Optional[np.ndarray] = None,
                     title: str = '3D Embedding',
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 3D projection of embeddings.
    
    Args:
        embeddings: Embeddings, shape (n, dim_H)
        labels: Optional labels for coloring
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    from sklearn.decomposition import PCA
    
    # Project to 3D if needed
    if embeddings.shape[1] > 3:
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embeddings)
        explained_var = pca.explained_variance_ratio_.sum()
        title += f' (PCA, {explained_var:.2%} variance)'
    else:
        embeddings_3d = embeddings
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if labels is not None:
        scatter = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], 
                           embeddings_3d[:, 2], c=labels, cmap='tab10', 
                           s=50, alpha=0.6, edgecolors='black')
        plt.colorbar(scatter, ax=ax, label='Class')
    else:
        ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], 
                  embeddings_3d[:, 2], s=50, alpha=0.6, edgecolors='black')
    
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_zlabel('Dimension 3', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    return fig


def plot_point_cloud_3d(points: np.ndarray,
                       color: Optional[np.ndarray] = None,
                       title: str = '3D Point Cloud',
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 3D point cloud.
    
    Args:
        points: Point cloud, shape (n, 3)
        color: Optional coloring
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    if points.shape[1] != 3:
        warnings.warn(f"Points have {points.shape[1]} dimensions, expected 3")
        return None
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if color is not None:
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                           c=color, cmap='viridis', s=20, alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Value')
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  s=20, alpha=0.6)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    return fig


def plot_lipschitz_analysis(lipschitz_estimates: np.ndarray,
                            theoretical_bounds: Optional[np.ndarray] = None,
                            title: str = 'Lipschitz Constant Analysis',
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Lipschitz constant analysis.
    
    Args:
        lipschitz_estimates: Estimated Lipschitz constants
        theoretical_bounds: Theoretical bounds (optional)
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(lipschitz_estimates, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(lipschitz_estimates), color='r', 
                   linestyle='--', linewidth=2, label=f'Mean: {np.mean(lipschitz_estimates):.4f}')
    axes[0].axvline(np.median(lipschitz_estimates), color='g', 
                   linestyle='--', linewidth=2, label=f'Median: {np.median(lipschitz_estimates):.4f}')
    axes[0].set_xlabel('Lipschitz Constant', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Lipschitz Constants', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Comparison with theoretical bounds
    if theoretical_bounds is not None:
        axes[1].scatter(theoretical_bounds, lipschitz_estimates, 
                       alpha=0.6, s=50, edgecolors='black')
        min_val = min(theoretical_bounds.min(), lipschitz_estimates.min())
        max_val = max(theoretical_bounds.max(), lipschitz_estimates.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 
                    'r--', linewidth=2, label='y=x')
        axes[1].set_xlabel('Theoretical Bound', fontsize=12)
        axes[1].set_ylabel('Estimated Lipschitz Constant', fontsize=12)
        axes[1].set_title('Theoretical vs Estimated', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No theoretical bounds provided', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Theoretical vs Estimated', fontsize=12)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    return fig


def create_summary_visualization(diagrams: List[np.ndarray],
                                embeddings: np.ndarray,
                                labels: Optional[np.ndarray] = None,
                                save_dir: Optional[str] = None) -> List[plt.Figure]:
    """
    Create comprehensive summary visualization.
    
    Args:
        diagrams: List of persistence diagrams
        embeddings: Embeddings, shape (n, dim_H)
        labels: Optional labels
        save_dir: Directory to save figures
    
    Returns:
        List of figures
    """
    figures = []
    
    # Plot sample persistence diagrams
    for i, diagram in enumerate(diagrams[:5]):  # First 5
        fig = plot_persistence_diagram(diagram, 
                                      title=f'Persistence Diagram {i+1}')
        figures.append(fig)
        if save_dir:
            fig.savefig(f'{save_dir}/persistence_diagram_{i+1}.png', 
                       dpi=150, bbox_inches='tight')
    
    # Plot embedding
    if embeddings.shape[1] >= 2:
        fig = plot_embedding_2d(embeddings, labels, 
                               title='Hilbert Space Embedding')
        figures.append(fig)
        if save_dir:
            fig.savefig(f'{save_dir}/embedding_2d.png', 
                       dpi=150, bbox_inches='tight')
    
    if embeddings.shape[1] >= 3:
        fig = plot_embedding_3d(embeddings, labels, 
                               title='Hilbert Space Embedding (3D)')
        figures.append(fig)
        if save_dir:
            fig.savefig(f'{save_dir}/embedding_3d.png', 
                       dpi=150, bbox_inches='tight')
    
    return figures



