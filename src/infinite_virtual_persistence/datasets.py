"""
Dataset loaders for graph and point cloud datasets.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List
import os


def load_tu_dataset(name: str) -> Tuple[List, np.ndarray, List]:
    """
    Load TU Dortmund dataset.
    
    Args:
        name: Dataset name (e.g., 'MUTAG', 'PROTEINS')
    
    Returns:
        (X_train, y_train, X_test) where X are graph data objects
    """
    try:
        from torch_geometric.datasets import TUDataset
        from torch_geometric.transforms import ToUndirected
    except ImportError:
        raise ImportError("torch_geometric not installed. Install with: pip install torch-geometric")
    
    # Try multiple possible paths
    possible_paths = [
        Path('data') / 'appropriate' / 'raw' / name,
        Path('data') / 'appropriate' / 'raw' / 'TUDataset' / name,
        Path('data') / 'network' / 'appropriate' / 'TUDataset' / name,
        Path('data') / 'raw' / 'TUDataset' / name.lower(),
        Path('data') / 'raw' / 'TUDataset' / name,
    ]
    
    dataset_path = None
    for path in possible_paths:
        if path.exists():
            dataset_path = path.parent
            break
    
    if dataset_path is None:
        # Use default PyTorch Geometric path
        dataset_path = Path('data') / 'TUDataset'
        dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = TUDataset(
        root=str(dataset_path),
        name=name,
        transform=ToUndirected(),
        pre_transform=None
    )
    
    # Split into train/test (80/20 split)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    
    indices = np.random.RandomState(14).permutation(n_total)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = [dataset[i] for i in train_indices]
    y_train = np.array([dataset[i].y.item() for i in train_indices])
    X_test = [dataset[i] for i in test_indices]
    
    return X_train, y_train, X_test


def load_ogb_dataset(name: str):
    """Load OGB dataset (placeholder)."""
    raise NotImplementedError("OGB datasets not implemented yet")


def load_zinc():
    """Load ZINC dataset (placeholder)."""
    raise NotImplementedError("ZINC dataset not implemented yet")


def load_qm9():
    """Load QM9 dataset (placeholder)."""
    raise NotImplementedError("QM9 dataset not implemented yet")
