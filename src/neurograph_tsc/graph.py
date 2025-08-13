from __future__ import annotations
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse

def load_positions_or_random(positions_csv: str | None, num_nodes: int = 32) -> np.ndarray:
    if positions_csv:
        arr = np.loadtxt(positions_csv, delimiter=",")
        if arr.shape != (num_nodes, 3):
            raise ValueError(f"positions CSV must be shape ({num_nodes}, 3), got {arr.shape}")
        return arr
    # fallback: random positions in unit cube
    rng = np.random.default_rng(42)
    return rng.random((num_nodes, 3))

def gaussian_adjacency_from_positions(positions: np.ndarray, sigma: float = 0.5) -> torch.Tensor:
    """
    positions: (N, 3) numpy
    returns: (N, N) torch float adjacency with self-loops zeroed.
    """
    dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)  # (N,N)
    adj = np.exp(-(dists ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(adj, 0.0)
    return torch.tensor(adj, dtype=torch.float32)

def to_edge_index(adj: torch.Tensor):
    """
    Convert dense adjacency to PyG COO edge_index and edge_weight (optional).
    Note: GATConv ignores edge_weight, but we return it for completeness.
    """
    edge_index, edge_weight = dense_to_sparse(adj)
    return edge_index, edge_weight
