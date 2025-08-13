from __future__ import annotations
import torch
import numpy as np
from typing import Tuple

def set_seed(seed: int = 42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def accuracy(pred_logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = pred_logits.argmax(dim=1)
    return (pred == targets).float().mean().item()

def select_timesteps(X: torch.Tensor, T: int) -> torch.Tensor:
    """
    X: [N, T_total, 32]
    """
    if T <= 0 or T >= X.shape[1]:
        return X
    return X[:, :T, :]
