from __future__ import annotations
import torch
import torch.nn as nn

def compute_total_loss(
    pred: torch.Tensor,               # [B, C]
    target: torch.Tensor,             # [B]
    A_pred: torch.Tensor,             # [N, N]
    A_prior: torch.Tensor,            # [N, N]
    x_model: torch.Tensor,            # [B, T]
    x_nmm: torch.Tensor,              # [B, T]
    p_alpha: torch.Tensor,            # scalar
    p_beta: torch.Tensor,             # scalar
    p_alpha_ref: torch.Tensor,        # scalar
    p_beta_ref: torch.Tensor,         # scalar
    lambda_conn: float = 0.1,
    lambda_nmm: float = 0.1,
    lambda_band: float = 0.1,
) -> torch.Tensor:
    """
    Cross-entropy + connection prior + NMM fit + bandpower proximity.
    """
    ce = nn.CrossEntropyLoss()(pred, target)
    conn = torch.norm(A_pred - A_prior, p='fro')
    nmm = nn.MSELoss()(x_model, x_nmm)
    band = (p_alpha - p_alpha_ref).abs() + (p_beta - p_beta_ref).abs()
    return ce + lambda_conn * conn + lambda_nmm * nmm + lambda_band * band
