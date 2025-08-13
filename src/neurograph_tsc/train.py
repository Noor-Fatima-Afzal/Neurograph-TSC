from __future__ import annotations
import argparse
import torch
from .config import TrainConfig
from .data import EEGDatasetNPY
from .graph import load_positions_or_random, gaussian_adjacency_from_positions, to_edge_index
from .model import NeuroGraphTSC
from .loss import compute_total_loss
from .utils import set_seed, accuracy, select_timesteps

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train NeuroGraph-TSC")
    p.add_argument("--data-path", required=True, type=str)
    p.add_argument("--labels-path", required=True, type=str)
    p.add_argument("--positions-csv", default=None, type=str)

    p.add_argument("--epochs", default=20, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--time-steps", default=100, type=int)
    p.add_argument("--gat-out", default=64, type=int)
    p.add_argument("--lstm-hidden", default=128, type=int)

    p.add_argument("--lambda-conn", default=0.1, type=float)
    p.add_argument("--lambda-nmm", default=0.1, type=float)
    p.add_argument("--lambda-band", default=0.1, type=float)

    return p

def main() -> int:
    args = build_argparser().parse_args()
    set_seed(42)

    cfg = TrainConfig(
        data_path=args.data_path,
        labels_path=args.labels_path,
        positions_csv=args.positions_csv,
        epochs=args.epochs,
        lr=args.lr,
        time_steps=args.time_steps,
        gat_out=args.gat_out,
        lstm_hidden=args.lstm_hidden,
        lambda_conn=args.lambda_conn,
        lambda_nmm=args.lambda_nmm,
        lambda_band=args.lambda_band,
    )

    # ---- data
    ds = EEGDatasetNPY(cfg.data_path, cfg.labels_path)
    X, y = ds.tensors()
    if cfg.num_classes is None:
        num_classes = int(y.max().item()) + 1
    else:
        num_classes = cfg.num_classes

    # time crop 
    X = select_timesteps(X, cfg.time_steps)  # [N, T, 32]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    y = y.to(device)

    # ---- graph
    pos = load_positions_or_random(cfg.positions_csv, num_nodes=32)
    A_prior = gaussian_adjacency_from_positions(pos, sigma=0.5).to(device)  # [32, 32]
    edge_index, edge_weight = to_edge_index(A_prior)  # edge_weight unused by GAT
    edge_index = edge_index.to(device)

    # ---- model
    model = NeuroGraphTSC(
        in_channels=1,
        gat_out=cfg.gat_out,
        lstm_hidden=cfg.lstm_hidden,
        num_classes=num_classes,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # ---- training 
    N = X.shape[0]
    for epoch in range(cfg.epochs):
        model.train()
        optim.zero_grad()
        logits = model(X, edge_index)  # [N, C]

        A_pred = torch.rand_like(A_prior)                 # [32, 32]
        x_model = X.mean(dim=-1)                          # [N, T]
        x_nmm = x_model + 0.1 * torch.randn_like(x_model) # synthetic target
        p_alpha = torch.tensor(0.8, device=device)
        p_beta  = torch.tensor(1.2, device=device)
        p_alpha_ref = torch.tensor(0.9, device=device)
        p_beta_ref  = torch.tensor(1.1, device=device)

        loss = compute_total_loss(
            logits, y, A_pred, A_prior, x_model, x_nmm,
            p_alpha, p_beta, p_alpha_ref, p_beta_ref,
            lambda_conn=cfg.lambda_conn,
            lambda_nmm=cfg.lambda_nmm,
            lambda_band=cfg.lambda_band,
        )
        loss.backward()
        optim.step()

        with torch.no_grad():
            acc = accuracy(logits, y) * 100.0

        print(f"Epoch {epoch+1:03d} | Loss {loss.item():.4f} | Acc {acc:.2f}%")

    return 0
