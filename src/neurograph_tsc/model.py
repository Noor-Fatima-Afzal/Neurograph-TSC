from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class NeuroGraphTSC(nn.Module):
    """
    - Per time step: apply a GAT layer over 32 nodes with 1 feature per node.
    - Stack time steps and feed to an LSTM.
    - Classify from the last hidden state.

    x_seq: [B, T, N] (N=32), values are scalar node features at each step.
    """
    def __init__(self, in_channels: int = 1, gat_out: int = 64,
                 lstm_hidden: int = 128, num_classes: int = 4):
        super().__init__()
        self.gat = GATConv(in_channels, gat_out)
        self.lstm = nn.LSTM(gat_out * 32, lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor):
        """
        x_seq: [B, T, N]
        edge_index: [2, E]
        """
        B, T, N = x_seq.shape
        outs = []
        # Process each item independently (simple & clear)
        for b in range(B):
            H_t = []
            for t in range(T):
                xt = x_seq[b, t].unsqueeze(-1)           # [N] -> [N, 1]
                h = self.gat(xt, edge_index)             # [N, gat_out]
                H_t.append(h.flatten())                   # [N*gat_out]
            H_t = torch.stack(H_t, dim=0)                 # [T, N*gat_out]
            out, _ = self.lstm(H_t.unsqueeze(0))          # [1, T, hidden]
            outs.append(out[:, -1, :])                    # last time step
        H_last = torch.cat(outs, dim=0)                   # [B, hidden]
        return self.fc(H_last)                            # [B, C]
