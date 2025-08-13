from __future__ import annotations
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

class EEGDatasetNPY:
    """
    Loads (N, T, 32) data and (N,) labels from .npy files.
    Provides label encoding and returns tensors.
    """
    def __init__(self, data_path: str, labels_path: str):
        X = np.load(data_path)   # (N, T, 32)
        y = np.load(labels_path) # (N,)
        if X.ndim != 3 or X.shape[-1] != 32:
            raise ValueError(f"Expected X shape (N, T, 32), got {X.shape}")

        self.label_encoder = LabelEncoder()
        if y.dtype.kind in ("U", "S", "O"):
            y_enc = self.label_encoder.fit_transform(y)
        else:
            # already numeric
            y_enc = y.astype(int)

        self.X = torch.tensor(X, dtype=torch.float32)   # [N, T, 32]
        self.y = torch.tensor(y_enc, dtype=torch.long)  # [N]
        self.classes_ = list(self.label_encoder.classes_) if hasattr(self.label_encoder, "classes_") else None

    def tensors(self):
        return self.X, self.y
