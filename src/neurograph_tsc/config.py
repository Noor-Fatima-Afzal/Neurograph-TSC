from dataclasses import dataclass

@dataclass
class TrainConfig:
    data_path: str
    labels_path: str
    positions_csv: str | None = None 

    # model
    in_channels: int = 1
    gat_out: int = 64
    lstm_hidden: int = 128
    num_classes: int | None = None

    # training
    epochs: int = 20
    batch_size: int = 0    
    lr: float = 1e-3
    time_steps: int = 100  

    # regularizers (λ)
    lambda_conn: float = 0.1
    lambda_nmm: float = 0.1
    lambda_band: float = 0.1
