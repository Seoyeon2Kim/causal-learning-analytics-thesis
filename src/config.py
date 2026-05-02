# src/config.py
from pathlib import Path

class ExperimentConfig:
    # Dataset
    dataset_name: str = "oulad"
    
    # Paths
    output_base: str = "/kaggle/working/results"
    
    @property
    def output_dir(self) -> Path:
        p = Path(self.output_base) / self.dataset_name
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Training windows
    early_weeks: list = None

    # Model hyperparameters
    hidden_dim: int = 64
    num_epochs: int = 15
    learning_rate: float = 1e-3
    batch_size: int = 128
    seq_max_len: int = 12
    random_state: int = 42

    # Causal
    causal_lambda: float = 0.10

    def __init__(self):
        self.early_weeks = [2, 4, 6, 8, 10, 12]