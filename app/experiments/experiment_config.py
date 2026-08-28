from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class ExperimentConfig:
    data_path: str = "data/5g_nidd/5g_nidd.csv"
    output_dir: str = "app/experiments/results"

    target: str = "Label"
    binary: bool = True

    test_size: float = 0.30
    validation_size: float = 0.15
    random_state: int = 42

    sequence_length: int = 5
    sequence_stride: int = 1

    hidden_size: int = 64
    num_layers: int = 1
    bitad_hidden_size_1: int = 64
    bitad_hidden_size_2: int = 32

    dropout: float = 0.30
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10
    patience: int = 3

    models: list = field(
        default_factory=lambda: ["lstm", "lstm_attention", "bilstm", "bitad"]
    )

    device: str = "cuda"

    def resolve_device(self):
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return self.device
