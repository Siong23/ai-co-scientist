from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    """Turns a feature matrix into fixed-length overlapping sequences."""
    def __init__(self, features, labels, sequence_length=5, stride=1):
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)

        if len(features) != len(labels):
            raise ValueError("Features and labels must have the same length.")
        if len(features) < sequence_length:
            raise ValueError(
                f"Need at least {sequence_length} rows, got {len(features)}."
            )

        self.x = []
        self.y = []

        for start in range(0, len(features) - sequence_length + 1, stride):
            end = start + sequence_length
            self.x.append(features[start:end])
            # Sequence label = label of final timestep.
            self.y.append(labels[end - 1])

        self.x = torch.tensor(np.stack(self.x), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(self.y), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def make_sequences(features, labels, sequence_length=5, stride=1):
    ds = SequenceDataset(features, labels, sequence_length, stride)
    return ds.x, ds.y
