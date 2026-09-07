import torch
from torch import nn

class TemporalSelfAttention(nn.Module):
    """
    Single-head temporal self-attention.
    The paper's BiTAD combines stacked BiLSTM layers with a single-head
    temporal self-attention mechanism.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** -0.5

    def forward(self, x):
        # x: [batch, time, hidden]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v)

        # Pool over time. Mean pooling keeps the representation stable.
        context = attended.mean(dim=1)
        return context, weights

class BiTAD(nn.Module):
    """
    PyTorch implementation of the BiTAD architecture described in
    Lau et al. (2025): stacked BiLSTM + single-head temporal attention.

    The paper describes two stacked BiLSTM layers with progressively
    reduced representation (128 -> 64 total output dimensions).
    Therefore hidden_size=64 per direction in layer 1 and 32 per
    direction in layer 2.
    """
    def __init__(self, input_size, hidden_size_1=64, hidden_size_2=32,
                 dropout=0.3, num_classes=2):
        super().__init__()

        self.bilstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.bilstm2 = nn.LSTM(
            input_size=hidden_size_1 * 2,
            hidden_size=hidden_size_2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        attention_dim = hidden_size_2 * 2
        self.attention = TemporalSelfAttention(attention_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(attention_dim, num_classes)

    def forward(self, x, return_attention=False):
        x, _ = self.bilstm1(x)
        x, _ = self.bilstm2(x)
        context, attention = self.attention(x)
        logits = self.classifier(self.dropout(context))

        if return_attention:
            return logits, attention
        return logits
