from .models import (
    LSTMClassifier,
    LSTMAttentionClassifier,
    BiLSTMClassifier,
    BiTAD,
)

def build_model(name, input_size, num_classes, config):
    name = name.lower().replace("-", "_").replace(" ", "_")

    if name == "lstm":
        return LSTMClassifier(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            num_classes=num_classes,
        )

    if name in {"lstm_attention", "lstm_attention_classifier"}:
        return LSTMAttentionClassifier(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            num_classes=num_classes,
        )

    if name == "bilstm":
        return BiLSTMClassifier(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            num_classes=num_classes,
        )

    if name == "bitad":
        return BiTAD(
            input_size=input_size,
            hidden_size_1=config.bitad_hidden_size_1,
            hidden_size_2=config.bitad_hidden_size_2,
            dropout=config.dropout,
            num_classes=num_classes,
        )

    raise ValueError(f"Unknown model: {name}")
