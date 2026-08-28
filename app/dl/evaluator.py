from pathlib import Path
import json
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

@torch.no_grad()
def predict(model, loader, device="cpu"):
    model.eval()
    y_true, y_pred, probabilities = [], [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).cpu().numpy()

        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred.tolist())
        probabilities.extend(probs.cpu().numpy().tolist())

    return np.asarray(y_true), np.asarray(y_pred), np.asarray(probabilities)

def evaluate(model, loader, class_names, device="cpu"):
    y_true, y_pred, probabilities = predict(model, loader, device)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_true, y_pred, target_names=class_names,
        output_dict=True, zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": report,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }

def save_json(result, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
