"""
Optional explainability utilities.

BiTAD in the reference paper uses TwinLens: SHAP for feature-level
attribution and TimeSHAP for temporal relevance. This file keeps the
experiment pipeline independent of optional SHAP dependencies.

Install optional dependencies when needed:
    pip install shap
"""
import torch

def get_temporal_attention(model, x, device="cpu"):
    """Return attention weights for models that expose them."""
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        output = model(x, return_attention=True)
    if isinstance(output, tuple):
        logits, attention = output
        return logits, attention
    return output, None

def aggregate_attention(attention):
    """
    Convert self-attention [batch,time,time] into per-timestep importance.
    Returns [batch,time].
    """
    if attention is None:
        return None
    if attention.dim() == 3:
        return attention.mean(dim=1)
    return attention
