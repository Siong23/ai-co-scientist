import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

from app.dl.data_loader import load_csv
from app.dl.preprocessing import Preprocessor, make_binary_target
from app.dl.datasets import make_sequences
from app.dl.model_builder import build_model
from app.dl.evaluator import evaluate, save_json
from app.experiments.experiment_config import ExperimentConfig


def load_trained_model(
    model_name,
    checkpoint_path,
    input_size,
    num_classes,
    cfg,
    device
):
    """
    Rebuild the model architecture and load the saved PyTorch weights.
    """

    # Build the same model architecture
    model = build_model(
        model_name,
        input_size,
        num_classes,
        cfg
    )

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    # Load saved model weights
    state_dict = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(state_dict)

    # Move model to device
    model.to(device)

    # Set evaluation mode
    model.eval()

    print(f"Loaded model: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")

    return model


def prepare_test_data(cfg):
    """
    Load the dataset and reproduce the same preprocessing pipeline.
    """

    # Load dataset
    df = load_csv(cfg.data_path)

    print(f"Loaded dataset: {df.shape}")

    # Convert to binary classification if required
    if cfg.binary:
        df = make_binary_target(
            df,
            cfg.target
        )

    # Reproduce the same 70/30 split
    train_df, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=df[cfg.target].astype(str)
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Fit preprocessing using training data
    pre = Preprocessor()

    pre.fit(
        train_df,
        target=cfg.target
    )

    # Transform test data
    test_features = pre.transform_features(
        test_df
    )

    test_labels = pre.transform_labels(
        test_df,
        cfg.target
    )

    # Create temporal sequences
    test_x, test_y = make_sequences(
        test_features,
        test_labels,
        cfg.sequence_length,
        cfg.sequence_stride
    )

    # Convert tensors to NumPy arrays
    test_x = test_x.numpy()
    test_y = test_y.numpy()

    class_names = pre.classes

    print(f"Test features: {test_x.shape[-1]}")
    print(f"Sequence length: {test_x.shape[1]}")
    print(f"Classes: {class_names}")
    print(f"Test sequences: {len(test_y)}")

    return (
        test_x,
        test_y,
        class_names
    )


def main(cfg, model_name, checkpoint_path):
    """
    Load a trained model and evaluate it.
    """

    # Resolve device
    device = cfg.resolve_device()

    print(f"Device: {device}")

    # Prepare test dataset
    test_x, test_y, class_names = prepare_test_data(
        cfg
    )

    num_features = test_x.shape[-1]
    num_classes = len(class_names)

    # Create test DataLoader
    test_dataset = TensorDataset(
        torch.tensor(
            test_x,
            dtype=torch.float32
        ),
        torch.tensor(
            test_y,
            dtype=torch.long
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False
    )

    # Load trained model
    model = load_trained_model(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        input_size=num_features,
        num_classes=num_classes,
        cfg=cfg,
        device=device
    )

    # Evaluate loaded model
    print("\nEvaluating loaded model...")

    result = evaluate(
        model,
        test_loader,
        class_names,
        device
    )

    # Add model information
    result["model"] = model_name
    result["checkpoint"] = str(checkpoint_path)

    # Count parameters
    result["num_parameters"] = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    # Save evaluation result
    output_path = (
        Path(cfg.output_dir)
        / "loaded_model_metrics"
        / f"{model_name}_loaded.json"
    )

    save_json(
        result,
        output_path
    )

    # Print results
    print("\n" + "=" * 70)
    print("LOADED MODEL EVALUATION RESULTS")
    print("=" * 70)

    print(f"Model: {model_name}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(
        f"Precision: "
        f"{result['precision_weighted']:.4f}"
    )
    print(
        f"Recall: "
        f"{result['recall_weighted']:.4f}"
    )
    print(
        f"F1-score: "
        f"{result['f1_weighted']:.4f}"
    )

    print(
        f"\nResults saved to: {output_path}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Load and evaluate a trained PyTorch model."
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "lstm",
            "lstm_attention",
            "bilstm",
            "bitad"
        ],
        help="Model architecture to load."
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the saved .pt model file."
    )

    parser.add_argument(
        "--data",
        default=None,
        help="Path to the dataset."
    )

    parser.add_argument(
        "--binary",
        action="store_true",
        help="Use binary classification."
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None
    )

    args = parser.parse_args()

    # Load experiment configuration
    cfg = ExperimentConfig()

    # Override configuration from command line
    if args.data:
        cfg.data_path = args.data

    if args.binary:
        cfg.binary = True

    if args.sequence_length:
        cfg.sequence_length = (
            args.sequence_length
        )

    # Run loaded model evaluation
    main(
        cfg=cfg,
        model_name=args.model,
        checkpoint_path=args.checkpoint
    )