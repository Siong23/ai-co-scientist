'''`
Run with:

    Single model visualization (run each one of the model)
        python app/experiments/experiment_visualization.py --checkpoint app/experiments/results/checkpoints/lstm.pt --metrics app/experiments/results/metrics/lstm.json

        python app/experiments/experiment_visualization.py --checkpoint app/experiments/results/checkpoints/lstm_attention.pt --metrics app/experiments/results/metrics/lstm_attention.json

        python app/experiments/experiment_visualization.py --checkpoint app/experiments/results/checkpoints/bilstm.pt --metrics app/experiments/results/metrics/bilstm.json

        python app/experiments/experiment_visualization.py --checkpoint app/experiments/results/checkpoints/bitad.pt --metrics app/experiments/results/metrics/bitad.json
    
    All-model comparison
        python app/experiments/experiment_visualization.py --compare --checkpoints app/experiments/results/checkpoints/lstm.pt app/experiments/results/checkpoints/lstm_attention.pt app/experiments/results/checkpoints/bilstm.pt app/experiments/results/checkpoints/bitad.pt --metrics-files app/experiments/results/metrics/lstm.json app/experiments/results/metrics/lstm_attention.json app/experiments/results/metrics/bilstm.json app/experiments/results/metrics/bitad.json
'''
from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

# ============================================================
# Configuration
# ============================================================

DEFAULT_RESULTS_DIR = Path(
    "app/experiments/results"
)

# ============================================================
# Load checkpoint
# ============================================================

def load_checkpoint(checkpoint_path, device="cpu"):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    return checkpoint

# ============================================================
# Load evaluation metrics
# ============================================================

def load_metrics(metrics_path):
    metrics_path = Path(metrics_path)

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}"
        )

    with open(
        metrics_path,
        "r",
        encoding="utf-8"
    ) as f:

        metrics = json.load(f)

    return metrics

# ============================================================
# Create output directory
# ============================================================

def create_output_directory(output_dir):
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return output_dir

# ============================================================
# Plot training and validation loss
# ============================================================

def plot_loss(history, model_name, output_dir):
    train_loss = history.get(
        "train_loss",
        []
    )

    val_loss = history.get(
        "val_loss",
        []
    )

    if not train_loss or not val_loss:

        print(
            "Loss history not found. "
            "Skipping loss visualization."
        )

        return

    epochs = range(
        1,
        len(train_loss) + 1
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        train_loss,
        marker="o",
        label="Train Loss"
    )

    plt.plot(
        epochs,
        val_loss,
        marker="o",
        label="Validation Loss"
    )

    plt.title(
        f"{model_name} Training and Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    output_path = (
        output_dir
        / f"{model_name}_loss.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved loss plot: "
        f"{output_path}"
    )

# ============================================================
# Plot training and validation accuracy
# ============================================================

def plot_accuracy(history, model_name, output_dir):
    train_accuracy = history.get(
        "train_accuracy",
        []
    )

    val_accuracy = history.get(
        "val_accuracy",
        []
    )

    if not train_accuracy or not val_accuracy:

        print(
            "Accuracy history not found. "
            "Skipping accuracy visualization."
        )

        return

    epochs = range(
        1,
        len(train_accuracy) + 1
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        train_accuracy,
        marker="o",
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        val_accuracy,
        marker="o",
        label="Validation Accuracy"
    )

    plt.title(
        f"{model_name} Training and Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    output_path = (
        output_dir
        / f"{model_name}_accuracy.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved accuracy plot: "
        f"{output_path}"
    )

# ============================================================
# Plot combined training curves
# ============================================================

def plot_training_curves(
    history,
    model_name,
    output_dir
):
    train_loss = history.get(
        "train_loss",
        []
    )

    val_loss = history.get(
        "val_loss",
        []
    )

    train_accuracy = history.get(
        "train_accuracy",
        []
    )

    val_accuracy = history.get(
        "val_accuracy",
        []
    )

    if (
        not train_loss
        or not val_loss
        or not train_accuracy
        or not val_accuracy
    ):

        print(
            "Incomplete training history. "
            "Skipping combined visualization."
        )

        return

    epochs = range(
        1,
        len(train_loss) + 1
    )

    # -----------------------------
    # Loss
    # -----------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        train_loss,
        marker="o",
        label="Train Loss"
    )

    plt.plot(
        epochs,
        val_loss,
        marker="o",
        label="Validation Loss"
    )

    plt.title(
        f"{model_name} Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    output_path = (
        output_dir
        / f"{model_name}_training_loss.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


    # -----------------------------
    # Accuracy
    # -----------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        train_accuracy,
        marker="o",
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        val_accuracy,
        marker="o",
        label="Validation Accuracy"
    )

    plt.title(
        f"{model_name} Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    output_path = (
        output_dir
        / f"{model_name}_training_accuracy.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        "Saved training curve visualizations."
    )

# ============================================================
# Plot confusion matrix
# ============================================================

def plot_confusion_matrix(
    metrics,
    model_name,
    output_dir
):
    if (
        "confusion_matrix"
        not in metrics
    ):

        print(
            "Confusion matrix not found. "
            "Skipping confusion matrix visualization."
        )

        return

    cm = np.array(
        metrics["confusion_matrix"]
    )

    # Try to obtain class names
    class_names = None

    if (
        "classification_report"
        in metrics
    ):

        report = metrics[
            "classification_report"
        ]

        class_names = [
            key
            for key in report.keys()
            if key not in [
                "accuracy",
                "macro avg",
                "weighted avg"
            ]
        ]

    if (
        class_names is None
        or len(class_names) != len(cm)
    ):

        class_names = [
            f"Class {i}"
            for i in range(len(cm))
        ]

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        f"{model_name} Confusion Matrix"
    )

    plt.colorbar()

    tick_marks = np.arange(
        len(class_names)
    )

    plt.xticks(
        tick_marks,
        class_names,
        rotation=45
    )

    plt.yticks(
        tick_marks,
        class_names
    )

    threshold = (
        cm.max() / 2.0
        if cm.size > 0
        else 0
    )

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                verticalalignment="center"
            )

    plt.ylabel(
        "True Label"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.tight_layout()

    output_path = (
        output_dir
        / f"{model_name}_confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved confusion matrix: "
        f"{output_path}"
    )

# ============================================================
# Plot performance metrics
# ============================================================

def plot_performance_metrics(
    metrics,
    model_name,
    output_dir
):
    metric_names = [
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted"
    ]

    values = [
        metrics.get(name, 0.0)
        for name in metric_names
    ]

    display_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        display_names,
        values
    )

    plt.title(
        f"{model_name} Test Performance Metrics"
    )

    plt.xlabel(
        "Metric"
    )

    plt.ylabel(
        "Score"
    )

    plt.ylim(
        0,
        1
    )

    for i, value in enumerate(values):

        plt.text(
            i,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    output_path = (
        output_dir
        / f"{model_name}_performance_metrics.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved performance metrics plot: "
        f"{output_path}"
    )

# ============================================================
# Compare validation loss for all models
# ============================================================

def plot_all_models_validation_loss(
    checkpoints,
    output_dir
):

    if not checkpoints:

        print(
            "No checkpoints provided."
        )

        return

    plt.figure(
        figsize=(10, 6)
    )

    plotted = False

    for model_name, checkpoint in checkpoints.items():

        history = checkpoint.get(
            "history",
            {}
        )

        val_loss = history.get(
            "val_loss",
            []
        )

        if not val_loss:

            print(
                f"No validation loss found "
                f"for {model_name}."
            )

            continue

        epochs = range(
            1,
            len(val_loss) + 1
        )

        plt.plot(
            epochs,
            val_loss,
            marker="o",
            label=model_name
        )

        plotted = True

    if not plotted:

        plt.close()

        print(
            "No validation loss data available."
        )

        return

    plt.title(
        "Validation Loss Comparison"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Validation Loss"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    output_path = (
        Path(output_dir)
        / "all_models_validation_loss.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved comparison: "
        f"{output_path}"
    )

# ============================================================
# Compare validation accuracy for all models
# ============================================================

def plot_all_models_validation_accuracy(
    checkpoints,
    output_dir
):

    if not checkpoints:

        print(
            "No checkpoints provided."
        )

        return

    plt.figure(
        figsize=(10, 6)
    )

    plotted = False

    for model_name, checkpoint in checkpoints.items():

        history = checkpoint.get(
            "history",
            {}
        )

        val_accuracy = history.get(
            "val_accuracy",
            []
        )

        if not val_accuracy:

            print(
                f"No validation accuracy found "
                f"for {model_name}."
            )

            continue

        epochs = range(
            1,
            len(val_accuracy) + 1
        )

        plt.plot(
            epochs,
            val_accuracy,
            marker="o",
            label=model_name
        )

        plotted = True

    if not plotted:

        plt.close()

        print(
            "No validation accuracy data available."
        )

        return

    plt.title(
        "Validation Accuracy Comparison"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Validation Accuracy"
    )

    plt.ylim(
        0,
        1
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    output_path = (
        Path(output_dir)
        / "all_models_validation_accuracy.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved comparison: "
        f"{output_path}"
    )

# ============================================================
# Compare training loss for all models
# ============================================================

def plot_all_models_training_loss(
    checkpoints,
    output_dir
):

    plt.figure(
        figsize=(10, 6)
    )

    plotted = False

    for model_name, checkpoint in checkpoints.items():

        history = checkpoint.get(
            "history",
            {}
        )

        train_loss = history.get(
            "train_loss",
            []
        )

        if not train_loss:

            continue

        epochs = range(
            1,
            len(train_loss) + 1
        )

        plt.plot(
            epochs,
            train_loss,
            marker="o",
            label=model_name
        )

        plotted = True

    if not plotted:

        plt.close()

        return

    plt.title(
        "Training Loss Comparison"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Training Loss"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    output_path = (
        Path(output_dir)
        / "all_models_training_loss.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved comparison: "
        f"{output_path}"
    )

# ============================================================
# Compare training accuracy for all models
# ============================================================

def plot_all_models_training_accuracy(
    checkpoints,
    output_dir
):

    plt.figure(
        figsize=(10, 6)
    )

    plotted = False

    for model_name, checkpoint in checkpoints.items():

        history = checkpoint.get(
            "history",
            {}
        )

        train_accuracy = history.get(
            "train_accuracy",
            []
        )

        if not train_accuracy:

            continue

        epochs = range(
            1,
            len(train_accuracy) + 1
        )

        plt.plot(
            epochs,
            train_accuracy,
            marker="o",
            label=model_name
        )

        plotted = True

    if not plotted:

        plt.close()

        return

    plt.title(
        "Training Accuracy Comparison"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Training Accuracy"
    )

    plt.ylim(
        0,
        1
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    output_path = (
        Path(output_dir)
        / "all_models_training_accuracy.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved comparison: "
        f"{output_path}"
    )

# ============================================================
# Create one function to run all training comparisons
# ============================================================

def visualize_all_models(
    checkpoint_paths,
    output_dir=None,
    device="cpu"
):

    checkpoints = load_multiple_checkpoints(
        checkpoint_paths,
        device
    )

    if output_dir is None:

        output_dir = (
            DEFAULT_RESULTS_DIR
            / "visualizations"
            / "model_comparison"
        )

    output_dir = create_output_directory(
        output_dir
    )

    print(
        "\nGenerating model comparison "
        "visualizations..."
    )

    plot_all_models_training_loss(
        checkpoints,
        output_dir
    )

    plot_all_models_validation_loss(
        checkpoints,
        output_dir
    )

    plot_all_models_training_accuracy(
        checkpoints,
        output_dir
    )

    plot_all_models_validation_accuracy(
        checkpoints,
        output_dir
    )

    print(
        "\nAll model comparison "
        "visualizations saved in:"
    )

    print(
        output_dir
    )

# ============================================================
# Compare final performance metrics across all models
# ============================================================

def plot_all_models_performance(
    metrics_paths,
    output_dir
):

    model_names = []

    accuracy_values = []
    precision_values = []
    recall_values = []
    f1_values = []

    for metrics_path in metrics_paths:

        metrics = load_metrics(
            metrics_path
        )

        model_name = metrics.get(
            "model",
            Path(metrics_path).stem
        )

        model_names.append(
            model_name
        )

        accuracy_values.append(
            metrics.get(
                "accuracy",
                0.0
            )
        )

        precision_values.append(
            metrics.get(
                "precision_weighted",
                0.0
            )
        )

        recall_values.append(
            metrics.get(
                "recall_weighted",
                0.0
            )
        )

        f1_values.append(
            metrics.get(
                "f1_weighted",
                0.0
            )
        )

    if not model_names:

        print(
            "No metrics available."
        )

        return

    x = np.arange(
        len(model_names)
    )

    width = 0.2

    plt.figure(
        figsize=(12, 7)
    )

    plt.bar(
        x - 1.5 * width,
        accuracy_values,
        width,
        label="Accuracy"
    )

    plt.bar(
        x - 0.5 * width,
        precision_values,
        width,
        label="Precision"
    )

    plt.bar(
        x + 0.5 * width,
        recall_values,
        width,
        label="Recall"
    )

    plt.bar(
        x + 1.5 * width,
        f1_values,
        width,
        label="F1 Score"
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Test Performance Comparison "
        "Across Models"
    )

    plt.xticks(
        x,
        model_names,
        rotation=20
    )

    plt.ylim(
        0,
        1
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        Path(output_dir)
        / "all_models_performance_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved performance comparison: "
        f"{output_path}"
    )

# ============================================================
# Compare execution time
# ============================================================

def plot_all_models_execution_time(
    metrics_paths,
    output_dir
):

    model_names = []

    training_times = []
    evaluation_times = []
    total_times = []

    for metrics_path in metrics_paths:

        metrics = load_metrics(
            metrics_path
        )

        model_name = metrics.get(
            "model",
            Path(metrics_path).stem
        )

        model_names.append(
            model_name
        )

        training_times.append(
            metrics.get(
                "training_seconds",
                0.0
            )
        )

        evaluation_times.append(
            metrics.get(
                "evaluation_seconds",
                0.0
            )
        )

        total_times.append(
            metrics.get(
                "total_execution_seconds",
                0.0
            )
        )

    x = np.arange(
        len(model_names)
    )

    width = 0.25

    plt.figure(
        figsize=(12, 7)
    )

    plt.bar(
        x - width,
        training_times,
        width,
        label="Training"
    )

    plt.bar(
        x,
        evaluation_times,
        width,
        label="Evaluation"
    )

    plt.bar(
        x + width,
        total_times,
        width,
        label="Total Execution"
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Time (seconds)"
    )

    plt.title(
        "Execution Time Comparison"
    )

    plt.xticks(
        x,
        model_names,
        rotation=20
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        Path(output_dir)
        / "all_models_execution_time.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved execution time comparison: "
        f"{output_path}"
    )

# ============================================================
# Load multiple checkpoint
# ============================================================

def load_multiple_checkpoints(checkpoint_paths, device="cpu"):

    checkpoints = {}

    for checkpoint_path in checkpoint_paths:

        checkpoint_path = Path(checkpoint_path)

        checkpoint = load_checkpoint(
            checkpoint_path,
            device
        )

        model_name = checkpoint.get(
            "model_name",
            checkpoint_path.stem
        )

        checkpoints[model_name] = checkpoint

    return checkpoints

# ============================================================
# Print checkpoint information
# ============================================================

def print_checkpoint_summary(checkpoint):
    print("\n" + "=" * 60)

    print(
        "CHECKPOINT INFORMATION"
    )

    print("=" * 60)

    print(
        f"Model Name: "
        f"{checkpoint.get('model_name')}"
    )

    print(
        f"Best Epoch: "
        f"{checkpoint.get('best_epoch')}"
    )

    print(
        f"Epochs Trained: "
        f"{checkpoint.get('epochs_trained')}"
    )

    print(
        f"Best Train Loss: "
        f"{checkpoint.get('train_loss')}"
    )

    print(
        f"Best Train Accuracy: "
        f"{checkpoint.get('train_accuracy')}"
    )

    print(
        f"Best Validation Loss: "
        f"{checkpoint.get('val_loss')}"
    )

    print(
        f"Best Validation Accuracy: "
        f"{checkpoint.get('val_accuracy')}"
    )

    print(
        f"Training Time: "
        f"{checkpoint.get('training_seconds')} seconds"
    )

    print("=" * 60)

# ============================================================
# Main visualization function
# ============================================================

def visualize_model(
    checkpoint_path,
    metrics_path=None,
    output_dir=None,
    device="cpu"
):
    checkpoint_path = Path(
        checkpoint_path
    )

    # Load checkpoint
    checkpoint = load_checkpoint(
        checkpoint_path,
        device
    )

    model_name = checkpoint.get(
        "model_name",
        checkpoint_path.stem
    )

    # Get training history
    history = checkpoint.get(
        "history",
        {}
    )

    # Output directory
    if output_dir is None:

        output_dir = (
            DEFAULT_RESULTS_DIR
            / "visualizations"
            / model_name
        )

    output_dir = create_output_directory(
        output_dir
    )

    # Print checkpoint summary
    print_checkpoint_summary(
        checkpoint
    )

    print(
        f"\nGenerating visualizations "
        f"for {model_name}..."
    )

    # Training visualizations
    plot_loss(
        history,
        model_name,
        output_dir
    )

    plot_accuracy(
        history,
        model_name,
        output_dir
    )

    # Evaluation visualizations
    if metrics_path:

        metrics = load_metrics(
            metrics_path
        )

        plot_confusion_matrix(
            metrics,
            model_name,
            output_dir
        )

        plot_performance_metrics(
            metrics,
            model_name,
            output_dir
        )

    else:

        print(
            "\nNo metrics JSON provided."
        )

        print(
            "Only training and validation "
            "visualizations will be generated."
        )

    print(
        f"\nAll visualizations saved in:"
    )

    print(
        output_dir
    )

# ============================================================
# Command-line execution
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Visualize training and evaluation "
            "results for one or multiple models."
        )
    )

    # --------------------------------------------------------
    # Mode
    # --------------------------------------------------------

    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Compare multiple models."
        )
    )

    # --------------------------------------------------------
    # Single model arguments
    # --------------------------------------------------------

    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to a single PyTorch checkpoint."
        )
    )

    parser.add_argument(
        "--metrics",
        default=None,
        help=(
            "Path to a single evaluation metrics JSON file."
        )
    )

    # --------------------------------------------------------
    # Multiple model arguments
    # --------------------------------------------------------

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help=(
            "Paths to multiple PyTorch checkpoints "
            "for model comparison."
        )
    )

    parser.add_argument(
        "--metrics-files",
        nargs="+",
        default=None,
        help=(
            "Paths to multiple evaluation metrics "
            "JSON files for model comparison."
        )
    )

    # --------------------------------------------------------
    # General arguments
    # --------------------------------------------------------

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory where visualizations "
            "will be saved."
        )
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "Device used to load checkpoints."
        )
    )

    args = parser.parse_args()

    # ========================================================
    # Compare multiple models
    # ========================================================

    if args.compare:

        if not args.checkpoints:

            parser.error(
                "--checkpoints is required "
                "when using --compare."
            )

        # Create output directory

        if args.output_dir is None:

            output_dir = (
                DEFAULT_RESULTS_DIR
                / "visualizations"
                / "model_comparison"
            )

        else:

            output_dir = Path(
                args.output_dir
            )

        output_dir = create_output_directory(
            output_dir
        )

        # ----------------------------------------------------
        # Training comparisons
        # ----------------------------------------------------

        visualize_all_models(

            checkpoint_paths=args.checkpoints,

            output_dir=output_dir,

            device=args.device
        )

        # ----------------------------------------------------
        # Test performance comparison
        # ----------------------------------------------------

        if args.metrics_files:

            plot_all_models_performance(

                metrics_paths=args.metrics_files,

                output_dir=output_dir
            )

            # ------------------------------------------------
            # Execution time comparison
            # ------------------------------------------------

            plot_all_models_execution_time(

                metrics_paths=args.metrics_files,

                output_dir=output_dir
            )

        else:

            print(
                "\nNo metrics files provided."
            )

            print(
                "Only training and validation "
                "comparisons were generated."
            )

    # ========================================================
    # Visualize single model
    # ========================================================

    else:

        if not args.checkpoint:

            parser.error(
                "--checkpoint is required "
                "when not using --compare."
            )

        visualize_model(

            checkpoint_path=args.checkpoint,

            metrics_path=args.metrics,

            output_dir=args.output_dir,

            device=args.device
        )