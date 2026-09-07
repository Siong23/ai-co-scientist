from dataclasses import dataclass
from pathlib import Path
import copy
import time

import torch
from torch import nn


@dataclass
class TrainingHistory:
    train_loss: list
    val_loss: list
    train_accuracy: list
    val_accuracy: list
    epochs: int
    training_seconds: float


def _run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    train=True
):
    model.train(train)

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)

        loss = criterion(logits, y)

        if train:
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

        total_loss += loss.item() * len(y)

        correct += (
            logits.argmax(dim=1) == y
        ).sum().item()

        total += len(y)

    return (
        total_loss / total,
        correct / total
    )


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=10,
    lr=1e-3,
    class_weights=None,
    device="cpu",
    patience=3,
    checkpoint_path=None,
    model_name=None,
    model_config=None,
    class_names=None
):

    model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=(
            class_weights.to(device)
            if class_weights is not None
            else None
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    history = TrainingHistory(
        train_loss=[],
        val_loss=[],
        train_accuracy=[],
        val_accuracy=[],
        epochs=0,
        training_seconds=0.0
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    start = time.perf_counter()

    for epoch in range(1, epochs + 1):

        # =========================
        # Training
        # =========================

        train_loss, train_acc = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True
        )

        # =========================
        # Validation
        # =========================

        val_loss, val_acc = _run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False
        )

        # =========================
        # Save training history
        # =========================

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_accuracy.append(train_acc)
        history.val_accuracy.append(val_acc)

        history.epochs = epoch

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss={train_loss:.4f} "
            f"acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} "
            f"acc={val_acc:.4f}"
        )

        # =========================
        # Check best model
        # =========================

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            bad_epochs = 0

            print(
                f"New best model found "
                f"(validation loss: {best_val_loss:.4f})"
            )

        else:

            bad_epochs += 1

            if bad_epochs >= patience:

                print(
                    f"Early stopping at epoch {epoch}."
                )

                break

    # =========================
    # Calculate training time
    # =========================

    history.training_seconds = (
        time.perf_counter() - start
    )

    # =========================
    # Restore best model
    # =========================

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    # =========================
    # Save final BEST checkpoint
    # =========================

    if checkpoint_path and best_state is not None:

        checkpoint_path = Path(
            checkpoint_path
        )

        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Metrics corresponding to BEST epoch
        best_train_loss = history.train_loss[
            best_epoch - 1
        ]

        best_train_accuracy = history.train_accuracy[
            best_epoch - 1
        ]

        best_val_loss_value = history.val_loss[
            best_epoch - 1
        ]

        best_val_accuracy = history.val_accuracy[
            best_epoch - 1
        ]

        checkpoint = {

            # =========================
            # Model information
            # =========================

            "model_name": model_name,

            "model_config": model_config,

            "class_names": class_names,

            # =========================
            # Epoch information
            # =========================

            "best_epoch": best_epoch,

            "epochs_trained": history.epochs,

            "epochs_requested": epochs,

            # =========================
            # Model weights
            # =========================

            "model_state_dict": copy.deepcopy(
                model.state_dict()
            ),

            # =========================
            # Optimizer
            # =========================

            "optimizer_state_dict":
                optimizer.state_dict(),

            # =========================
            # Best epoch metrics
            # =========================

            "train_loss": best_train_loss,

            "train_accuracy": best_train_accuracy,

            "val_loss": best_val_loss_value,

            "val_accuracy": best_val_accuracy,

            "best_val_loss": best_val_loss,

            # =========================
            # Complete training history
            # =========================

            "history": {

                "train_loss":
                    history.train_loss.copy(),

                "val_loss":
                    history.val_loss.copy(),

                "train_accuracy":
                    history.train_accuracy.copy(),

                "val_accuracy":
                    history.val_accuracy.copy(),

            },

            # =========================
            # Training information
            # =========================

            "training_seconds":
                history.training_seconds,

            "learning_rate":
                lr,

            "patience":
                patience,

            "device":
                str(device),
        }

        torch.save(
            checkpoint,
            checkpoint_path
        )

        print(
            f"Best checkpoint saved: "
            f"{checkpoint_path}"
        )

        print(
            f"Best epoch: {best_epoch}"
        )

        print(
            f"Training time: "
            f"{history.training_seconds:.2f} seconds"
        )

    return model, history