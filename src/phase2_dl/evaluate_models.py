import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report


# ---------------------------------------------------------
# GeoSense Phase II - Exercise 4.2
# Evaluate U-Net vs ResNet-50 fine-tuned substitute
# ---------------------------------------------------------

CHIP_DIR = Path("data/satellite/chips")
LABEL_DIR = Path("data/satellite/labels")

MODEL_DIR = Path("models/saved")
EVALUATION_DIR = Path("models/evaluation")
PLOT_DIR = Path("outputs/plots")

EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


YEARS = [2015, 2023]

CHIP_SIZE = 224
STRIDE = 174

IN_CHANNELS = 6
NUM_CLASSES = 5

BATCH_SIZE = 4
RANDOM_SEED = 42

CLASS_NAMES = [
    "Urban",
    "Vegetation",
    "Water",
    "Bare Land",
    "Agriculture"
]


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class EvaluationDataset(Dataset):

    def __init__(self, samples):

        self.samples = samples

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        chip_path, label_path, row, col = (
            self.samples[index]
        )

        image = np.load(
            chip_path
        ).astype(np.float32)

        label_mask = np.load(
            label_path
        )

        mask = label_mask[
            row:row + CHIP_SIZE,
            col:col + CHIP_SIZE
        ].astype(np.int64)

        image_tensor = torch.from_numpy(
            image.copy()
        ).float()

        mask_tensor = torch.from_numpy(
            mask.copy()
        ).long()

        return image_tensor, mask_tensor


# ---------------------------------------------------------
# Build chip-label pairs
# ---------------------------------------------------------

def build_samples():

    samples = []

    for year in YEARS:

        chip_dir = CHIP_DIR / str(year)
        label_file = LABEL_DIR / f"labels_{year}.npy"

        chip_files = sorted(
            chip_dir.glob("chip_*.npy")
        )

        if len(chip_files) == 0:

            raise RuntimeError(
                f"No chips found in {chip_dir}"
            )

        label_mask = np.load(
            label_file
        )

        height, width = label_mask.shape

        positions = []

        for row in range(
            0,
            height - CHIP_SIZE,
            STRIDE
        ):

            for col in range(
                0,
                width - CHIP_SIZE,
                STRIDE
            ):

                positions.append(
                    (row, col)
                )

        if len(chip_files) != len(positions):

            raise RuntimeError(
                f"{year}: "
                f"{len(chip_files)} chips but "
                f"{len(positions)} positions."
            )

        for chip_path, (row, col) in zip(
            chip_files,
            positions
        ):

            samples.append(
                (
                    chip_path,
                    label_file,
                    row,
                    col
                )
            )

    return samples


# ---------------------------------------------------------
# Recreate the same 80/20 validation split
# ---------------------------------------------------------

def get_validation_samples(samples):

    dataset = EvaluationDataset(
        samples
    )

    train_size = int(
        0.8 * len(dataset)
    )

    val_size = (
        len(dataset) - train_size
    )

    _, validation_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(
            RANDOM_SEED
        )
    )

    return validation_dataset


# ---------------------------------------------------------
# Build models
# ---------------------------------------------------------

def build_unet():

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES
    )

    return model


def build_resnet50():

    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES
    )

    return model


# ---------------------------------------------------------
# Load model weights
# ---------------------------------------------------------

def load_model(
    model,
    checkpoint_path,
    device
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint
    )

    model = model.to(device)
    model.eval()

    return model


# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------

def collect_predictions(
    model,
    loader,
    device
):

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(device)

            outputs = model(
                images
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.append(
                predictions.cpu().numpy()
            )

            all_targets.append(
                masks.numpy()
            )

    predictions = np.concatenate(
        all_predictions
    ).reshape(-1)

    targets = np.concatenate(
        all_targets
    ).reshape(-1)

    return predictions, targets


# ---------------------------------------------------------
# IoU
# ---------------------------------------------------------

def compute_iou(
    predictions,
    targets,
    num_classes
):

    ious = []

    for class_id in range(
        num_classes
    ):

        prediction_mask = (
            predictions == class_id
        )

        target_mask = (
            targets == class_id
        )

        intersection = np.logical_and(
            prediction_mask,
            target_mask
        ).sum()

        union = np.logical_or(
            prediction_mask,
            target_mask
        ).sum()

        if union == 0:

            iou = float("nan")

        else:

            iou = (
                intersection / union
            )

        ious.append(iou)

    valid_ious = [
        value
        for value in ious
        if np.isfinite(value)
    ]

    if valid_ious:

        mean_iou = float(
            np.mean(valid_ious)
        )

    else:

        mean_iou = float("nan")

    return ious, mean_iou


# ---------------------------------------------------------
# Accuracy
# ---------------------------------------------------------

def compute_accuracy(
    predictions,
    targets
):

    return float(
        np.mean(
            predictions == targets
        )
    )


# ---------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------

def plot_confusion_matrix(
    targets,
    predictions,
    title,
    output_file
):

    cm = confusion_matrix(
        targets,
        predictions,
        labels=list(range(NUM_CLASSES))
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(
        NUM_CLASSES
    )

    plt.xticks(
        tick_marks,
        CLASS_NAMES,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        tick_marks,
        CLASS_NAMES
    )

    threshold = (
        cm.max() / 2
        if cm.size > 0
        else 0
    )

    for i in range(
        NUM_CLASSES
    ):

        for j in range(
            NUM_CLASSES
        ):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                )
            )

    plt.ylabel(
        "True Class"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return cm


# ---------------------------------------------------------
# Evaluation of one model
# ---------------------------------------------------------

def evaluate_model(
    name,
    model,
    checkpoint,
    validation_loader,
    device
):

    print("\n" + "=" * 60)
    print(f"Evaluating: {name}")
    print("=" * 60)

    print(
        "Checkpoint:",
        checkpoint
    )

    model = load_model(
        model,
        checkpoint,
        device
    )

    predictions, targets = (
        collect_predictions(
            model,
            validation_loader,
            device
        )
    )

    ious, mean_iou = compute_iou(
        predictions,
        targets,
        NUM_CLASSES
    )

    accuracy = compute_accuracy(
        predictions,
        targets
    )

    print(
        "Validation pixel accuracy:",
        f"{accuracy * 100:.2f}%"
    )

    print(
        "\nPer-class IoU:"
    )

    for class_name, iou in zip(
        CLASS_NAMES,
        ious
    ):

        if np.isfinite(iou):

            print(
                f"{class_name}: "
                f"{iou:.4f}"
            )

        else:

            print(
                f"{class_name}: N/A"
            )

    print(
        "\nMean IoU:",
        f"{mean_iou:.4f}"
    )

    report = classification_report(
        targets,
        predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0
    )

    print(
        "\nClassification report:"
    )

    print(report)

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    confusion_file = (
        PLOT_DIR /
        f"confusion_matrix_{safe_name}.png"
    )

    plot_confusion_matrix(
        targets,
        predictions,
        f"{name} Confusion Matrix",
        confusion_file
    )

    return {
        "accuracy": accuracy,
        "iou": {
            class_name: (
                None
                if not np.isfinite(iou)
                else float(iou)
            )
            for class_name, iou in zip(
                CLASS_NAMES,
                ious
            )
        },
        "mean_iou": mean_iou,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            targets,
            predictions,
            labels=list(range(NUM_CLASSES))
        ).tolist(),
        "confusion_matrix_file": str(
            confusion_file
        )
    }


# ---------------------------------------------------------
# Training curve comparison
# ---------------------------------------------------------

def load_history(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def plot_training_curves(
    unet_history,
    resnet_history
):

    epochs_unet = range(
        1,
        len(unet_history["train_loss"]) + 1
    )

    epochs_resnet = range(
        1,
        len(resnet_history["train_loss"]) + 1
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        epochs_unet,
        unet_history["train_loss"],
        marker="o",
        label="ResNet-34 U-Net Training Loss"
    )

    plt.plot(
        epochs_unet,
        unet_history["val_loss"],
        marker="o",
        label="ResNet-34 U-Net Validation Loss"
    )

    plt.plot(
        epochs_resnet,
        resnet_history["train_loss"],
        marker="s",
        label="ResNet-50 Substitute Training Loss"
    )

    plt.plot(
        epochs_resnet,
        resnet_history["val_loss"],
        marker="s",
        label="ResNet-50 Substitute Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Training and Validation Loss Comparison"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output_file = (
        PLOT_DIR /
        "training_curves_comparison.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_file


def plot_accuracy_comparison(
    unet_history,
    resnet_history
):

    epochs_unet = range(
        1,
        len(unet_history["val_acc"]) + 1
    )

    epochs_resnet = range(
        1,
        len(resnet_history["val_acc"]) + 1
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        epochs_unet,
        [
            value * 100
            for value in unet_history["val_acc"]
        ],
        marker="o",
        label="ResNet-34 U-Net"
    )

    plt.plot(
        epochs_resnet,
        [
            value * 100
            for value in resnet_history["val_acc"]
        ],
        marker="s",
        label="ResNet-50 Substitute"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Validation Accuracy (%)"
    )

    plt.title(
        "Validation Accuracy Comparison"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output_file = (
        PLOT_DIR /
        "validation_accuracy_comparison.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_file


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "GeoSense Phase II - Model Evaluation"
    )

    print(
        "Evaluation device:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        print(
            "GPU: CPU execution"
        )


    # -----------------------------------------------------
    # Build and split data
    # -----------------------------------------------------

    samples = build_samples()

    print(
        "\nTotal samples:",
        len(samples)
    )

    validation_samples = (
        get_validation_samples(
            samples
        )
    )

    print(
        "Validation samples:",
        len(validation_samples)
    )

    validation_loader = DataLoader(
        validation_samples,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


    # -----------------------------------------------------
    # Load both models
    # -----------------------------------------------------

    unet_model = build_unet()

    resnet50_model = build_resnet50()


    # -----------------------------------------------------
    # Evaluate U-Net
    # -----------------------------------------------------

    unet_results = evaluate_model(
        "ResNet-34 U-Net",
        unet_model,
        MODEL_DIR / "unet_best.pth",
        validation_loader,
        device
    )


    # -----------------------------------------------------
    # Evaluate ResNet-50 substitute
    # -----------------------------------------------------

    resnet50_results = evaluate_model(
        "ResNet-50 Substitute",
        resnet50_model,
        MODEL_DIR / "prithvi_finetuned.pth",
        validation_loader,
        device
    )


    # -----------------------------------------------------
    # Load training histories
    # -----------------------------------------------------

    unet_history = load_history(
        EVALUATION_DIR / "unet_history.json"
    )

    resnet_history = load_history(
        EVALUATION_DIR / "prithvi_history.json"
    )


    # -----------------------------------------------------
    # Training plots
    # -----------------------------------------------------

    loss_plot = plot_training_curves(
        unet_history,
        resnet_history
    )

    accuracy_plot = plot_accuracy_comparison(
        unet_history,
        resnet_history
    )


    # -----------------------------------------------------
    # Determine better model
    # -----------------------------------------------------

    unet_miou = unet_results["mean_iou"]
    resnet_miou = resnet50_results["mean_iou"]

    if (
        np.isfinite(unet_miou)
        and np.isfinite(resnet_miou)
    ):

        if unet_miou > resnet_miou:

            better_model = (
                "ResNet-34 U-Net"
            )

        elif resnet_miou > unet_miou:

            better_model = (
                "ResNet-50 Substitute"
            )

        else:

            better_model = "Equal mean IoU"

    else:

        better_model = (
            "Mean IoU comparison unavailable"
        )


    # -----------------------------------------------------
    # Save metrics
    # -----------------------------------------------------

    evaluation = {
        "device": str(device),
        "validation_samples": len(
            validation_samples
        ),
        "models": {
            "ResNet-34 U-Net": unet_results,
            "ResNet-50 Substitute": resnet50_results
        },
        "better_model_by_mean_iou": better_model,
        "plots": {
            "loss_comparison": str(
                loss_plot
            ),
            "accuracy_comparison": str(
                accuracy_plot
            )
        }
    }

    metrics_file = (
        EVALUATION_DIR /
        "evaluation_metrics.json"
    )

    with open(
        metrics_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            evaluation,
            f,
            indent=4
        )


    # -----------------------------------------------------
    # Written comparison
    # -----------------------------------------------------

    comparison_text = f"""
GeoSense Phase II - Model Comparison
====================================

Models evaluated:
1. ResNet-34 U-Net trained from scratch
2. ResNet-50 substitute fine-tuned using pretrained ImageNet weights

Validation samples:
{len(validation_samples)}

ResNet-34 U-Net
--------------
Validation pixel accuracy:
{unet_results["accuracy"] * 100:.2f}%

Mean IoU:
{unet_results["mean_iou"]:.4f}


ResNet-50 Substitute
--------------------
Validation pixel accuracy:
{resnet50_results["accuracy"] * 100:.2f}%

Mean IoU:
{resnet50_results["mean_iou"]:.4f}


Model with higher mean IoU:
{better_model}

Interpretation:
The comparison indicates which model produced better segmentation
performance on the same held-out validation samples.

A fine-tuned pretrained model can potentially perform better than
a model trained from scratch when training data are limited because
pretrained weights provide previously learned visual features.
Fine-tuning can therefore start from a stronger representation
rather than learning all useful features from randomly initialized
weights.
"""

    comparison_file = (
        EVALUATION_DIR /
        "model_comparison.txt"
    )

    with open(
        comparison_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            comparison_text.strip()
        )


    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print("\n" + "=" * 60)

    print(
        "Model evaluation completed successfully."
    )

    print("=" * 60)

    print(
        "\nMetrics:",
        metrics_file
    )

    print(
        "Comparison:",
        comparison_file
    )

    print(
        "Loss comparison plot:",
        loss_plot
    )

    print(
        "Accuracy comparison plot:",
        accuracy_plot
    )

    print(
        "\nResNet-34 U-Net Mean IoU:",
        f"{unet_miou:.4f}"
    )

    print(
        "ResNet-50 Substitute Mean IoU:",
        f"{resnet_miou:.4f}"
    )

    print(
        "Better model by Mean IoU:",
        better_model
    )


if __name__ == "__main__":
    main()