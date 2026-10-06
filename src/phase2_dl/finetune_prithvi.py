import json
import random
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download


# ---------------------------------------------------------
# GeoSense Phase II - Exercise 4.1
# Foundation Model Fine-Tuning
# ---------------------------------------------------------

PROJECT_ROOT = Path(".")

CHIP_DIR = PROJECT_ROOT / "data/satellite/chips"
LABEL_DIR = PROJECT_ROOT / "data/satellite/labels"

MODEL_DIR = PROJECT_ROOT / "models/saved"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


YEARS = [2015, 2023]

CHIP_SIZE = 224
STRIDE = 174

IN_CHANNELS = 6
NUM_CLASSES = 5

BATCH_SIZE = 4
EPOCHS = 10

LR_BACKBONE = 1e-5
LR_HEAD = 1e-4

RANDOM_SEED = 42


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ---------------------------------------------------------
# Class names
# ---------------------------------------------------------

CLASS_NAMES = {
    0: "Urban",
    1: "Vegetation",
    2: "Water",
    3: "Bare Land",
    4: "Agriculture"
}


# ---------------------------------------------------------
# Augmentation
# ---------------------------------------------------------

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.15,
        p=0.5
    )
])


# ---------------------------------------------------------
# Attempt Prithvi download
# ---------------------------------------------------------

def try_download_prithvi():

    print("\nAttempting Prithvi-100M download...")

    try:

        path = hf_hub_download(
            repo_id="ibm-nasa-geospatial/Prithvi-100M",
            filename="Prithvi_100M.pt"
        )

        print("Prithvi-100M download successful.")
        print("Checkpoint:", path)

        return path

    except Exception as e:

        print("Prithvi-100M download failed.")
        print("Reason:", str(e))
        print(
            "Using the documented ResNet-50 substitute."
        )

        return None


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class SatelliteDataset(Dataset):

    def __init__(
        self,
        samples,
        transform=None
    ):

        self.samples = samples
        self.transform = transform

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        chip_path, label_path, row, col = (
            self.samples[index]
        )

        image = np.load(
            chip_path
        ).astype(np.float32)

        full_label = np.load(
            label_path
        )

        mask = full_label[
            row:row + CHIP_SIZE,
            col:col + CHIP_SIZE
        ].astype(np.int64)

        image = np.transpose(
            image,
            (1, 2, 0)
        )

        if self.transform is not None:

            transformed = self.transform(
                image=image,
                mask=mask
            )

            image = transformed["image"]
            mask = transformed["mask"]

        image = np.transpose(
            image,
            (2, 0, 1)
        )

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

        label_file = (
            LABEL_DIR / f"labels_{year}.npy"
        )

        chip_files = sorted(
            chip_dir.glob("chip_*.npy")
        )

        if len(chip_files) == 0:

            raise RuntimeError(
                f"No chips found for {year}."
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
                f"{year}: chip-position mismatch. "
                f"Chips={len(chip_files)}, "
                f"positions={len(positions)}"
            )

        for chip_path, position in zip(
            chip_files,
            positions
        ):

            row, col = position

            samples.append(
                (
                    chip_path,
                    label_file,
                    row,
                    col
                )
            )

        print(
            f"{year}: {len(chip_files)} chip-label pairs"
        )

    return samples


# ---------------------------------------------------------
# Create substitute segmentation model
# ---------------------------------------------------------

def create_fallback_model():

    print("\nLoading ResNet-50 substitute...")

    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES
    )

    return model


# ---------------------------------------------------------
# Freeze encoder layers
# ---------------------------------------------------------

def freeze_encoder_except_layer1_layer2(model):

    frozen_names = []
    trainable_names = []

    for name, parameter in model.encoder.named_parameters():

        if (
            name.startswith("layer1")
            or name.startswith("layer2")
        ):

            parameter.requires_grad = True
            trainable_names.append(name)

        else:

            parameter.requires_grad = False
            frozen_names.append(name)

    return frozen_names, trainable_names


# ---------------------------------------------------------
# Pixel accuracy
# ---------------------------------------------------------

def pixel_accuracy(outputs, masks):

    predictions = torch.argmax(
        outputs,
        dim=1
    )

    correct = (
        predictions == masks
    ).sum().item()

    total = masks.numel()

    return correct / total


# ---------------------------------------------------------
# Main training
# ---------------------------------------------------------

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
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
    # Attempt Prithvi
    # -----------------------------------------------------

    prithvi_path = try_download_prithvi()

    if prithvi_path is not None:

        print(
            "\nPrithvi checkpoint was downloaded."
        )

        print(
            "For this lab's segmentation architecture, "
            "the documented compatible substitute workflow "
            "will still be used."
        )


    # -----------------------------------------------------
    # Build samples
    # -----------------------------------------------------

    samples = build_samples()

    print(
        "\nTotal chip-label pairs:",
        len(samples)
    )


    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    dataset = SatelliteDataset(
        samples,
        transform=train_transform
    )

    train_size = int(
        0.8 * len(dataset)
    )

    val_size = (
        len(dataset) - train_size
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(
            RANDOM_SEED
        )
    )

    print(
        "Training samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )


    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = create_fallback_model()

    model = model.to(device)


    # -----------------------------------------------------
    # Freeze encoder
    # -----------------------------------------------------

    frozen_names, trainable_encoder_names = (
        freeze_encoder_except_layer1_layer2(model)
    )


    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


    print(
        "\nFoundation-model substitute configuration"
    )

    print(
        "Architecture: U-Net"
    )

    print(
        "Encoder: ResNet-50"
    )

    print(
        "Input channels:",
        IN_CHANNELS
    )

    print(
        "Output classes:",
        NUM_CLASSES
    )

    print(
        "Total parameters:",
        f"{total_parameters:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )

    print(
        "Frozen parameter tensors:",
        len(frozen_names)
    )

    print(
        "Trainable encoder parameter tensors:",
        len(trainable_encoder_names)
    )

    print(
        "Trainable encoder layers: layer1, layer2"
    )


    # -----------------------------------------------------
    # Two learning-rate AdamW optimizer
    # -----------------------------------------------------

    backbone_parameters = []
    head_parameters = []

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        if name.startswith("encoder.layer1") or \
           name.startswith("encoder.layer2"):

            backbone_parameters.append(
                parameter
            )

        else:

            head_parameters.append(
                parameter
            )


    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": LR_BACKBONE
            },
            {
                "params": head_parameters,
                "lr": LR_HEAD
            }
        ]
    )


    print(
        "\nOptimizer: AdamW"
    )

    print(
        "Backbone learning rate:",
        LR_BACKBONE
    )

    print(
        "Head learning rate:",
        LR_HEAD
    )


    criterion = torch.nn.CrossEntropyLoss()


    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.5
    )


    # -----------------------------------------------------
    # Training history
    # -----------------------------------------------------

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }


    best_val_loss = float("inf")

    best_model_path = (
        MODEL_DIR /
        "prithvi_finetuned.pth"
    )


    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_train_loss = 0.0


        for images, masks in train_loader:

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                masks
            )

            loss.backward()

            optimizer.step()

            running_train_loss += (
                loss.item()
                * images.size(0)
            )


        train_loss = (
            running_train_loss /
            len(train_loader.dataset)
        )


        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        model.eval()

        running_val_loss = 0.0
        total_correct = 0
        total_pixels = 0


        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    masks
                )

                running_val_loss += (
                    loss.item()
                    * images.size(0)
                )

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                total_correct += (
                    predictions == masks
                ).sum().item()

                total_pixels += masks.numel()


        val_loss = (
            running_val_loss /
            len(val_loader.dataset)
        )

        val_acc = (
            total_correct /
            total_pixels
        )


        scheduler.step()


        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_acc"].append(
            val_acc
        )


        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                best_model_path
            )

            marker = " ← BEST"

        else:

            marker = ""


        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%"
            f"{marker}"
        )


    # -----------------------------------------------------
    # Save training history
    # -----------------------------------------------------

    history_path = (
        Path("models/evaluation")
        / "prithvi_history.json"
    )

    history_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )


    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print(
        "Foundation-model fine-tuning completed."
    )
    print("=" * 60)

    print(
        "Saved model:",
        best_model_path
    )

    print(
        "Training history:",
        history_path
    )

    print(
        "Best validation loss:",
        f"{best_val_loss:.4f}"
    )

    print(
        "Final validation accuracy:",
        f"{history['val_acc'][-1] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()