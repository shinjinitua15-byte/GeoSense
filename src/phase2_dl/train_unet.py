import json
import random
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp


# ---------------------------------------------------------
# GeoSense Phase II - Exercise 3.2
# U-Net semantic segmentation
# ---------------------------------------------------------

CHIP_DIR = Path("data/satellite/chips")
LABEL_DIR = Path("data/satellite/labels")

MODEL_DIR = Path("models/saved")
EVALUATION_DIR = Path("models/evaluation")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


YEARS = [2015, 2023]

CHIP_SIZE = 224
STRIDE = 174
NUM_CLASSES = 5
IN_CHANNELS = 6

BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.001

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
# Dataset
# ---------------------------------------------------------

class SatelliteDataset(Dataset):
    """
    Dataset that pairs each Sentinel-2 chip with its
    corresponding label patch.

    The label patch is extracted from the full-size
    label mask using the same row/column positions used
    when creating the image chips.
    """

    def __init__(self, samples, transform=None):

        self.samples = samples
        self.transform = transform

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        chip_path, label_path, row, col = self.samples[index]

        # Load six-band image chip
        image = np.load(chip_path).astype(np.float32)

        # Load complete label mask
        label_mask = np.load(label_path)

        # Extract matching 224 x 224 label patch
        mask = label_mask[
            row:row + CHIP_SIZE,
            col:col + CHIP_SIZE
        ].astype(np.int64)

        # Convert channel-first -> channel-last
        image_hwc = np.transpose(image, (1, 2, 0))

        # Apply augmentation
        if self.transform is not None:

            transformed = self.transform(
                image=image_hwc,
                mask=mask
            )

            image_hwc = transformed["image"]
            mask = transformed["mask"]

        # Convert back to channel-first
        image_chw = np.transpose(
            image_hwc,
            (2, 0, 1)
        )

        image_tensor = torch.from_numpy(
            image_chw.copy()
        ).float()

        mask_tensor = torch.from_numpy(
            mask.copy()
        ).long()

        return image_tensor, mask_tensor


# ---------------------------------------------------------
# Create sample list
# ---------------------------------------------------------

def build_samples():

    samples = []

    for year in YEARS:

        chip_dir = CHIP_DIR / str(year)
        label_file = LABEL_DIR / f"labels_{year}.npy"

        if not chip_dir.exists():
            raise FileNotFoundError(
                f"Chip directory not found: {chip_dir}"
            )

        if not label_file.exists():
            raise FileNotFoundError(
                f"Label file not found: {label_file}"
            )

        chip_files = sorted(
            chip_dir.glob("chip_*.npy")
        )

        if len(chip_files) == 0:
            raise RuntimeError(
                f"No chips found in {chip_dir}"
            )

        # Read label dimensions
        label_mask = np.load(label_file)

        height, width = label_mask.shape

        # Reproduce the same window positions used
        # for the generated 48 chips.
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

                positions.append((row, col))

        if len(chip_files) != len(positions):

            raise RuntimeError(
                f"Chip/position mismatch for {year}: "
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

        print(
            f"{year}: {len(chip_files)} "
            f"chip-label pairs"
        )

    return samples


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

def create_model():

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES
    )

    return model


# ---------------------------------------------------------
# Accuracy
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
# Training
# ---------------------------------------------------------

def train_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )
    else:
        print("GPU: CPU execution")

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

    full_dataset = SatelliteDataset(
        samples,
        transform=train_transform
    )

    # 80/20 split
    train_size = int(
        0.8 * len(full_dataset)
    )

    val_size = (
        len(full_dataset) - train_size
    )

    train_dataset, val_dataset = random_split(
        full_dataset,
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
    # Data loaders
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

    model = create_model()

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
        "\nU-Net model created successfully."
    )

    print(
        "Encoder: ResNet-34"
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

    model = model.to(device)

    # -----------------------------------------------------
    # Loss and optimizer
    # -----------------------------------------------------

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

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
        MODEL_DIR / "unet_best.pth"
    )

    final_model_path = (
        MODEL_DIR / "unet_final.pth"
    )

    history_path = (
        EVALUATION_DIR / "unet_history.json"
    )

    # -----------------------------------------------------
    # Epoch loop
    # -----------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        # ================================================
        # Training
        # ================================================

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
            running_train_loss
            / len(train_loader.dataset)
        )

        # ================================================
        # Validation
        # ================================================

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
            running_val_loss
            / len(val_loader.dataset)
        )

        val_acc = (
            total_correct
            / total_pixels
        )

        scheduler.step()

        # Save history
        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_acc"].append(
            val_acc
        )

        # Save best model
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                best_model_path
            )

            best_status = " ← BEST"

        else:

            best_status = ""

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%"
            f"{best_status}"
        )

    # -----------------------------------------------------
    # Save final model
    # -----------------------------------------------------

    torch.save(
        model.state_dict(),
        final_model_path
    )

    # -----------------------------------------------------
    # Save history
    # -----------------------------------------------------

    history["epochs"] = EPOCHS

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
    # Final output
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("U-Net training completed successfully.")
    print("=" * 60)

    print(
        "Best model:",
        best_model_path
    )

    print(
        "Final model:",
        final_model_path
    )

    print(
        "Training history:",
        history_path
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Final validation accuracy: "
        f"{history['val_acc'][-1] * 100:.2f}%"
    )


if __name__ == "__main__":
    train_model()