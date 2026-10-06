import json
from pathlib import Path

import matplotlib.pyplot as plt


HISTORY_FILE = Path("models/evaluation/unet_history.json")
OUTPUT_DIR = Path("outputs/plots")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


with open(HISTORY_FILE, "r", encoding="utf-8") as f:
    history = json.load(f)

epochs = range(1, len(history["train_loss"]) + 1)

# ---------------------------------------------------------
# Plot 1: Training and validation loss
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("U-Net Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)

loss_file = OUTPUT_DIR / "unet_loss_curves.png"

plt.savefig(
    loss_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ---------------------------------------------------------
# Plot 2: Validation accuracy
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    [x * 100 for x in history["val_acc"]],
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("U-Net Validation Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

accuracy_file = OUTPUT_DIR / "unet_validation_accuracy.png"

plt.savefig(
    accuracy_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Training curves created successfully.")
print("Loss plot:", loss_file)
print("Accuracy plot:", accuracy_file)