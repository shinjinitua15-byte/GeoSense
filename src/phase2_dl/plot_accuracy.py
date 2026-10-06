import json
from pathlib import Path

import matplotlib.pyplot as plt


HISTORY_FILE = Path("models/evaluation/unet_history.json")
OUTPUT_FILE = Path("outputs/plots/unet_validation_accuracy.png")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(HISTORY_FILE, "r", encoding="utf-8") as f:
    history = json.load(f)

epochs = range(1, len(history["val_acc"]) + 1)
accuracy = [value * 100 for value in history["val_acc"]]

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    accuracy,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("U-Net Validation Accuracy")
plt.xticks(list(epochs))
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

print("Validation accuracy graph created successfully.")
print("Saved to:", OUTPUT_FILE)

plt.show()