# GeoSense Agent 2.0 — Phase II Results

## Phase II: Deep Learning + Satellite Imagery

This phase implements a deep-learning workflow for multi-temporal Sentinel-2 imagery over the Agartala Municipal Corporation (AMC) study area.

### Data pipeline

Sentinel-2 imagery was prepared for 2015 and 2023 using the AMC study-area boundary.

| Item | 2015 | 2023 |
|---|---:|---:|
| Input bands | 6 | 6 |
| Raster shape | 6 × 1469 × 1267 | 6 × 1469 × 1267 |
| Mean NDVI | 0.4079 | 0.4261 |
| Mean NDWI | -0.4176 | -0.4276 |
| Cloud-free percentage | 100% | 100% |
| 224×224 chips | 48 | 48 |

Five pseudo-label classes were used: Urban, Vegetation, Water, Bare Land and Agriculture.

### U-Net training

A ResNet-34 U-Net was trained with 6 input channels and 5 output classes for 10 epochs.

- Total parameters: 24,446,357
- Best validation loss: 0.4248
- Final validation accuracy: 81.83%
- Model: `models/saved/unet_best.pth`

### Foundation-model fine-tuning / substitute configuration

The Phase II fine-tuning workflow attempted the Prithvi foundation-model path. The working saved model used the lab's ResNet-50 substitute configuration.

- Architecture: U-Net
- Encoder: ResNet-50
- Input channels: 6
- Output classes: 5
- Total parameters: 32,531,093
- Trainable parameters: 10,449,045
- Frozen parameter tensors: 90
- Trainable encoder layers: layer1, layer2
- Optimizer: AdamW
- Backbone learning rate: 1e-5
- Head learning rate: 1e-4
- Final validation accuracy: 75.70%
- Highest validation accuracy: 76.25% at epoch 9
- Model: `models/saved/prithvi_finetuned.pth`

### Model evaluation

| Model | Pixel Accuracy | Mean IoU |
|---|---:|---:|
| ResNet-34 U-Net | 84.65% | 0.4143 |
| ResNet-50 Fine-tuned Substitute | 77.07% | 0.2573 |

The ResNet-34 U-Net achieved the better evaluation performance by mean IoU on the held-out validation samples.

### Evaluation plots

Generated in `outputs/plots/`:

- `confusion_matrix_resnet_34_u_net.png`
- `confusion_matrix_resnet_50_substitute.png`
- `training_curves_comparison.png`
- `validation_accuracy_comparison.png`
- `unet_loss_curves.png`
- `unet_validation_accuracy.png`

### image_classifier.py

`src/phase2_dl/image_classifier.py` integrates the Phase II inference pipeline. It loads the fine-tuned model first, converts latitude/longitude to raster pixels, extracts a 224 × 224 × 6 patch, fills invalid values with the band mean, computes NDVI/NDWI, performs land-cover inference, returns class/confidence/distribution, and compares 2015 and 2023 NDVI using the threshold `|ΔNDVI| > 0.15`.

### Change-detection validation

#### Changed location

- Coordinates: 23.909821, 91.230331
- Land cover: Vegetation
- Class ID: 1
- Confidence: 59.25%
- NDVI 2015 centre: 0.4118
- NDVI 2023 centre: 0.5907
- Absolute NDVI difference: 0.1789
- Change flag: True
- Changed pixels in patch: 28.65%
- Inference time: 0.1477 s

#### Unchanged location

- Coordinates: 23.825971, 91.268249
- Land cover: Agriculture
- Class ID: 4
- Confidence: 71.48%
- NDVI 2015 centre: 0.2661
- NDVI 2023 centre: 0.2084
- Absolute NDVI difference: 0.0577
- Change flag: False
- Changed pixels in patch: 11.58%
- Inference time: 0.1210 s

Both inference tests completed well below the required 3-second limit.
