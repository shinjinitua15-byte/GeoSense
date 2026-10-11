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
---

# Phase III: LiDAR Point-Cloud Analysis + Street-View Semantics

Phase III extends GeoSense Agent 2.0 with LiDAR point-cloud processing, street-view semantic segmentation, and a unified site-profile workflow.

## LiDAR analysis

The `src/phase3_lidar/` module contains the following components:

- `lidar_analyzer.py` — analyzes spatial bounding boxes and derives building-height, canopy, impervious-surface, and slope-risk indicators.
- `lidar_features.py` — supports LiDAR feature extraction and point-cloud processing.
- `mapillary_download.py` — supports street-view image collection.
- `sam_rooftop_extraction.py` — supports rooftop extraction using the Segment Anything Model (SAM).
- `streetview_semantic.py` — analyzes street-view scenes and estimates semantic class proportions and walkability indicators.
- `unified_site_profile.py` — combines available LiDAR and street-view analysis results into a structured site profile.

### Example LiDAR results

For the tested LiDAR bounding box near longitude -114.03424 and latitude 45.38575:

| Metric | Result |
|---|---:|
| Points analyzed | 1,121,805 |
| Average building height | 14.82 m |
| Canopy point share | 7.81% |
| Impervious point share | 12.87% |
| Slope-risk score | 10.0 / 10 |

**Interpretation:** Canopy and impervious percentages are point-cloud-based proxies, not direct measurements of land-area coverage. Results depend on the classification rules and the spatial extent of the selected bounding box.

## Street-view semantic analysis

The street-view workflow uses image segmentation to estimate the proportions of scene classes, including roads, sidewalks, buildings, vegetation, sky, and cars. These measurements support comparative streetscape analysis and exploratory walkability scoring.

Example results from the tested locations:

| Location | Images | Road (%) | Sidewalk (%) | Vegetation (%) | Walkability score |
|---|---:|---:|---:|---:|---:|
| Original Area 1 | 11 | 32.9 | 5.0 | 26.7 | 1.83 |
| Original Area 2 | 14 | 33.1 | 4.9 | 26.2 | 1.81 |
| Chennai Centre | 1 | 29.9 | 0.0 | 3.9 | 0.00 |

These scores are exploratory indicators rather than validated pedestrian-safety or accessibility assessments. The two original areas have overlapping image sets, and the Chennai Centre result is based on only one image.

## Unified site profile

The `unified_site_profile.py` module defines an `EnhancedSiteProfile` structure with geographic coordinates, building-height estimates, canopy and impervious indicators, slope risk, scene composition, and walkability score.

The workflow saves a JSON report under `outputs/reports/`. Phase 1 suitability and Phase 2 imagery-classification outputs remain incomplete in the current integrated profile and should not be interpreted as validated results.

## Testing and troubleshooting

The Phase III test suite was executed using the project's Python environment:

```bash
D:\GeoSense_Phase3\geosense_lidar\python.exe -m pytest D:\GeoSense_Phase3\tests\test_phase3_modules.py -v
```

**Verified test result:** 11 tests passed.

Troubleshooting notes are maintained in `outputs/reports/phase3_troubleshooting_notes.txt`. Issues investigated included GPU availability, missing PDAL support, empty street-view results, memory constraints, and test-import corrections.

## Limitations and future improvements

- Validate point-cloud class labels against reference data.
- Estimate canopy and impervious land-area coverage using suitable imagery or reference datasets.
- Test additional locations with sufficient street-view coverage.
- Integrate and validate Phase 1 suitability and Phase 2 imagery-classification outputs in the unified profile.
- Add a color-coded point-cloud screenshot and example JSON profiles to the portfolio.


