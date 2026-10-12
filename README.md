# GeoSense Agent 2.0
### Multi-Temporal Satellite Imagery, Deep Learning, LiDAR and Street-View Analysis

GeoSense Agent 2.0 is a geospatial analysis project that brings together satellite imagery, deep-learning-based land-cover analysis, LiDAR point-cloud processing and street-level semantic analysis. Its objective is to support richer site assessment by combining information derived from different geospatial data sources.

The repository documents Phase II results for multi-temporal Sentinel-2 imagery over the Agartala Municipal Corporation (AMC) study area and Phase III experiments involving LiDAR and street-view imagery.

> **Project status:** Phase II model experiments and Phase III component tests are documented. Full end-to-end integration of Phase I, Phase II and Phase III has not yet been verified.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Phase II: Deep Learning and Satellite Imagery](#phase-ii-deep-learning-and-satellite-imagery)
- [Phase III: LiDAR and Street-View Analysis](#phase-iii-lidar-and-street-view-analysis)
- [Testing and Evidence](#testing-and-evidence)
- [Repository Documentation](#repository-documentation)
- [Limitations](#limitations)
- [Future Work](#future-work)

---

## Project Overview

GeoSense explores how different geospatial data sources can contribute to site assessment.

| Phase | Focus | Documented contribution |
|---|---|---|
| Phase I | Earlier site-assessment workflow | Integration outputs require further verification |
| Phase II | Multi-temporal Sentinel-2 imagery and deep learning | Land-cover model experiments, evaluation and NDVI change detection |
| Phase III | LiDAR and street-view analysis | Point-cloud metrics, streetscape indicators and JSON profile generation |

The phases are intended to complement one another. Their combined workflow should only be considered complete after the required inputs and outputs have been connected and tested using compatible data.

## Phase II: Deep Learning and Satellite Imagery

Phase II implements a deep-learning workflow for multi-temporal Sentinel-2 imagery over the Agartala Municipal Corporation (AMC) study area.

### Data preparation

Imagery was prepared for 2015 and 2023 using the AMC study-area boundary.

| Item | 2015 | 2023 |
|---|---:|---:|
| Input bands | 6 | 6 |
| Raster shape | 6 × 1469 × 1267 | 6 × 1469 × 1267 |
| Mean NDVI | 0.4079 | 0.4261 |
| Mean NDWI | -0.4176 | -0.4276 |
| Cloud-free percentage | 100% | 100% |
| 224 × 224 chips | 48 | 48 |

Five pseudo-label classes were used:

- Urban
- Vegetation
- Water
- Bare Land
- Agriculture

### U-Net training

A ResNet-34 U-Net was trained with six input channels and five output classes for ten epochs.

- Total parameters: 24,446,357
- Best validation loss: 0.4248
- Final validation accuracy: 81.83%
- Saved model: `models/saved/unet_best.pth`

### Foundation-model fine-tuning workflow

The fine-tuning workflow attempted a Prithvi foundation-model approach. The documented working saved model used a ResNet-50 substitute configuration.

- Architecture: U-Net
- Encoder: ResNet-50
- Input channels: 6
- Output classes: 5
- Total parameters: 32,531,093
- Trainable parameters: 10,449,045
- Frozen parameter tensors: 90
- Trainable encoder layers: `layer1`, `layer2`
- Optimizer: AdamW
- Backbone learning rate: `1e-5`
- Head learning rate: `1e-4`
- Final validation accuracy: 75.70%
- Highest validation accuracy: 76.25% at epoch 9
- Saved model: `models/saved/prithvi_finetuned.pth`

The saved model uses the substitute configuration described above; it should not be represented as a successfully fine-tuned Prithvi foundation model.

### Model evaluation

| Model | Pixel Accuracy | Mean IoU |
|---|---:|---:|
| ResNet-34 U-Net | 84.65% | 0.4143 |
| ResNet-50 fine-tuned substitute | 77.07% | 0.2573 |

On the reported evaluation, the ResNet-34 U-Net achieved higher pixel accuracy and mean IoU than the ResNet-50 substitute.

Generated evaluation plots are stored in `outputs/plots/`:

- `confusion_matrix_resnet_34_u_net.png`
- `confusion_matrix_resnet_50_substitute.png`
- `training_curves_comparison.png`
- `validation_accuracy_comparison.png`
- `unet_loss_curves.png`
- `unet_validation_accuracy.png`

### Inference and change detection

The Phase II inference module is located at `src/phase2_dl/image_classifier.py`.

The documented pipeline loads the fine-tuned model first, converts geographic coordinates to raster pixels, extracts a 224 × 224 × 6 patch, fills invalid values with band means, computes NDVI and NDWI, performs land-cover inference, and compares 2015 and 2023 NDVI using the threshold `|ΔNDVI| > 0.15`.

**Changed-location example**

- Coordinates: `23.909821, 91.230331`
- Predicted land cover: Vegetation
- Class ID: 1
- Confidence: 59.25%
- NDVI in 2015: 0.4118
- NDVI in 2023: 0.5907
- Absolute NDVI difference: 0.1789
- Change flag: `True`
- Changed pixels in patch: 28.65%
- Reported inference time: 0.1477 seconds

**Unchanged-location example**

- Coordinates: `23.825971, 91.268249`
- Predicted land cover: Agriculture
- Class ID: 4
- Confidence: 71.48%
- NDVI in 2015: 0.2661
- NDVI in 2023: 0.2084
- Absolute NDVI difference: 0.0577
- Change flag: `False`
- Changed pixels in patch: 11.58%
- Reported inference time: 0.1210 seconds

Both documented inference examples completed below the three-second target. These examples are reported outputs, not independent proof of classification accuracy or change-detection validity.

---

## Phase III: LiDAR and Street-View Analysis

Phase III extends the project with point-cloud processing, street-view semantic analysis and a unified JSON site-profile workflow.

### Core modules

The Phase III modules are organized under `src/phase3_lidar/`.

| Module | Purpose |
|---|---|
| `lidar_analyzer.py` | Filters LiDAR points by geographic bounding box and derives point-cloud metrics |
| `lidar_features.py` | Supports LiDAR feature extraction and point-cloud processing |
| `mapillary_download.py` | Supports street-view image collection |
| `sam_rooftop_extraction.py` | Supports rooftop extraction using the Segment Anything Model (SAM) |
| `streetview_semantic.py` | Analyzes street-view scenes and estimates semantic proportions and walkability indicators |
| `unified_site_profile.py` | Organizes available analysis results into a structured site profile |

### LiDAR analysis

The documented LiDAR example uses a bounding box near longitude `-114.03424` and latitude `45.38575`.

| Metric | Reported result |
|---|---:|
| Points analyzed | 1,121,805 |
| Average building height | 14.82 m |
| Canopy point share | 7.81% |
| Impervious point share | 12.87% |
| Slope-risk score | 10.0 / 10 |

These metrics describe the available point-cloud analysis. Canopy and impervious percentages are point-share proxies rather than validated land-area measurements. The slope-risk score is an indicator produced by the implemented method, not a standalone engineering or safety assessment.

#### LiDAR point-cloud visualization

The color-coded point-cloud screenshot is stored at:

[`docs/phase3/phase3_lidar_point_cloud.png`](docs/phase3/phase3_lidar_point_cloud.png)

### Street-view semantic analysis

The street-view workflow estimates the proportions of visible scene classes, including roads, sidewalks and vegetation, and produces exploratory streetscape indicators.

| Example location | Images | Road (%) | Sidewalk (%) | Vegetation (%) | Walkability score |
|---|---:|---:|---:|---:|---:|
| Original Area 1 | 11 | 32.9 | 5.0 | 26.7 | 1.83 |
| Original Area 2 | 14 | 33.1 | 4.9 | 26.2 | 1.81 |
| Chennai Centre | 1 | 29.9 | 0.0 | 3.9 | 0.00 |

These values are exploratory outputs. They have not been established as validated measures of pedestrian safety, accessibility or walkability. The two original areas have overlapping image sets, and the Chennai Centre result is based on only one image.

### Unified site profile

The `unified_site_profile.py` module creates a structured site profile from geographic coordinates, a search radius and the available analysis results.

Example profiles and their explanations are stored in:

- [`docs/phase3/profiles/`](docs/phase3/profiles/)
- [`docs/phase3/profiles/README.md`](docs/phase3/profiles/README.md)

The profiles demonstrate the report structure and how available data are represented. The current profiles do not contain verified Phase I suitability predictions or Phase II imagery-classification outputs.

**Geographic coverage limitation:** The LiDAR dataset and Chennai Centre street-view example represent different geographic areas. The Chennai profile reports zero matching LiDAR points because the available point cloud does not cover Chennai. Their metrics must not be combined or interpreted as measurements from the same location.

---

## Testing and Evidence

The recorded Phase III test command is:

```bash
D:\GeoSense_Phase3\geosense_lidar\python.exe -m pytest D:\GeoSense_Phase3\tests\test_phase3_modules.py -v
```

**Recorded result: 11 tests passed.**

The test cases cover LiDAR point classification, slope-risk behavior, site-profile fields, geographic and radius input validation, and walkability-score behavior.

The saved output is available at:

[`docs/phase3/checklist/phase3_test_results.txt`](docs/phase3/checklist/phase3_test_results.txt)

This is evidence that the included test cases passed. It does not establish that all Phase I, Phase II and Phase III components work together end to end.

Troubleshooting records are maintained at `outputs/reports/phase3_troubleshooting_notes.txt`.

---

## Repository Documentation

The following Phase III documentation is included in the repository:

- [Phase III completion checklist](docs/phase3/checklist/phase3_completion_checklist.md)
- [Phase III practicum notes](docs/phase3/checklist/phase3_practicum_notes.md)
- [Phase III test results](docs/phase3/checklist/phase3_test_results.txt)
- [LiDAR point-cloud screenshot](docs/phase3/phase3_lidar_point_cloud.png)
- [Example site profiles](docs/phase3/profiles/)
- [Profile documentation](docs/phase3/profiles/README.md)

---

## Limitations

- Full end-to-end integration of Phase I suitability, Phase II imagery classification and Phase III outputs remains pending.
- LiDAR and Chennai Centre street-view examples have different geographic coverage.
- Point-share metrics are not validated land-area estimates.
- Street-view semantic classifications and walkability indicators require validation against reference labels and independent measurements.
- Example profiles contain incomplete upstream outputs where Phase I and Phase II results are unavailable.
- Model metrics and example predictions should be interpreted in the context of the documented training and evaluation data.
- Data provenance, licensing, software-version reproducibility and backup verification remain part of the completion process.

## Future Work

1. Integrate actual Phase I suitability predictions and Phase II imagery-classification outputs into the unified site profile.
2. Test the complete workflow using spatially compatible datasets from the same study area.
3. Validate LiDAR metrics against independent reference data.
4. Evaluate street-view segmentation against manually labelled imagery.
5. Improve impervious-surface and canopy estimates using suitable reference data.
6. Document required software versions and reproducible environment setup.
7. Verify dataset provenance, coverage and licensing.
8. Complete final repository review and maintain verified project backups.

---

## Project Status

GeoSense Agent 2.0 documents multi-temporal satellite-image model experiments and Phase III LiDAR and street-view analysis components. The Phase III test record contains 11 passing tests, and the repository includes example profiles, a point-cloud visualization and supporting documentation.

**The project should be considered a partially integrated geospatial workflow until the Phase I and Phase II outputs have been connected to the unified profile and the combined workflow has been tested and validated.**
