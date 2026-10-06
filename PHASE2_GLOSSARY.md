# GeoSense Agent 2.0 — Phase II Eight-Term Glossary Application

## 1. Deep Learning
Deep learning uses multi-layer neural networks to learn patterns directly from data. In GeoSense Phase II, it is used to learn spatial and spectral patterns from six-band Sentinel-2 image patches.

## 2. CNN
A Convolutional Neural Network (CNN) learns spatial patterns using convolution operations. CNN-based encoders are used to extract land-cover features from satellite imagery.

## 3. U-Net
U-Net is an encoder-decoder architecture designed for image segmentation. The encoder extracts features and the decoder reconstructs a detailed pixel-level class map.

## 4. Semantic Segmentation
Semantic segmentation assigns a class label to each pixel. In this project the five classes are Urban, Vegetation, Water, Bare Land and Agriculture.

## 5. Foundation Model
A foundation model is trained on broad data and adapted for a specific downstream task. The Phase II workflow attempted the Prithvi foundation-model path and used the working ResNet-50 substitute configuration.

## 6. Fine-tuning
Fine-tuning adapts a pretrained model to a target dataset and task. In the Phase II implementation, selected encoder layers were trainable while other encoder parameters were frozen, with separate learning rates for backbone and head.

## 7. NDVI
The Normalized Difference Vegetation Index is calculated as NDVI = (NIR − Red) / (NIR + Red). It is used to indicate vegetation condition and is compared between 2015 and 2023 for change detection.

## 8. IoU
Intersection over Union (IoU) measures overlap between predicted and reference regions: IoU = Intersection / Union. Per-class IoU and mean IoU were used to evaluate the two segmentation models.
