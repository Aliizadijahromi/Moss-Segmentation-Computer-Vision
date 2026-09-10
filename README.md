
# Moss Segmentation Using Computer Vision

MATLAB computer vision project for detecting and segmenting moss/vegetation on wall surfaces using classical image processing and deep learning.

## Project Overview

This project investigates two approaches for semantic segmentation of moss/vegetation from wall images:

1. Hybrid Image Processing + Active Contour
2. DeepLabv3+ Semantic Segmentation with ResNet-50

The objective was to identify vegetation regions at pixel level and compare a traditional computer vision pipeline with a deep-learning-based segmentation approach.

## Method 1 – Hybrid Image Processing

The classical computer vision pipeline consists of:

- Green-channel extraction
- CLAHE contrast enhancement
- Median filtering
- Canny edge detection
- Morphological closing
- Chan-Vese active contour segmentation
- Post-processing and noise removal

### Results

Average performance:

| Metric | Score |
|---|---:|
| Precision (CR) | 0.833 |
| Recall (CM) | 0.902 |
| F-measure | 0.831 |

## Method 2 – Deep Learning

A DeepLabv3+ semantic segmentation network with a ResNet-50 backbone was implemented in MATLAB.

Main configuration:

- Input image size: 256 × 256
- Classes: Background / Vegetation
- Data augmentation using rotation and flipping
- Class weighting for class imbalance
- SGDM optimizer
- 25 training epochs
- Morphological post-processing of predicted masks

The experiments produced high-quality segmentation in the best-performing samples, with best test F-measure reaching approximately 0.99.

## Evaluation

Segmentation performance was evaluated using:

- Precision
- Recall
- F-measure

The project demonstrates the difference between a manually designed image-processing pipeline and a learned semantic-segmentation model.

## Technologies & Skills

MATLAB • Computer Vision • Image Processing • Semantic Segmentation • Deep Learning • DeepLabv3+ • ResNet-50 • Active Contours • Canny Edge Detection • Morphological Processing

## Author

Ali Izadi Jahromi
