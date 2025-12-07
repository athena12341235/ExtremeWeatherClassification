# Extreme Weather Classification
This repository provides the full workflow for preprocessing satellite images, generating EDA, training deep learning models, and producing interpretability outputs such as Grad-CAM heatmaps. The dataset comes from the Harvard Dataverse version of the Li & Momen (2021) weather-events dataset. The goal of this case study is to achieve ≥ 92% test accuracy with both MobileNetV2 and EfficientNetV2, matching the benchmark from the original researchers.

## Repository Structure
```text
DATA/
  raw_data/          # Original Dataverse images (user downloads)
  cleaned_data/      # Produced by preprocess.py after cleaning + relabeling

OUTPUT/
  EDA/                           # All EDA plots
  gradcam_mobilenetv2/           # Grad-CAM visualizations
  gradcam_efficientnetv2/        # Grad-CAM visualizations
  mobilenetv2_*                  # Reports, metrics, confusion matrix
  efficientnet_v2_*              # Reports, metrics, confusion matrix

SCRIPTS/
  preprocess.py
  eda.py
  train_mobilenetv2.py
  train_efficientnetv2.py

README.md
Hook.pdf  
Rubric.pdf  
Resources.pdf
```
## How to Reproduce Results
1. Clone this repository (https://github.com/athena12341235/ExtremeWeatherClassification).
2. Download the raw dataset. See `DATA/raw_data.pdf` for the official download link. 
3. Unzip the dataset. You should see five folders, one for each original weather category. Place all five folders into `DATA/`
4. Preprocess the data: `python3 SCRIPTS/preprocess.py`. This creates `cleaned_data/` containing the final `extreme/` and `normal/` image folders.
5. Run EDA: `python3 SCRIPTS/eda.py`. Plots appear in `OUTPUT/EDA/`.
6. Train MobileNetV2: `python3 SCRIPTS/train_mobilenetv2.py --split-root dataset_split --gradcam`.
7. Train EfficientNetV2: `python3 SCRIPTS/train_efficientnetv2.py --split-root dataset_split --gradcam`.

## References
1. Li, Ye, and Mostafa Momen. 2024. “9081 Images Dataset for: Detection of Weather Events in Optical Satellite Data Using Deep Convolutional Neural Networks.” Harvard Dataverse. https://doi.org/10.7910/DVN/PUIHVC.
2. Li, Ye, and Mostafa Momen. 2021. “Detection of Weather Events in Optical Satellite Data Using Deep Convolutional Neural Networks.” Remote Sensing Letters 12 (12): 1227–37. https://doi.org/10.1080/2150704X.2021.1978581.