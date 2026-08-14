# Liver Microenvironment AI Profiler

## Ethics and Compliance

> **Data Declaration**
>
> This code repository contains no real patient data. Actual data must be obtained by users independently and used in compliance with applicable local ethical regulations including but not limited to HIPAA, GDPR, and institutional IRB requirements.

## Data Input Specifications

- **Format**: Microsoft Excel (`.csv` / `.xlsx`) and standard image formats (`.png`, `.tif`)
- **Required Columns for Statistical Analysis**:

| **Column Name**            | **Data Type** | **Description**                                              |
| -------------------------- | ------------- | ------------------------------------------------------------ |
| `WSI_ID`                   | String        | Anonymized whole slide image identifier                      |
| `Group`                    | Categorical   | Experimental cohort assignment (`LT-free group` or `LT-group`) |
| `Patch_Count`              | Numeric       | Total number of valid tissue patches extracted               |
| `CD4`, `CD8`, `PD-1`, etc. | Numeric       | Spatial omics expression levels                              |

## Directory Structure

Plaintext

```
Project/
├── 01_WSI_Data_Preprocessing/
│   └── data/
│       └── patches_normalized/
├── 08_GigaTIME_Spatial_Omics/
│   ├── scripts/
│   ├── output/
│   │   ├── NPY_Files/
│   │   ├── Comprehensive_Panels/
│   │   └── statistical_analysis/
│   ├── step1_wsi_level_inference.py
│   └── step2_wsi_statistical_analysis.py
```

## Pipeline Workflow

1. **Whole Slide Image Preprocessing**: Perform image resampling, tissue segmentation, and patch generation from liver biopsy and transplantation cohorts.
2. **Spatial Omics Feature Extraction**: Execute the GigaTIME foundation model to predict the spatial distribution of 21 multiplex immunofluorescence markers from H&E patches.
3. **Single-Cell Visualization**: Integrate StarDist instance segmentation to map multi-omics expressions onto individual cells and generate 5x5 comprehensive spatial panels.
4. **WSI-Level Quantification**: Calculate the distribution of molecular expressions across the entire whole slide image and export the mean features.
5. **Statistical Significance Testing**: Calculate the consistency of feature variables and perform Mann-Whitney U tests to evaluate group differences (`LT-free` vs `LT-group`).
6. **Feature Importance Analysis**: Generate LogP waterfall plots using multivariable logistic regression signatures and evaluate model performance.

## Environment Requirements

Plaintext

```
torch
torchvision
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
opencv-python
stardist
csbdeep
skimage
huggingface_hub
tensorflow==2.10.0
```