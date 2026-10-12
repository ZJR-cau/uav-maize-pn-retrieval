# UAV Maize Pn Retrieval

An explainable machine learning framework for estimating maize leaf net photosynthetic rate (Pn) from UAV multispectral, thermal infrared, and LiDAR observations.

## Overview

Direct measurement of leaf Pn by gas exchange is labor intensive and difficult to scale across field populations. This project investigates whether complementary UAV observations can support plot level retrieval of average leaf Pn under field conditions, and examines how the sensing modalities and derived features contribute to prediction.

The repository accompanies the study **“From Multimodal Canopy Features to Photosynthetic Function: Explainable Machine Learning Disentangles the Contributions of UAV Remote Sensing to Maize Photosynthetic Rate Retrieval.”**

## Study data

The study dataset contains 2,376 spatially matched samples collected at two locations in China across three site year combinations: Fangshan in 2025, and Gongzhuling in 2025 and 2026. Field measurements and UAV observations were collected at the V6, V9, and V12 growth stages.

For each sampling region, the ground truth was the average of six gas exchange measurements from healthy, fully expanded leaves in the middle to upper canopy. The UAV data included:

- Four band multispectral imagery: green, red, red edge, and near infrared
- Thermal infrared imagery
- LiDAR point clouds

The number of samples differed substantially among site year combinations. In particular, Gongzhuling 2025 contributed 60 samples, so performance should be interpreted with this imbalance in mind.

**Data access:** Raw imagery, point clouds, and plot level physiological measurements are not included in this repository. Add the approved data repository, DOI, or access instructions here when available.

## Analysis workflow

The analysis follows these main steps:

1. Extract spectral, thermal, and LiDAR canopy features.
2. Reduce 110 candidate predictors to 20 features through Pearson correlation screening, variance inflation factor filtering, Boruta ranking, and recursive feature elimination with support vector regression.
3. Compare 11 machine learning models and evaluate ensemble strategies.
4. Assess predictive performance on a held out independent test set.
5. Interpret predictions and feature contributions using modality level Shapley attribution, SHAP, and leave one feature out (LOFO) analysis.
6. Examine how nitrogen rate, planting density, and variety affect feature values and feature relationships with Pn.

Feature selection, model development, and tuning use the training data only. The independent test set is reserved for final performance evaluation.

## Reproducibility

The manuscript reports model development in Python 3.11. Add the tested dependency versions, environment file, data configuration, and exact commands for running feature extraction, model training, evaluation, and interpretation alongside the released code. These instructions should match the scripts and configuration files distributed with this repository.

## Citation

Please cite the associated published manuscript. Add the final author list, journal details, and DOI here after publication.

## License

Add the repository license and any separate terms that apply to the data before public release.
