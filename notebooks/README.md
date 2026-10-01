# Notebooks Directory

This directory contains interactive Jupyter and Google Colab notebooks utilized for exploratory data analysis, dataset partitioning, and model training workflows.

## Notebook Overviews

* **`train.ipynb`**: The primary end-to-end training pipeline configured for YOLO11n on the 33-class alphanumeric dataset using an NVIDIA Tesla T4 GPU backend.
* **`gathering_data.ipynb`**: Utilities and scripts used during the initial multi-device acquisition phase across diverse indoor backgrounds and lighting conditions.
* **`nsl-alphanumeric-dataset-code.ipynb`**: Data curation, cleaning, and signer-independent splitting logic (S1–S3 training, S4 validation, S5 testing) designed to prevent data leakage.
