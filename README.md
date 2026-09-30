# Nigerian Sign Language (NSL) Alphanumeric Detection Using YOLO

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-2.10-red.svg)](https://pytorch.org/)

Official repository for the research paper: **"Optimizing Lightweight Convolutional Neural Networks for Real-Time Alphanumeric Sign Language Detection"**.

This repository contains materials for real-time sign recognition using the Nigerian Sign Language (NSL) Alphanumeric Dataset. The collection serves as an annotated computer vision resource to evaluate lightweight deep learning models. It includes 33 static alphanumeric gestures. The categories feature numbers 1 through 9 and letters A through Y. Motion-dependent gestures J and Z remain excluded. The composite gesture 10 is also absent. The data reflects varied indoor and outdoor lighting conditions, incorporating multiple camera angles and distinct background environments.

## Scope and Exclusions in data.yaml

* **Included Classes:** Numbers 1 to 9 and Letters A to Y.
* **Excluded Classes:** J, Z, and 10.
* **Total Classes:** 33.

## Dataset Structure

The files follow standard YOLO formatting directories:

* **images/**: Contains .jpg gesture frames across multiple subjects.
* **labels/**: Contains matching normalized YOLO .txt files.
* **data.yaml**: Defines the 33-class label mapping for training models.

## Intended Use Cases

* Benchmarking real-time object detectors such as YOLOv8 and YOLO11.
* Assistive technology research for deaf populations.
* Edge device deployment on platforms like Android and iOS.

## Hardware Deployment Benchmarks

Evaluated across physical Android environments using unquantized FP32 model weights (`best_float32.tflite`, 10.06 MB):

| Device Model | Chipset / OS | RAM | Mean Latency (ms) | FPS | RAM Usage (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Samsung SM-G991U1** | Snapdragon 888 (API 35) | 7.4 GB | 405.5 ms | 2.5 | 22.4 MB |
| **Samsung SM-A075F** | MediaTek Helio G99 (API 36) | 3.6 GB | 480.8 ms | 2.1 | 14.3 MB |
| **TECNO KM5** | MediaTek Helio G85 (API 35) | 3.7 GB | 724.8 ms | 1.4 | 16.6 MB |
| **Infinix X6517** | Unisoc SC9863A (API 31) | 3.9 GB | 1192.7 ms | 0.8 | 22.3 MB |
| **Itel A663L** | Unisoc SC9863A (API 33) | 1.9 GB | 1759.4 ms | 0.6 | 15.6 MB |

## Installation and Setup

Clone the repository to your local machine:

```bash
git clone [https://github.com/adelekeadeniyan/NSL-Alphanumeric-Real-time-Sign-language-detection-using-YOLO.git](https://github.com/adelekeadeniyan/NSL-Alphanumeric-Real-time-Sign-language-detection-using-YOLO.git)
cd NSL-Alphanumeric-Real-time-Sign-language-detection-using-YOLO
