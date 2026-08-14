# Nigerian Sign Language (NSL) Alphanumeric Detection Using YOLO

This repository contains materials for real-time sign recognition using the Nigerian Sign Language (NSL) Alphanumeric Dataset. The collection serves as an annotated computer vision resource to evaluate lightweight deep learning models. It includes 33 static alphanumeric gestures. The categories feature numbers 1 through 9 and letters A through Y. Motion-dependent gestures J and Z remain excluded. The composite gesture 10 is also absent. The data reflects varied indoor and outdoor lighting conditions. The images incorporate multiple camera angles and distinct background environments.

## Scope and Exclusions in data.yaml

* Included Classes: Numbers 1 to 9 and Letters A to Y.
* Excluded Classes: J, Z, and 10.
* Total Classes: 33.

## Dataset Structure

The files follow standard YOLO formatting directories.

* **images/**: Contains .jpg gesture frames across multiple subjects.
* **labels/**: Contains matching normalized YOLO .txt files.
* **dataset.yaml**: Defines the 33-class label mapping for training models.

## Intended Use Cases

* Benchmarking real-time object detectors such as YOLOv8 and YOLO11.
* Assistive technology research for deaf populations.
* Edge device deployment on platforms like Android and iOS.

## Installation and Setup

Clone the repository to your local machine.

```bash
git clone [https://github.com/adelekeadeniyan/NSL-Alphnumeric-Real-time-Sign-language-detection-using-YOLO.git](https://github.com/adelekeadeniyan/NSL-Alphnumeric-Real-time-Sign-language-detection-using-YOLO.git)
cd NSL-Alphnumeric-Real-time-Sign-language-detection-using-YOLO

## Install the requirements to ensure all dependencies function.

```bash
pip install -r requirements.txt

## If testing on Kaggle, we already Import the training dataset directly on Kaggle by calling.

```bash
dataset_path = '/kaggle/input/datasets/adelekeadeniyan/nsl-alphanumeric-dataset/content/split_dataset'

## If testing locally, download dataset from zenodo.org or Kaggle.com the links are below.

Adeleke, A. (2026). Nigerian Sign Language (NSL) Alphanumeric Dataset [Dataset]. Zenodo. (https://doi.org/10.5281/zenodo.21672976)
or
(https://www.kaggle.com/datasets/adelekeadeniyan/nsl-alphanumeric-dataset)

## Execution Instructions
You must generate the configuration file before running the model or use the attached data.yaml.E
Execute the following command to create the dataset YAML file.

```bash
python create_yaml.py

## Realtime testing Instruction
Use the detection script to test the model locally. The script reads the best.pt weights from the root directory. Ensure your trained best.pt file resides in the folder runs\detect\train_yolo11n\weights before execution, or adjust the location in the code.

```bash
python realtime_detect.py

Note: if testing locally, you may need to adjust folder paths. I run the code on Kaggle because of resource constraints and later test it in real time on my local machine using the realtime_detect.py

## Dataset Download and Citation
The dataset resides on Zenodo for public download. Please apply the following citation when referencing this data in academic contexts.
Adeleke, A. (2026). Nigerian Sign Language (NSL) Alphanumeric Dataset [Dataset]. Zenodo. (https://doi.org/10.5281/zenodo.21672976)
