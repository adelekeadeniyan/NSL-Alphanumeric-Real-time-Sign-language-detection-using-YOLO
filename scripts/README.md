# Scripts Directory

This directory contains production and utility Python scripts used for data preprocessing, annotation formatting, and real-time inference deployment.

## Script Overviews

* **`extract_frames_from_video.py`**: Extracts raw video frames at a controlled sampling rate to build the foundational visual corpus for Nigerian Sign Language (NSL) data collection.
* **`generate_yolo_label.py`**: Automatically processes bounding box coordinates and formats annotations into normalized YOLO text files required for model ingestion.
* **`realtime_detect.py`**: Integrates trained YOLO11n weights (`best.pt`) to execute offline, real-time object detection and gesture classification locally.
