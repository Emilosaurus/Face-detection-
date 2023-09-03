# Face Recognition with OpenCV and dlib

This Python script demonstrates face recognition using OpenCV and dlib. It captures live video from the default camera, detects faces, and matches them against a set of known faces stored in the 'faces' folder. The script also logs the recognized faces and the time of recognition in a CSV file.

![Face Recognition Demo](demo.gif)

## Features

- Real-time face detection and recognition.
- Confidence level displayed for recognized faces.
- Logging of recognized faces with timestamps in a CSV file.
- Supports multiple known faces.

## Prerequisites

Before running the script, make sure you have the following installed:

- Python (3.6 or higher)
- OpenCV with GPU support
- dlib with GPU support
- face_recognition library
- CUDA and cuDNN (for GPU acceleration)

You can install the required libraries using `pip`. For GPU support, ensure you have the necessary GPU drivers and libraries installed.

```bash
pip install opencv-python-headless opencv-contrib-python-headless
pip install dlib-gpu
pip install face_recognition
