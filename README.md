# ZeroBlur

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org/)
[![Android](https://img.shields.io/badge/Android-API%2021+-green.svg)](https://developer.android.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Real-time camera blur detection and correction system with on-device ML inference**

ZeroBlur is an intelligent camera application that eliminates motion blur in real-time by dynamically adjusting camera settings using computer vision and machine learning. The system processes frames every 40ms and provides sub-1 second capture confirmation, ensuring crisp, sharp photos without requiring external APIs.

## Key Features

- **Real-time Blur Detection**: Custom CNN classifier with 94%+ accuracy on real-world images
- **Dynamic Camera Control**: Automatic ISO and shutter speed adjustment based on scene analysis
- **Low Latency Processing**: 40ms frame processing with real-time feedback loop
- **On-Device Inference**: Fully offline processing using TensorFlow Lite
- **Android Integration**: Native mobile app with intuitive camera interface
- **Sub-1 Second Response**: Instant capture confirmation and blur assessment

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Camera Feed   │───▶│  Frame Analysis  │───▶│  CNN Classifier │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Camera Controls │◄───│ Feedback System  │◄───│ Blur Prediction │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Model Details

### CNN Architecture
- **Input Size**: 128x128x3 RGB images
- **Architecture**: Deep CNN with batch normalization and dropout
- **Layers**: 4 convolutional blocks + 3 dense layers
- **Optimization**: TensorFlow Lite quantization (FP16)
- **Model Size**: Optimized for mobile deployment

### Training Dataset
- **Source**: Motion blur dataset with augmented video frames
- **Classes**: Binary classification (blur/non-blur)
- **Augmentation**: Varying motion-blur levels for robustness
- **Performance**: 94%+ accuracy on real-world test images

## Tech Stack

### Mobile Development
- **Platform**: Android (API 21+)
- **Language**: Java
- **IDE**: Android Studio
- **ML Framework**: TensorFlow Lite

### Computer Vision
- **Framework**: OpenCV, TensorFlow
- **Languages**: Python, Java
- **Algorithms**: Optical flow, Laplacian variance, DCT analysis

## Installation

### Prerequisites
```bash
# Python dependencies
pip install tensorflow opencv-python numpy scikit-image kagglehub

# Android development
# Install Android Studio and SDK
```

### Quick Start
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ZeroBlur.git
   cd ZeroBlur
   ```

2. **Install the Android app**
   ```bash
   # Install the pre-built APK
   adb install ZeroBlur.apk
   ```

3. **Run the Python pipeline** (for testing)
   ```bash
   python pipelinepython.py
   ```

## Project Structure

```
ZeroBlur/
├── ZeroBlur.apk              # Android application
├── blur_classifier.tflite    # Trained TensorFlow Lite model
├── blur_classifier.ipynb     # Model training notebook
├── blurdetect.py            # Advanced blur detection algorithms
├── pipelinepython.py        # Real-time processing pipeline
├── testingpy.py             # Testing utilities
├── Android Code.zip         # Android source code
└── README.md                # This file
```

## How It Works

### 1. Frame Capture & Analysis
```python
# Extract features from each frame
features = extract_image_features(frame, prev_gray)
- Blur score (Laplacian variance)
- Motion detection (optical flow)
- Edge density analysis
- Brightness assessment
```

### 2. CNN Blur Classification
```python
# Real-time inference using TensorFlow Lite
interpreter.set_tensor(input_details[0]['index'], frame)
interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]['index'])
is_blurry = prediction[0] > 0.5
```

### 3. Dynamic Camera Adjustment
```python
# Feedback loop for optimal settings
if motion_detected:
    increase_shutter_speed()
if brightness < threshold:
    adjust_iso()
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Frame Processing Time** | 40ms |
| **Model Accuracy** | 94%+ |
| **Capture Confirmation** | < 1 second |
| **Model Size** | Optimized for mobile |
| **Offline Processing** | 100% on-device |

## Configuration

### Camera Settings
- **Shutter Speed Range**: Automatic adjustment based on motion
- **ISO Range**: Dynamic based on lighting conditions
- **Focus Mode**: Continuous autofocus with blur feedback

### Model Parameters
```python
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
BLUR_THRESHOLD = 200  # Laplacian variance threshold
FLOW_THRESHOLD = 1.0  # Motion detection sensitivity
```

## Use Cases

- **Photography**: Eliminate motion blur in handheld shots
- **Document Scanning**: Ensure text clarity and readability
- **Quality Control**: Automated image quality assessment
- **Sports Photography**: Real-time blur detection for action shots
- **Mobile Photography**: Enhanced camera experience

## Future Enhancements

- [ ] **Multi-class Blur Types**: Classify different blur types (motion, defocus, gaussian)
- [ ] **Advanced Stabilization**: Integration with gyroscope data
- [ ] **Low-light Optimization**: Enhanced performance in poor lighting
- [ ] **Video Mode**: Real-time video stabilization and blur correction
- [ ] **Cloud Sync**: Optional cloud-based model updates

---
