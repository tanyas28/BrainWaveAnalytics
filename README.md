# BrainWaveAnalytics
BrainWaveAnalytics 🧠⚡ is an AI-powered deep learning model designed to detect epilepsy and schizophrenia from spectrogram images derived from EEG signals. It utilizes Discrete Wavelet Transform (DWT) for feature extraction and LSTM networks for classification, achieving high accuracy and strong diagnostic potential.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Framework](https://img.shields.io/badge/Framework-TensorFlow-orange)

**AI-Powered Detection of Epilepsy and Schizophrenia Using Spectrogram Analysis**  

This repository contains a deep learning model that analyzes spectrogram images derived from EEG signals to detect neurological disorders such as epilepsy and schizophrenia.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
This project leverages **Discrete Wavelet Transform (DWT)** for feature extraction and **LSTM networks** to classify spectrogram images into epilepsy, schizophrenia, or healthy cases. The model achieves **92% test accuracy** and a **0.98 AUC score**, demonstrating strong diagnostic potential.

---

## Features
- **Spectrogram Preprocessing**: Resizing and normalization of EEG-derived images.
- **Feature Engineering**:  
  - Wavelet-based feature extraction.  
  - Statistical features (mean, skewness, energy).  
- **Dimensionality Reduction**: PCA for efficient feature compression.  
- **LSTM Model**: Sequential deep learning for time-series pattern recognition.  
- **Interpretability**: ROC curves, training/validation accuracy/loss plots.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<tanyas28>/BrainWaveAnalytics.git
   cd NeuroSpectroDetect
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

**Example `requirements.txt`:**
```
numpy==1.21.5
tensorflow==2.8.0
opencv-python==4.5.5
scikit-learn==1.0.2
matplotlib==3.5.1
pywavelets==1.3.0
```

---

## Dataset
- **Epilepsy**: 2,000 spectrogram images from EEG signals.
- **Schizophrenia**: 2,000 spectrogram images from EEG signals.

## Methodology
### Data Preparation:
- Random sampling of 2,000 images per class.
- Train/Validation/Test split (70%/15%/15%).

### Feature Extraction:
- Wavelet transforms (DWT) + statistical metrics.
- Standardization and PCA for dimensionality reduction.

### Model Architecture:
```python
Model: "Sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm (LSTM)                  (None, 64)                33024     
dropout (Dropout)            (None, 64)                0         
dense (Dense)                (None, 32)                2080      
dropout_1 (Dropout)          (None, 32)                0         
dense_1 (Dense)              (None, 1)                 33        
=================================================================
```

---

## Results
- **Test Accuracy**: 92%
- **AUC Score**: 0.98

## Usage
### Preprocess Data:
```python
# Example from the project
from utils import process_images_in_batches
features = process_images_in_batches("epilepsy_sample", num_images=2000)
```

### Train Model:
```bash
python train.py --epochs 20 --batch_size 64
```

### Evaluate:
```python
model.evaluate(X_test, y_test)
```

---

## Contributing
Contributions are welcome! Open an issue or submit a PR for:
- Bug fixes
- Performance enhancements
- Additional neurological disorders

---
```

---

### 📌 Replace `<yourusername>`, dataset sources, and add images before publishing!

