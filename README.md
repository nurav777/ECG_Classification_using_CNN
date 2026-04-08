# ECG Classification Models - Performance Comparison

---

## Overview

This repository contains three ECG classification models with varying complexity, all trained on MIT-BIH Arrhythmia Dataset. Models range from ultra-lightweight (3K parameters) to heavyweight (7M parameters) for different deployment scenarios.

---

## Dataset

**Source**: MIT-BIH Arrhythmia Database (Kaggle CSV)

- **Input**: 187 ECG time-series samples per record
- **Task**: Binary classification - Normal (0) vs Abnormal (1-4)
- **Original Distribution**: 82.8% normal, 17.2% abnormal
- **Training Samples**: 144,942 (after SMOTE balancing)

---

## Model Comparison

| Model         | Parameters | Model Size | Test Accuracy | Test Loss | Key Features                                      |
| ------------- | ---------- | ---------- | ------------- | --------- | ------------------------------------------------- |
| **3K Model**  | 3,969      | 15.5 KB    | **97.07%**    | 0.0942    | SMOTE, GlobalAveragePooling1D, BatchNormalization |
| **46K Model** | 46,785     | 182.75 KB  | **97.56%**    | 0.1098    | 2 Conv1D layers, Class weights, Dropout           |
| **7M Model**  | 7,157,633  | 27.30 MB   | **98.85%**    | 0.1131    | Deep ResNet, Multi-scale, SE attention, SMOTE     |

---

## Model Details

### 🚀 **3K Parameters Model** (`ecg_3k_params.ipynb`)

**Best for**: Edge devices, microcontrollers, real-time monitoring

**Architecture**:

- Input: (187, 1) ECG signals
- Conv1D(16) → BatchNorm → MaxPool1D
- Conv1D(32) → BatchNorm → MaxPool1D
- GlobalAveragePooling1D
- Dense(32) → Dropout(0.3) → Dense(1)

**Performance**:

- Normal class: 97% precision, 99% recall
- Anomaly class: 95% precision, 88% recall
- F1-score: 0.97

**Deployment**:

- TFLite float16: 16.0 KB (98.90% accuracy)
- Inference: <5ms on Cortex-M4
- Power: <0.1 mJ per inference

---

### ⚖️ **46K Parameters Model** (`ecg_46k_params.ipynb`)

**Best for**: Mobile applications, moderate resource devices

**Architecture**:

- Input: (187, 1) ECG signals
- Conv1D(16, kernel_size=5) → MaxPool1D
- Conv1D(32, kernel_size=5) → MaxPool1D
- Flatten → Dense(32) → Dropout(0.3) → Dense(1)

**Performance**:

- Normal class: 98% precision, 99% recall
- Anomaly class: 95% precision, 90% recall
- F1-score: 0.98

**Features**:

- Class weights for imbalance handling
- Kaggle-compatible data loading
- TFLite export ready

---

### 🏋️ **7M Parameters Model** (`ecg_7M_params.ipynb`)

**Best for**: High-performance servers, research, maximum accuracy

**Architecture**:

- Multi-scale feature extraction (3,5,7 kernel sizes)
- Residual blocks with Squeeze-and-Excitation attention
- Deep ResNet architecture (64→128→256→512 channels)
- GlobalAveragePooling + GlobalMaxPooling
- Dense(512→256→128) → Dense(1)

**Performance**:

- Normal class: 99% precision, 99% recall
- Anomaly class: 94% precision, 96% recall
- F1-score: 0.98

**Features**:

- Advanced training techniques (learning rate scheduling)
- SMOTE data balancing
- State-of-the-art architecture

---

## Key Insights

### 📊 **Performance vs Complexity**

- **3K → 46K**: +0.49% accuracy for 12x parameter increase
- **46K → 7M**: +1.29% accuracy for 153x parameter increase
- **Diminishing returns** beyond 46K parameters

### 💡 **Recommendations**

- **Edge/Mobile**: Use 3K model - best efficiency-to-performance ratio
- **Web/Desktop**: Use 46K model - balanced approach
- **Research/Cloud**: Use 7M model - maximum accuracy

### 🎯 **Deployment Scenarios**

| Scenario         | Recommended Model | Reason                                |
| ---------------- | ----------------- | ------------------------------------- |
| Wearable devices | 3K Model          | Ultra-low power, <16KB                |
| Mobile apps      | 46K Model         | Good accuracy, reasonable size        |
| Hospital systems | 7M Model          | Maximum accuracy, unlimited resources |
| IoT sensors      | 3K Model          | Real-time processing                  |

---

## Files Structure

```
├── ecg_3k_params.ipynb      # Ultra-lightweight model with SMOTE
├── ecg_46k_params.ipynb     # Mid-range model with class weights
├── ecg_7M_params.ipynb      # Heavy ResNet model
├── README.md                 # This file
└── output/                  # Generated model files
    ├── ecg_model_3k_params.h5
    ├── ecg_model_46k_params.h5
    ├── ecg_model_7M_params.h5
    └── *.tflite           # TensorFlow Lite exports
```

---

## Usage

### Running Models

1. **Kaggle**: Upload any notebook and run directly
2. **Local**: Install dependencies and run cells sequentially
3. **Data**: Automatically downloads from Kaggle dataset

### Model Export

Each notebook exports:

- **H5 format**: Full Keras model with weights
- **TFLite float32**: Baseline TensorFlow Lite model
- **TFLite float16**: Optimized for edge deployment

### Inference Example

```python
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="ecg_model_3k_params_float16.tflite")
interpreter.allocate_tensors()

# Predict ECG sample
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], ecg_sample)
interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]['index'])
```

---

## Performance Metrics Summary

| Metric             | 3K Model | 46K Model | 7M Model  |
| ------------------ | -------- | --------- | --------- |
| **Accuracy**       | 97.07%   | 97.56%    | 98.85%    |
| **Precision**      | 0.96     | 0.97      | 0.97      |
| **Recall**         | 0.94     | 0.95      | 0.98      |
| **F1-Score**       | 0.95     | 0.96      | 0.98      |
| **Model Size**     | 15.5 KB  | 182.75 KB | 27.30 MB  |
| **Parameters**     | 3,969    | 46,785    | 7,157,633 |
| **Inference Time** | <5ms     | ~15ms     | ~100ms    |

---

## Conclusion

This project demonstrates trade-offs between model complexity and performance in ECG classification:

- **3K model** achieves 97% accuracy with minimal resources - ideal for edge deployment
- **46K model** provides balanced performance for mobile applications
- **7M model** reaches 99% accuracy for research/clinical use

The **3K parameter model** offers the best efficiency-to-performance ratio, making it ideal for real-world edge deployment in wearable ECG monitoring devices.

---

_All models are production-ready with TFLite export for deployment across platforms._
