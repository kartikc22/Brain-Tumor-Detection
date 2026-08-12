# 🧠 Brain Tumor Detection using ResNet50

A deep learning-based brain tumor classification system that analyzes MRI images and classifies them into four categories using a ResNet50 transfer learning model.

## 📌 Project Overview

This project uses deep learning and computer vision to classify brain MRI images into:

- Glioma
- Meningioma
- No Tumor
- Pituitary

The trained model is integrated with a Streamlit web application that allows users to upload an MRI image and receive a prediction with confidence.

Grad-CAM is also used to visualize the regions of the MRI image that influenced the model's prediction.

---

## 🚀 Features

- MRI image classification
- ResNet50 transfer learning
- Four-class tumor classification
- Prediction confidence
- Grad-CAM visualization
- Streamlit web interface
- Dataset analysis using Python
- Saved trained Keras model

---

## 🧠 Model

### Architecture

**ResNet50**

The model uses transfer learning to extract useful features from MRI images and classify them into four categories.

### Training

- Input image size: `224 × 224`
- Number of classes: `4`
- Training epochs: `20`
- Framework: TensorFlow / Keras

### Test Results

| Metric | Result |
|---|---:|
| Test Accuracy | **74.87%** |
| Test Loss | **0.7516** |

---

## 📂 Project Structure

```text
Brain-Tumor-Detection/
│
├── app.py
├── predict.py
├── gradcam.py
├── dataset_analysis.ipynb
├── requirements.txt
├── .gitignore
│
└── sample_images/
    └── test1.jpg