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

## 📊 Dataset

### Dataset Source

This project uses the **Brain Tumor MRI Dataset** by **Masoud Nickparvar**, obtained from Kaggle.

- **Source:** Kaggle
- **Dataset:** Brain Tumor MRI Dataset
- **Creator:** Masoud Nickparvar
- **Task:** Multi-class brain MRI image classification
- **Total Images:** 7,200
- **Training Images:** 5,600
- **Testing Images:** 1,600
- **Number of Classes:** 4

### Classes

| Class | Description |
|---|---|
| Glioma | Brain tumor originating from glial cells |
| Meningioma | Tumor originating from the meninges |
| Pituitary | Tumor involving the pituitary gland |
| No Tumor | MRI images classified without a tumor |

### Dataset Origin

The dataset was obtained from **Kaggle**. Kaggle is the hosting/source platform from which the dataset was downloaded for this project.

The available dataset documentation does not clearly specify a single country or hospital as the origin of all MRI images. Therefore, a specific country of origin is not assumed in this project.

### Dataset Usage

The dataset is used for **educational and research purposes** to develop and evaluate a deep-learning image-classification model.

The dataset is suitable for:
- Brain MRI image classification
- Computer vision experiments
- Deep learning model development
- Model evaluation and explainability experiments such as Grad-CAM

However, this dataset and the resulting model should **not be considered clinically validated for medical diagnosis**.

### Important Limitation

The project uses MRI image data rather than detailed patient metadata such as patient history, age, gender, hospital information, or other clinical information.

Therefore, the model's predictions should be considered experimental and educational rather than a substitute for evaluation by a qualified medical professional.

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

## 🤖 How Does the Model Make Predictions?

The system uses a **ResNet50-based Convolutional Neural Network (CNN)** trained to classify brain MRI images into four categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary

### Prediction Pipeline

The prediction process follows these steps:

**1. MRI Image Upload**

The user uploads a brain MRI image through the Streamlit web application.

**2. Image Preprocessing**

The uploaded image is:

- Converted to RGB format
- Resized to **224 × 224 pixels**
- Converted into a numerical array
- Normalized by dividing pixel values by 255

This converts the image into a format suitable for the neural network.

**3. ResNet50 Feature Extraction**

The processed MRI image is passed through the trained **ResNet50 architecture**.

The convolutional layers automatically learn visual features from the MRI, such as edges, textures, shapes, and higher-level patterns.

**4. Classification**

The final layers of the model use the extracted features to calculate a probability for each of the four classes.

For example:

```text
Glioma       →  0.83%
Meningioma   → 33.37%
No Tumor     → 64.26%
Pituitary    →  1.54%

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