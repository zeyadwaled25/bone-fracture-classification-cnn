# Bone Fracture Classification using Deep Learning

A Convolutional Neural Network (CNN) model for automatic bone fracture detection from X-ray images.

---

## Project Overview

This project implements a deep learning pipeline for classifying X-ray images into:

- Fractured
- Non-Fractured

The system includes:

- Data preprocessing
- CNN model training
- Model evaluation
- Inference pipeline using saved model
- Simple deployment script (app.py)

---

## Dataset

Medical X-ray images organized into:

dataset/
├── train/
├── validation/
├── test/

Images are resized and normalized before training.

---

## Model Architecture

- Convolutional layers for feature extraction
- MaxPooling layers
- Fully connected layers
- Softmax / Sigmoid output layer

Saved Model:

- model_bone.h5 (excluded from repo due to size)

---

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV

---

## How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run inference app:

```
python app.py
```

---

## Project Structure

```
├── app.py
├── bone_fracture_classification.py
├── bone_fracture_classification.ipynb
├── requirements.txt
├── docs/
├── dataset/ (excluded)
├── model_bone.h5 (excluded)
```

---

## Results

- Achieved high validation accuracy
- Stable performance on unseen test images
- Real-time inference capability

---

## Author

Zeyad Waled  
Machine Learning Engineer | AI Engineer
