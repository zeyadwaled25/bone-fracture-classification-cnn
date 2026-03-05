<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,50:16213e,100:0f3460&height=200&section=header&text=Bone%20Fracture%20Classification&fontSize=38&fontColor=e94560&animation=fadeIn&fontAlignY=38&desc=CNN-powered%20X-ray%20Fracture%20Detection&descAlignY=58&descAlign=50&descColor=a8dadc" width="100%"/>

<br/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&pause=1000&color=E94560&center=true&vCenter=true&width=600&lines=Deep+Learning+%7C+Medical+Imaging;CNN+%7C+TensorFlow+%7C+Streamlit;Automated+Fracture+Detection+from+X-rays)](https://git.io/typing-svg)

<br/>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

</div>

---

## 🩻 Project Overview

A deep learning pipeline that **automatically detects bone fractures** from X-ray images using a custom Convolutional Neural Network (CNN).  
The model classifies each image into one of two categories:

| Class | Description |
|-------|-------------|
| 🔴 **Fractured** | X-ray shows signs of a bone fracture |
| 🟢 **Normal** | X-ray shows no signs of fracture |

**Key capabilities:**

- 📐 CLAHE-based contrast enhancement for X-ray preprocessing
- 🧠 Custom multi-layer CNN with BatchNormalization & Dropout
- 📊 Full evaluation suite — accuracy, precision, recall, AUC, confusion matrix, ROC curve
- 🖥️ Interactive Streamlit web app for real-time inference

---

## 🔄 Pipeline

```
 X-ray Image
     │
     ▼
┌─────────────────────────┐
│  Preprocessing (CLAHE)  │  Grayscale → Resize 224×224 → Contrast enhance → Normalize
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│   Data Augmentation     │  Rotation · Zoom · Shift · Horizontal Flip
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│      CNN Model          │  Conv → BN → ReLU → MaxPool → Dropout  (×3)
│                         │  GlobalAveragePooling → Dense → Softmax
└─────────────────────────┘
     │
     ▼
  Prediction
  Fractured / Normal + Confidence %
```

---

## 🏗️ Model Architecture

The model is a Sequential CNN with three convolutional blocks followed by a classification head:

```
Input (224 × 224 × 1 — grayscale)
│
├── Block 1 ─ Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool2D → Dropout(0.25)
│
├── Block 2 ─ Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool2D → Dropout(0.25)
│
├── Block 3 ─ Conv2D(128, 3×3) → BatchNorm → ReLU → MaxPool2D → Dropout(0.30)
│
├── GlobalAveragePooling2D
│
├── Dense(256) → BatchNorm → ReLU → Dropout(0.5)
│
└── Dense(2) → Softmax  ──►  [P(Fractured), P(Normal)]
```

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam (lr = 1e-4) |
| Loss | Categorical Cross-Entropy |
| Metrics | Accuracy · Precision · Recall · AUC |
| Epochs | 50 (EarlyStopping, patience=12) |
| Batch Size | 32 |
| Input Size | 224 × 224 (grayscale) |
| Saved Model | `model_bone.h5` |

---

## 📦 Dataset

Medical X-ray images organized into three splits:

```
dataset/
├── train/
│   ├── fractured/
│   └── normal/
├── val/
│   ├── fractured/
│   └── normal/
└── test/
    ├── fractured/
    └── normal/
```

> ⚠️ The dataset is **not included** in this repository due to size constraints.

**Preprocessing applied to every split:**

1. Convert to grayscale
2. Resize to **224 × 224**
3. Apply **CLAHE** (Contrast Limited Adaptive Histogram Equalization, 8×8 tile grid)
4. Normalize pixel values to **[0, 1]**

**Training-only augmentation:**

- Random rotation ± 15°
- Zoom ± 10%
- Width / Height shift ± 5%
- Horizontal flip

---

## 🖥️ Streamlit Web App (`app.py`)

An interactive web interface for real-time bone fracture detection:

```
┌──────────────────────────────────────────┐
│  🦴 Bone Fracture Detection              │
│                                          │
│  [ Upload X-ray image (jpg/png) ]        │
│                                          │
│  ┌────────────────────────────────────┐  │
│  │         Uploaded X-ray             │  │
│  └────────────────────────────────────┘  │
│                                          │
│  ✅ Prediction: Normal                   │
│  ℹ️  Confidence: 94.27%                  │
│                                          │
│  Fractured  ████░░░░░░  0.06            │
│  Normal     ██████████  0.94            │
└──────────────────────────────────────────┘
```

**Features:**
- Upload any JPG / PNG X-ray image
- Automatic preprocessing (grayscale → CLAHE → normalize)
- Displays prediction label with color-coded result
- Shows per-class confidence scores

---

## 📊 Evaluation

The notebook provides a comprehensive evaluation suite:

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Precision | Fraction of fracture detections that are real |
| Recall | Fraction of actual fractures correctly detected |
| AUC | Area under the ROC curve |
| Confusion Matrix | Heatmap of TP / TN / FP / FN |
| ROC Curve | True-positive vs false-positive trade-off |

Training curves (accuracy & loss) are plotted for both the training and validation sets across all epochs.

---

## 📁 Project Structure

```
bone-fracture-classification-cnn/
│
├── app.py                                   # Streamlit inference app
├── bone_fracture_classification.ipynb       # Full training & evaluation notebook
├── requirements.txt                         # Python dependencies
├── docs/
│   ├── Report_DL.pdf                        # Project report
│   └── vertopal.com_bone_fracture_          # Documentation PDF
│       classification_documentation.pdf
│
├── dataset/                                 # (excluded — not in repo)
│   ├── train/
│   ├── val/
│   └── test/
│
└── model_bone.h5                            # (excluded — saved model weights)
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/zeyadwaled25/bone-fracture-classification-cnn.git
cd bone-fracture-classification-cnn
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your trained model

Place `model_bone.h5` in the project root (train it via the notebook or obtain it separately).

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser, upload an X-ray image, and get instant predictions.

### 5. (Optional) Re-train the model

Open and run `bone_fracture_classification.ipynb` in Jupyter or Google Colab with your dataset under `dataset/`.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| **TensorFlow / Keras** | CNN model definition, training, inference |
| **OpenCV** | Image loading, resizing, CLAHE preprocessing |
| **NumPy** | Array manipulation |
| **Matplotlib / Seaborn** | Training curves, confusion matrix, ROC curve |
| **scikit-learn** | Class weights, metrics, classification report |
| **Streamlit** | Interactive web app for real-time inference |

---

## 👤 Author

<div align="center">

**Zeyad Waled**  
*Machine Learning Engineer | AI Engineer*

[![GitHub](https://img.shields.io/badge/GitHub-zeyadwaled25-181717?style=for-the-badge&logo=github)](https://github.com/zeyadwaled25)

</div>

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f3460,50:16213e,100:1a1a2e&height=120&section=footer" width="100%"/>

</div>
