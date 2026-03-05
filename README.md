# Chest X-Ray Pneumonia Classifier (PyTorch)

## Overview

This project builds a **deep learning model using PyTorch** to classify chest X-ray images into two categories: **NORMAL** or **PNEUMONIA**. The model uses **transfer learning with the ResNet-18 architecture**, a convolutional neural network pretrained on a large image dataset.

Chest X-ray analysis is an important application of **medical image classification** in artificial intelligence. By leveraging transfer learning, the model can reuse visual features learned from large datasets and adapt them to identify pneumonia in X-ray images.

This project demonstrates a complete deep learning workflow including:

- Loading and preprocessing medical image datasets
- Applying **transfer learning** with a pretrained CNN
- Training a deep learning model using PyTorch
- Evaluating the model on test images
- Exporting predictions to a CSV file

The repository demonstrates how **deep learning models can be applied to medical imaging tasks**.

---

# Dataset

The dataset used for this project is available on Kaggle:

**Kaggle Dataset:**  
https://www.kaggle.com/datasets/munemshariarshams/chest-x-ray-dataset

Due to GitHub limitations on the number of files in a repository, the dataset is **not included directly in this repository**.

### Download Instructions

1. Download the dataset from the Kaggle link above.
2. Extract the dataset inside the project directory.
3. Ensure the dataset structure matches the following:

```
dataset/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Each folder contains chest X-ray images corresponding to the respective class.

---

# Model Architecture

This project uses **ResNet-18**, a convolutional neural network architecture designed for image recognition tasks.

ResNet (Residual Network) introduces **skip connections**, allowing deep neural networks to train more effectively.

### Model Workflow

```
Input X-Ray Image (224×224)

↓
ResNet-18 Convolutional Layers (pretrained)

↓
Feature Extraction

↓
Modified Fully Connected Layer

↓
Binary Output
NORMAL or PNEUMONIA
```

The pretrained layers are **frozen**, and only the final classification layer is trained for this binary classification task.

---

# Project Workflow

The project consists of two main stages: **training the model** and **evaluating the model**.

---

## 1. Training the Model

The training process is handled by `train.py`.

The script performs the following steps:

1. Loads the chest X-ray training dataset
2. Applies image preprocessing and resizing
3. Loads a pretrained **ResNet-18 model**
4. Replaces the final layer for binary classification
5. Trains the model on the dataset
6. Saves the trained model weights

Output produced:

```
resnet_xray_model.pth
```

This file contains the trained neural network parameters.

---

## 2. Evaluating the Model

Model evaluation is handled by `evaluate.py`.

The script performs the following steps:

1. Loads the trained model
2. Processes the test dataset
3. Generates predictions for each image
4. Saves predictions to a CSV file

Output produced:

```
outputs/predictions.csv
```

### Example Output

| image_index | predicted_label |
|-------------|----------------|
| 0 | 1 |
| 1 | 0 |
| 2 | 1 |

Where:

| Value | Meaning |
|------|------|
| 0 | NORMAL |
| 1 | PNEUMONIA |

---

# Installation and Dependencies

Install the required Python libraries using:

```
python -m pip install torch torchvision pandas numpy
```

---

# Python Libraries Used

| Library | Purpose |
|--------|--------|
| **torch** | Core deep learning framework used to build and train neural networks |
| **torchvision** | Provides pretrained models and image dataset utilities |
| **pandas** | Used to organize predictions and export them to CSV |
| **numpy** | Provides numerical computation utilities |

---

# How to Run the Project

### Step 1 — Train the Model

```
python train.py
```

This will:

- load the dataset
- train the neural network
- save the trained model

Output generated:

```
resnet_xray_model.pth
```

---

### Step 2 — Generate Predictions

```
python evaluate.py
```

This will:

- load the trained model
- evaluate the test dataset
- export predictions

Output generated:

```
outputs/predictions.csv
```

---

# Files Included

| File | Description |
|-----|-------------|
| `model.py` | Defines the ResNet-18 transfer learning model architecture |
| `train.py` | Trains the model on the chest X-ray dataset |
| `evaluate.py` | Loads the trained model and generates predictions |
| `outputs/predictions.csv` | CSV file containing model predictions |
| `resnet_xray_model.pth` | Saved trained neural network model |
| `README.md` | Project documentation |

---

# Project Objective

The objective of this project is to demonstrate how **deep learning and transfer learning can be applied to medical image classification**. By training a convolutional neural network on chest X-ray images, the model learns visual patterns that help distinguish between healthy lungs and pneumonia-infected lungs.

---

# Example Applications

Deep learning models like this can be applied to:

- **Medical image diagnostics**
- **AI-assisted radiology tools**
- **Healthcare decision support systems**
- **Automated medical screening systems**

---


