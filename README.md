# computer-Vision-
# 🫁 TB Chest X-Ray Classification Using PyTorch

A Computer Vision project for classifying chest X-ray images into **Normal** and **Tuberculosis (TB)** using Deep Learning with **PyTorch**.

The project compares two approaches:

1. **CNN trained from scratch**
2. **Pretrained ResNet-18 fine-tuned for TB classification**

Weights & Biases (**W&B**) is used to monitor and track the training experiments.

---

## 📌 Project Overview

Tuberculosis is an infectious disease that can affect the lungs. Chest X-ray imaging is commonly used as part of the clinical evaluation of pulmonary abnormalities.

In this educational Computer Vision project, a labeled chest X-ray dataset is used to build a binary image classification model.

> **Important:** This project is for educational and machine-learning purposes. It is not intended to provide medical diagnosis or replace professional clinical evaluation.

---

## 📂 Dataset

**TB Chest X-Ray Dataset**

Source:

https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset

### Classes

* `Normal`
* `Tuberculosis`

The dataset is divided using a stratified:

* **70% Training**
* **15% Validation**
* **15% Testing**

Stratification is used to preserve the class distribution across the three subsets.

---

## 🛠️ Technologies & Libraries

* Python
* PyTorch
* TorchVision
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* KaggleHub
* Weights & Biases (W&B)
* Google Colab

---

## 🔄 Project Workflow

```text
TB Chest X-Ray Dataset
          │
          ▼
   Dataset Exploration
          │
          ▼
 Image Preprocessing
          │
          ▼
 Data Augmentation
          │
          ▼
 Stratified Data Split
  ┌───────┼────────┐
  ▼       ▼        ▼
Train   Validation Test
  │       │        │
  └───────┼────────┘
          ▼
   ┌───────────────┐
   │               │
   ▼               ▼
CNN From       ResNet-18
Scratch        Pretrained
   │               │
   ▼               ▼
Training        Fine-tuning
   │               │
   └───────┬───────┘
           ▼
      W&B Tracking
           │
           ▼
  Loss & Accuracy
     Comparison
           │
           ▼
 Confusion Matrices
 & Classification Reports
```

---

# 🧠 Model 1 — CNN From Scratch

A custom convolutional neural network was developed without using pretrained weights.

### Architecture

```text
Input Image
128 × 128 × 3
      │
      ▼
Conv2D 32 filters
BatchNorm
ReLU
MaxPooling
      │
      ▼
Conv2D 64 filters
BatchNorm
ReLU
MaxPooling
      │
      ▼
Conv2D 128 filters
BatchNorm
ReLU
MaxPooling
      │
      ▼
Conv2D 256 filters
BatchNorm
ReLU
MaxPooling
      │
      ▼
Adaptive Average Pooling
      │
      ▼
Dropout
      │
      ▼
Fully Connected Layer
      │
      ▼
Normal / Tuberculosis
```

---

# 🧠 Model 2 — ResNet-18 Fine-Tuning

The second approach uses **ResNet-18 pretrained on ImageNet**.

The original final classification layer is replaced with a new layer suitable for the two classes in this dataset.

```text
Input X-Ray
224 × 224 × 3
      │
      ▼
Pretrained ResNet-18
      │
      ▼
ImageNet Features
      │
      ▼
Replace Final FC Layer
      │
      ▼
Fine-Tuning
      │
      ▼
2 Classes
      │
      ├── Normal
      └── Tuberculosis
```

---

# 🖼️ Image Preprocessing

### CNN From Scratch

Images are:

* Resized to `128 × 128`
* Converted to 3 channels
* Randomly horizontally flipped
* Randomly rotated
* Converted to tensors
* Normalized

### ResNet-18

Images are:

* Resized to `224 × 224`
* Converted to 3 channels
* Randomly horizontally flipped
* Randomly rotated
* Converted to tensors
* Normalized using ImageNet normalization

---

# ⚙️ Training

Both models are trained using:

```text
Loss Function:
CrossEntropyLoss

Optimizer:
Adam

Learning Rate:
CNN: 0.001
ResNet-18: 0.0001

Batch Size:
32

Epochs:
10
```

A learning-rate scheduler is also used to reduce the learning rate when validation loss stops improving.

---

# 📊 Weights & Biases

**Weights & Biases (W&B)** is used for experiment tracking.

The following metrics are monitored:

* Training Loss
* Validation Loss
* Training Accuracy
* Validation Accuracy
* Learning Rate
* Model parameters and gradients

Example tracked metrics:

```text
Epoch
Train Loss
Validation Loss
Train Accuracy
Validation Accuracy
Learning Rate
```

W&B allows the training runs of the Scratch CNN and ResNet-18 to be monitored and compared.

**W&B Project:**
[https://wandb.ai/nouralislam1977-ai/TB-Chest-Xray-PyTorch/overview]

---

# 📈 Results

After running the experiment, the models can be compared using:

| Model                |  Test Loss | Test Accuracy |
| -------------------- | ---------: | ------------: |
| CNN From Scratch     | `[0.0678]` |   `[97.62%%`] |
| ResNet-18 Fine-Tuned | `[0.0049]` |     [99.68%]` |
ResNet-18 Test Accuracy   :


### Training Curves

The project generates:

```text
loss_comparison.png
accuracy_comparison.png
```

These visualize the training and validation performance of both models.

---

# 📊 Additional Evaluation

To provide more information than accuracy alone, the project also generates:

### Confusion Matrix

```text
confusion_matrices.png
```

### Classification Reports

The classification report includes:

* Precision
* Recall
* F1-score
* Support

This is particularly useful for an imbalanced medical-image dataset, where accuracy alone may not fully describe model performance.

---

# 📁 Project Outputs

After training, the following files are generated:

```text
best_Scratch_CNN.pth
best_ResNet18_Finetuned.pth

loss_comparison.png
accuracy_comparison.png
test_accuracy_comparison.png
confusion_matrices.png

TB_training_history.csv
TB_model_comparison.csv
TB_final_results.json
```

---

# 📂 Suggested Repository Structure

```text
TB-Chest-Xray-Classification/
│
├── README.md
│
├── TB_Chest_Xray_Classification.ipynb
│
├── models/
│   ├── best_Scratch_CNN.pth
│   └── best_ResNet18_Finetuned.pth
│
├── results/
│   ├── loss_comparison.png
│   ├── accuracy_comparison.png
│   ├── test_accuracy_comparison.png
│   ├── confusion_matrices.png
│   ├── TB_training_history.csv
│   ├── TB_model_comparison.csv
│   └── TB_final_results.json
│
└── requirements.txt
```

---

# 🎯 Learning Objectives

This project demonstrates practical experience with:

* Computer Vision
* Image classification
* CNN architecture design
* Convolutional layers
* Batch Normalization
* Pooling
* Dropout
* Data augmentation
* Transfer Learning
* ResNet architectures
* PyTorch
* TorchVision
* Model training and validation
* Experiment tracking
* Weights & Biases
* Model evaluation
* Confusion matrices
* Classification reports
* Data visualization

---

# 🚀 Future Improvements

Possible extensions include:

* ROC-AUC analysis
* Sensitivity and specificity
* Class-weighted loss
* Weighted sampling
* More advanced data augmentation
* Early stopping
* Hyperparameter tuning
* Grad-CAM visualization
* Comparing additional pretrained architectures
* Explainable AI for chest X-ray classification

---

# ⚠️ Disclaimer

This project is intended for **educational and research purposes only**.

The model should not be used as a standalone medical diagnostic system. Predictions from machine-learning models require appropriate clinical validation and should not replace evaluation by qualified healthcare professionals.

---

## 👩‍💻 Author

**Dr. Ghada Elfeki**

PhD in Biochemistry | Molecular Biology | Data Analysis | Machine Learning | Deep Learning | Computer Vision

GitHub: [GhadaElfeki77 ]

LinkedIn: [www.linkedin.com/in/ghada-elfeki-b5b6692a6]
Kaggle: [https://www.kaggle.com/code/ghadaelfeki/notebookgh-tuberculosis-tb-chest-x-ray-detection]
