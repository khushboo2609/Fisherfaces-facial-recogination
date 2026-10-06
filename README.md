# Fisherfaces Face Recognition

## 📌 Project Overview

This project implements **Fisherfaces-based face recognition** using OpenCV and the **Olivetti Faces dataset**.

Fisherfaces is a classical face-recognition technique based on **Linear Discriminant Analysis (LDA)**. Unlike Eigenfaces, which focuses on maximizing overall image variance using PCA, Fisherfaces focuses on finding features that maximize the separation between different facial identities.

The project includes:

- Dataset preparation
- Fisherfaces model training
- Train/test evaluation
- Accuracy measurement
- Precision, recall, and F1-score
- Confusion matrix
- Trained model storage
- Real-time webcam face detection and recognition

---

## 🎯 Objectives

The main objectives of this project are:

1. Understand the Fisherfaces algorithm.
2. Implement Fisherfaces using OpenCV.
3. Train the model using a multi-person face dataset.
4. Evaluate the model using unseen test images.
5. Calculate classification performance metrics.
6. Save and reload the trained Fisherfaces model.
7. Implement real-time webcam inference.
8. Understand the limitations of classical face-recognition algorithms.

---

## 🧠 What are Fisherfaces?

**Fisherfaces** is a classical face-recognition approach based on **Linear Discriminant Analysis (LDA)**.

The main objective is to find a projection that:

- Minimizes variation between images belonging to the same person.
- Maximizes separation between different people.

The general workflow is:

```text
Face Images
     ↓
Preprocessing
     ↓
PCA Dimension Reduction
     ↓
LDA / Fisher Discriminant Analysis
     ↓
Fisherface Features
     ↓
Face Classification
```

OpenCV provides the implementation through:

```python
cv2.face.FisherFaceRecognizer_create()
```

---

## 🔬 Fisherfaces vs Eigenfaces

| Feature | Eigenfaces | Fisherfaces |
|---|---|---|
| Main technique | PCA | PCA + LDA |
| Learning type | Unsupervised | Supervised |
| Main objective | Maximize variance | Maximize class separation |
| Uses class labels | No | Yes |
| Sensitive to lighting | Relatively sensitive | Generally more robust |
| Suitable for multiple identities | Yes | Yes |
| Classification | Distance-based | Distance-based |

Fisherfaces uses identity labels during training, which allows it to learn features that are more discriminative between different people.

---

## 📊 Dataset

This project uses the **Olivetti Faces dataset** provided through scikit-learn.

### Dataset characteristics

- **Total images:** 400
- **Number of subjects:** 40
- **Images per subject:** 10
- **Image size:** 64 × 64 pixels
- **Image type:** Grayscale
- **Classes:** 40

The dataset is automatically obtained using:

```python
from sklearn.datasets import fetch_olivetti_faces
```

The dataset is then converted into individual image files and organized according to subject identity.

---

## 📁 Dataset Structure

After preparation, the dataset is organized as:

```text
dataset/
│
├── person_0/
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── ...
│   └── 10.jpg
│
├── person_1/
│   ├── 1.jpg
│   ├── ...
│   └── 10.jpg
│
├── person_2/
│
├── ...
│
└── person_39/
```

Each folder represents one subject.

---

## ⚙️ Methodology

### 1. Dataset Loading

The Olivetti Faces dataset is loaded using scikit-learn.

```python
faces = fetch_olivetti_faces(
    shuffle=True,
    random_state=42
)
```

---

### 2. Image Preprocessing

Each image is:

- Converted to 8-bit grayscale.
- Resized to **64 × 64 pixels**.
- Stored with its corresponding subject label.

---

### 3. Train/Test Split

The dataset is divided into:

```text
Total images = 400

Training = 320 images
Testing  = 80 images
```

An 80/20 split is used.

Stratified splitting ensures that each subject is represented in both training and testing data.

---

### 4. Fisherface Training

The Fisherface recognizer is created using OpenCV:

```python
recognizer = cv2.face.FisherFaceRecognizer_create()
```

The training data and corresponding subject labels are then supplied to the recognizer.

---

### 5. Prediction

For each test image, the model predicts:

```text
Predicted Identity
+
Distance
```

The predicted identity corresponds to one of the 40 subjects.

---

## 📈 Model Evaluation

The trained Fisherfaces model was evaluated using the 80-image test set.

### Results

| Metric | Result |
|---|---:|
| Total images | 400 |
| Training images | 320 |
| Testing images | 80 |
| Subjects | 40 |
| Image size | 64 × 64 |
| Accuracy | **98.75%** |
| Macro F1-score | **0.99** |
| Weighted F1-score | **0.99** |

### Accuracy

The model achieved:

**98.75% accuracy**

This means that:

```text
79 / 80 test images
```

were correctly classified, while:

```text
1 / 80 test images
```

was misclassified.

---

## 📋 Classification Report

The classification report was generated using:

```python
classification_report(
    y_test,
    y_pred,
    zero_division=0
)
```

Most subjects achieved:

```text
Precision = 1.00
Recall    = 1.00
F1-score  = 1.00
```

Two classes showed reduced performance:

- **Person 2:** F1-score = 0.80
- **Person 25:** F1-score = 0.67

The overall macro and weighted F1-scores were approximately **0.99**.

---

## 🔲 Confusion Matrix

A confusion matrix was generated to examine the classification results for all 40 subjects.

```python
cm = confusion_matrix(
    y_test,
    y_pred
)
```

The matrix showed that the large majority of test samples were correctly classified.

Only one test image was incorrectly classified in the evaluated split.

---

## 💾 Model Storage

The trained Fisherfaces model is saved as:

```text
trainer/fisherface_model.yml
```

The model can later be loaded using:

```python
recognizer = cv2.face.FisherFaceRecognizer_create()

recognizer.read(
    "trainer/fisherface_model.yml"
)
```

---

## 📷 Real-Time Webcam Recognition

A webcam implementation is also included.

The webcam pipeline uses a Haar Cascade classifier for face detection and the trained Fisherfaces model for classification.

```text
Webcam
   ↓
Face Detection
   ↓
Grayscale Conversion
   ↓
Face Cropping
   ↓
64 × 64 Resize
   ↓
Fisherfaces Model
   ↓
Identity Prediction
   ↓
Display Result
```

The webcam implementation successfully:

- Opens the webcam.
- Detects faces.
- Loads the trained Fisherfaces model.
- Processes detected faces.
- Performs real-time prediction.
- Displays the predicted identity and distance.

---

## 🛠️ Technologies Used

- Python
- OpenCV
- OpenCV Contrib
- NumPy
- Scikit-learn
- Haar Cascade
- Fisherfaces
- Linear Discriminant Analysis
- PCA
- Matplotlib

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the project

```bash
cd 03_fisherfaces
```

### 3. Create a virtual environment

```bash
python -m venv fisherfaces_env
```

### 4. Activate the environment

#### Windows PowerShell

```powershell
.\fisherfaces_env\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Step 1 — Prepare the dataset

```bash
python download_dataset.py
```

This downloads the Olivetti Faces dataset and creates the required folder structure.

---

### Step 2 — Train the Fisherfaces model

```bash
python train_fisherfaces.py
```

The model will be saved to:

```text
trainer/fisherface_model.yml
```

---

### Step 3 — Run webcam recognition

```bash
python fisherface_webcam.py
```

Press:

```text
Q
```

to exit the webcam.

---

## 📂 Project Structure

```text
03_fisherfaces/
│
├── dataset/
│   ├── person_0/
│   ├── person_1/
│   ├── ...
│   └── person_39/
│
├── trainer/
│   └── fisherface_model.yml
│
├── download_dataset.py
├── train_fisherfaces.py
├── fisherface_webcam.py
├── haarcascade_frontalface_default.xml
├── requirements.txt
└── README.md
```

---

## ⚠️ Limitations

Although the model achieved 98.75% accuracy, several limitations should be considered.

### 1. Small Dataset

The dataset contains only 400 images.

A larger and more diverse dataset would provide a stronger evaluation.

### 2. Controlled Dataset

The Olivetti Faces dataset is relatively controlled compared with real-world conditions.

Performance may decrease with:

- Different lighting
- Large pose changes
- Occlusions
- Masks
- Different camera quality
- Significant facial expressions

### 3. Webcam Recognition Limitation

The webcam subject was not part of the Olivetti training dataset.

Therefore, the webcam demonstration should not be interpreted as proof that the model can recognize an arbitrary new person.

### 4. Unknown-Person Detection

The model is primarily designed to classify among the identities it was trained on.

A reliable unknown-person rejection threshold requires additional validation using identities that were not present during training.

### 5. Classical Algorithm

Fisherfaces is a classical computer-vision technique and may not perform as robustly as modern deep-learning face-recognition approaches such as:

- FaceNet
- ArcFace
- InsightFace
- DeepFace

---

## 🔮 Future Work

Possible improvements include:

1. Use a larger face dataset.
2. Add more images for each identity.
3. Test different train/test splits.
4. Perform cross-validation.
5. Add unknown-person testing.
6. Improve real-world face detection.
7. Add illumination normalization.
8. Add multi-person webcam recognition.
9. Compare Fisherfaces with Eigenfaces and LBPH.
10. Compare classical methods with FaceNet and ArcFace.
11. Implement real-time attendance recognition.
12. Add liveness detection.
13. Develop a graphical user interface.
14. Deploy the system as a web or desktop application.

---

## 📊 Conclusion

This project successfully implemented the Fisherfaces face-recognition algorithm using OpenCV and the Olivetti Faces dataset.

The dataset contained **400 images belonging to 40 subjects**, with 10 images per subject. After preprocessing, the dataset was divided into 320 training images and 80 testing images.

The trained Fisherfaces model achieved an accuracy of **98.75%** on the selected test split, with approximately **0.99 macro and weighted F1-scores**. The results demonstrate that Fisherfaces can effectively distinguish between multiple facial identities under the conditions represented in the dataset.

The project also implemented real-time webcam inference using Haar Cascade face detection and the trained Fisherfaces model.

However, the evaluation was performed on a controlled dataset, and the webcam subject was not included in the training dataset. Therefore, the reported 98.75% accuracy should be interpreted specifically as the performance on the selected Olivetti Faces test split rather than as general real-world face-recognition accuracy.

Overall, this project provides a practical understanding of classical facial-recognition techniques and establishes a foundation for comparison with more advanced methods such as **LBPH, FaceNet, DeepFace, ArcFace, and InsightFace**.

---

## 👩‍💻 Author

**Khushboo Kumari**

M.Tech — Artificial Intelligence & Data Science  
Specialization: Cyber Security

---

## ⭐ Key Result

```text
Dataset        : Olivetti Faces
Subjects       : 40
Total Images   : 400
Train Images   : 320
Test Images    : 80
Image Size     : 64 × 64
Algorithm      : Fisherfaces
Accuracy       : 98.75%
Macro F1       : 0.99
Weighted F1    : 0.99
```
