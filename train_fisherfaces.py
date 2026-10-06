import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "dataset"
MODEL_PATH = "trainer/fisherface_model.yml"

IMAGE_SIZE = (64, 64)

print("=" * 70)
print("FISHERFACES TRAINING")
print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

images = []
labels = []

print("\nLoading dataset...")

for person_id in range(40):

    person_folder = os.path.join(
        DATASET_PATH,
        f"person_{person_id}"
    )

    for filename in os.listdir(person_folder):

        image_path = os.path.join(
            person_folder,
            filename
        )

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        image = cv2.resize(
            image,
            IMAGE_SIZE
        )

        images.append(image)
        labels.append(person_id)


images = np.array(images)
labels = np.array(labels)


print("\nDataset loaded successfully!")

print(f"Total images : {len(images)}")
print(f"Image size   : {images[0].shape}")
print(f"Total labels : {len(labels)}")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

print(f"Training images : {len(X_train)}")
print(f"Testing images  : {len(X_test)}")


# ============================================================
# CREATE FISHERFACE RECOGNIZER
# ============================================================

print("\nCreating Fisherface recognizer...")

recognizer = cv2.face.FisherFaceRecognizer_create()


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Fisherfaces model...")
print("Please wait...")

recognizer.train(
    list(X_train),
    y_train
)

print("\nTraining completed successfully!")


# ============================================================
# PREDICTION
# ============================================================

print("\nTesting model...")

y_pred = []

for image in X_test:

    predicted_label, confidence = recognizer.predict(image)

    y_pred.append(predicted_label)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print(f"\nAccuracy: {accuracy * 100:.2f}%")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

recognizer.write(MODEL_PATH)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(f"\nModel path:")
print(MODEL_PATH)

print("\nFisherfaces training completed successfully!")

print("=" * 70)