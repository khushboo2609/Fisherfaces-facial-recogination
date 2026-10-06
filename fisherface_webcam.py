import cv2
import os


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "trainer/fisherface_model.yml"

FACE_SIZE = (64, 64)

# Fisherfaces returns a distance.
# Lower distance generally means a better match.
DISTANCE_THRESHOLD = 1000


# ============================================================
# LOAD FISHERFACE MODEL
# ============================================================

print("=" * 70)
print("FISHERFACES WEBCAM RECOGNITION")
print("=" * 70)

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Fisherface model not found!")
    print(f"Expected path: {MODEL_PATH}")
    exit()

recognizer = cv2.face.FisherFaceRecognizer_create()

recognizer.read(MODEL_PATH)

print("\nFisherface model loaded successfully!")


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

CASCADE_PATH = "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    print("\nERROR: Haar Cascade could not be loaded!")
    print(f"Expected file: {CASCADE_PATH}")
    exit()

print("Haar Cascade loaded successfully.")

# ============================================================
# OPEN WEBCAM
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("\nERROR: Could not open webcam.")
    exit()

print("\nWebcam opened successfully.")

print("\nControls:")
print("Q - Quit")
print("=" * 70)


# ============================================================
# WEBCAM LOOP
# ============================================================

while True:

    ret, frame = camera.read()

    if not ret:
        print("Could not read frame.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        # Extract face
        face = gray[
            y:y + h,
            x:x + w
        ]

        # Resize to Fisherface input size
        face = cv2.resize(
            face,
            FACE_SIZE
        )

        # Predict identity
        label, distance = recognizer.predict(face)

        # Determine recognition result
        if distance < DISTANCE_THRESHOLD:

            name = f"Person {label}"

            result = "MATCH"

        else:

            name = "UNKNOWN"

            result = "UNKNOWN"

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Display identity
        cv2.putText(
            frame,
            f"{result}: {name}",
            (x, y - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Display distance
        cv2.putText(
            frame,
            f"Distance: {distance:.2f}",
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

    # Display number of detected faces
    cv2.putText(
        frame,
        f"Faces detected: {len(faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Fisherfaces Webcam Recognition",
        frame
    )

    # Quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print("\nQ pressed.")
        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()
cv2.destroyAllWindows()

print("\n" + "=" * 70)
print("FISHERFACES WEBCAM RECOGNITION STOPPED")
print("=" * 70)