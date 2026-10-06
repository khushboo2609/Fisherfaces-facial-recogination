from sklearn.datasets import fetch_olivetti_faces
import os
import cv2


print("=" * 60)
print("OLIVETTI FACES DATASET")
print("=" * 60)

print("\nDownloading/loading dataset...")

faces = fetch_olivetti_faces(
    shuffle=True,
    random_state=42
)

print("\nDataset loaded successfully!")

print(f"Total images : {len(faces.images)}")
print(f"Image shape  : {faces.images[0].shape}")
print(f"Total people : {len(set(faces.target))}")

dataset_path = "dataset"

print("\nCreating dataset folders...")

for person_id in range(40):

    person_folder = os.path.join(
        dataset_path,
        f"person_{person_id}"
    )

    os.makedirs(person_folder, exist_ok=True)

    count = 0

    for i, target in enumerate(faces.target):

        if target == person_id:

            image = faces.images[i]

            # Convert 0-1 floating point image
            # to 0-255 grayscale image
            image = (image * 255).astype("uint8")

            filename = os.path.join(
                person_folder,
                f"{count + 1}.jpg"
            )

            cv2.imwrite(filename, image)

            count += 1

print("\nDataset preparation completed!")

print("\nDataset structure:")
print("dataset/")
print("├── person_0/")
print("├── person_1/")
print("├── person_2/")
print("├── ...")
print("└── person_39/")

print("\nEach person should contain 10 images.")

print("=" * 60)