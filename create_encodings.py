import os

import cv2 as cv
import numpy as np


DETECTOR_PATH = "weights/face_detection_yunet_2026may.onnx"
RECOGNIZER_PATH = "weights/face_recognition_sface_2021dec.onnx"

DATASET_FOLDER_PATH = "production_material/test_images"
ENCODING_FOLDER_PATH = "production_material/test_encodings"


detector = cv.FaceDetectorYN.create(
    DETECTOR_PATH,
    "",
    (320, 320),
    0.8,
)

recognizer = cv.FaceRecognizerSF.create(
    RECOGNIZER_PATH,
    "",
)


if os.path.exists(DATASET_FOLDER_PATH):
    # Iterate through each person's folder.
    for folder in os.listdir(DATASET_FOLDER_PATH):
        full_folder_path = os.path.join(
            DATASET_FOLDER_PATH,
            folder,
        )

        if os.path.isdir(full_folder_path):
            encoding_folder_path = os.path.join(
                ENCODING_FOLDER_PATH,
                folder,
            )

            os.makedirs(
                encoding_folder_path,
                exist_ok=True,
            )

            count = 0

            # Iterate through each image in the person's folder.
            for image in os.listdir(full_folder_path):
                if not image.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                full_image_path = os.path.join(
                    full_folder_path,
                    image,
                )

                np_image = cv.imread(full_image_path)

                if np_image is None:
                    print(f"{full_image_path}: Could not read image")
                    continue

                height, width, _ = np_image.shape

                detector.setInputSize((width, height))
                _, detected_face = detector.detect(np_image)

                if detected_face is None:
                    print(f"{full_image_path}: No face detected")

                elif len(detected_face) == 1:
                    print(f"{full_image_path}: 1 face detected")

                    aligned_face = recognizer.alignCrop(
                        np_image,
                        detected_face,
                    )

                    feature = recognizer.feature(aligned_face)

                    encoding_path = os.path.join(
                        encoding_folder_path,
                        f"{folder}_{count}.npy",
                    )

                    np.save(encoding_path, feature)
                    count += 1

                else:
                    print(
                        f"{full_image_path}: "
                        f"{len(detected_face)} faces detected"
                    )

else:
    print("Dataset path not found")