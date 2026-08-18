import os

import cv2 as cv
import numpy as np


DETECTOR_PATH = "weights/face_detection_yunet_2026may.onnx"
RECOGNIZER_PATH = "weights/face_recognition_sface_2021dec.onnx"

# TODO: Change this to the relative path of the folder containing
# encoding folders.
ENCODING_FOLDER_PATH = "production_material/test_encodings"


encodings = {}

for person in os.listdir(ENCODING_FOLDER_PATH):
    encodings[person] = []

    person_path = os.path.join(
        ENCODING_FOLDER_PATH,
        person,
    )

    for encoding in os.listdir(person_path):
        encoding_path = os.path.join(
            person_path,
            encoding,
        )

        np_encoding = np.load(encoding_path)
        encodings[person].append(np_encoding)


# Detector setup.
# The resolution can be changed as long as the camera resolution
# is changed accordingly.
detector = cv.FaceDetectorYN.create(
    DETECTOR_PATH,
    "",
    (640, 480),
)

# Recognizer setup.
recognizer = cv.FaceRecognizerSF.create(
    RECOGNIZER_PATH,
    "",
)


camera = cv.VideoCapture(0)

if not camera.isOpened():
    print("Camera cannot be opened")
    exit()

print("Camera opened successfully")


try:
    while True:
        ret, frame = camera.read()

        if not ret:
            print("Cannot receive frame")
            break

        frame = cv.flip(frame, 1)

        _, faces_detected = detector.detect(frame)

        if faces_detected is not None:
            for face in faces_detected:
                aligned_face = recognizer.alignCrop(
                    frame,
                    face,
                )

                feature = recognizer.feature(aligned_face)

                x, y, w, h = map(int, face[:4])

                cv.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2,
                )

                for person in encodings:
                    for encoding in encodings[person]:
                        score = recognizer.match(
                            feature,
                            encoding,
                            cv.FaceRecognizerSF_FR_COSINE,
                        )

                        if score > 0.7:
                            print(
                                f"Recognized face: {person}"
                            )

                            cv.putText(
                                frame,
                                person,
                                (x, y - 15),
                                cv.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2,
                                cv.LINE_AA,
                            )

        cv.imshow(
            "Face detection and recognition - Hamza ElNahtawy",
            frame,
        )

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv.destroyAllWindows()