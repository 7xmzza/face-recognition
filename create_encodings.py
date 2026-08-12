import cv2
import numpy as np
import os
from PIL import Image

detector_path = "weights/face_detection_yunet_2026may.onnx"
recognizer_path = "weights/face_recognition_sface_2021dec.onnx"
folder_path = "knownFaces"

detector = cv2.FaceDetectorYN.create(
    detector_path,
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)
recognizer = cv2.FaceRecognizerSF.create(recognizer_path, "")

count = 1 

if os.path.exists(folder_path):
    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".png")):
            full_path = f"{folder_path}/{file}"

            with Image.open(full_path) as img:
                npImage = cv2.imread(full_path)
                h, w, _ = npImage.shape
                detector.setInputSize((w,h))
                retval, detectedFace = detector.detect(npImage)
                if detectedFace is None:
                    print(f"{file}: No face detected")
                elif len(detectedFace) == 1:
                    print(f"{file}: 1 face detected")
                else:
                    print(f"{file}: {len(detectedFace)} faces detected")
                if detectedFace is not None and len(detectedFace) == 1:
                    feature = recognizer.feature(recognizer.alignCrop(npImage, detectedFace))
                    np.save(f"encodings/{count}.npy", feature)
                    count += 1
