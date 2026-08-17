import cv2 as cv
import numpy as np
import os

detector_path = "weights/face_detection_yunet_2026may.onnx"
recognizer_path = "weights/face_recognition_sface_2021dec.onnx"
dataset_path = "test_images"             #TODO: Change to known_faces
encoding_path = "test_encodings"         #TODO: Change to encodings


detector = cv.FaceDetectorYN.create(
    detector_path,
    "",
    (320, 320),
    0.8,
)

recognizer = cv.FaceRecognizerSF.create(
    recognizer_path,
    ""
)

count = 0 

if os.path.exists(dataset_path):
    for folder in os.listdir(dataset_path):
        full_folder_path = os.path.join(dataset_path, folder)

        if os.path.isdir(full_folder_path):
            os.mkdir(os.path.join(encoding_path, folder))

            for image in os.listdir(full_folder_path):

                if image.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_image_path = os.path.join(full_folder_path, image)
                    npImage = cv.imread(full_image_path)
                    h, w, _ = npImage.shape
                    detector.setInputSize((w, h))
                    retval, detectedFace = detector.detect(npImage)

                    if detectedFace is None:
                        print(f"{full_image_path}: No face detected")

                    elif len(detectedFace) == 1:
                        print(f"{full_image_path}: 1 face detected")
                        feature = recognizer.feature(recognizer.alignCrop(npImage, detectedFace))
                        np.save(f"{encoding_path}/{folder}/{count}.npy", feature)
                        count += 1
                        
                    else:
                        print(f"{full_image_path}: {len(detectedFace)} faces detected")

else:
    print("Dataset path not found")