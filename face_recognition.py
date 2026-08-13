import cv2
import numpy as np
import os

detector_path = "weights/face_detection_yunet_2026may.onnx"
recognizer_path = "weights/face_recognition_sface_2021dec.onnx"

detector = cv2.FaceDetectorYN.create(detector_path, "", (640,480))
recognizer = cv2.FaceRecognizerSF.create(recognizer_path, "")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera cannot be opened")
    exit()

print("Webcam opened succesfully")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        print("Cannot receive frame")
        break

    retval, facesDetected = detector.detect(frame)

    if facesDetected is not None:
        for face in facesDetected:
            alignedFace = recognizer.alignCrop(frame, face)
            feature = recognizer.feature(alignedFace)
            for encoding in os.listdir("encodings"):
                score = recognizer.match(feature, np.load(f"encodings/{encoding}"), cv2.FaceRecognizerSF_FR_COSINE)
                if score > 0.7:
                    print("Recognized face")
            x, y, w, h = map(int, face[0:4])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)


    cv2.imshow('Face detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()