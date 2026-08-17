import cv2 as cv
import numpy as np
import os

DETECTOR_PATH = "weights/face_detection_yunet_2026may.onnx"
RECOGNIZER_PATH = "weights/face_recognition_sface_2021dec.onnx"
ENCODING_FOLDER_PATH = "test_encodings"


detector = cv.FaceDetectorYN.create(
    DETECTOR_PATH,
    "",
    (640,480)
)

recognizer = cv.FaceRecognizerSF.create(
    RECOGNIZER_PATH,
    ""
)

camera = cv.VideoCapture(0)

if not camera.isOpened():
    print("Camera cannot be opened")
    exit()

print("Camera opened succesfully")

try:
    while True:
        ret, frame = camera.read()
        frame = cv.flip(frame, 1)

        if not ret:
            print("Cannot receive frame")
            break

        _, faces_detected = detector.detect(frame)

        if faces_detected is not None:
            for face in faces_detected:
                aligned_face = recognizer.alignCrop(frame, face)
                feature = recognizer.feature(aligned_face)

                for person in os.listdir(ENCODING_FOLDER_PATH):
                    full_folder_path = os.path.join(ENCODING_FOLDER_PATH, person)
                    if os.path.isdir(full_folder_path):
                        for file in os.listdir(full_folder_path):

                            score = recognizer.match(
                                feature, 
                                np.load(
                                    os.path.join(
                                        ENCODING_FOLDER_PATH, 
                                        person, 
                                        file
                                    )
                                ), 
                                cv.FaceRecognizerSF_FR_COSINE
                            )

                            x, y, w, h = map(int, face[0:4])
                            cv.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
                            
                            if score > 0.7:
                                print(f"Recognized face: {person}")


                                cv.putText(frame, person, (x, y - 15), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),  2, cv.LINE_AA)


        cv.imshow('Face detection & recognition - Hamza ElNahtawy', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    camera.release()
    cv.destroyAllWindows()