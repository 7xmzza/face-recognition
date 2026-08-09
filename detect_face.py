import cv2

model_path = "weights/face_detection_yunet_2026may.onnx"
detector = cv2.FaceDetectorYN.create(model_path, "", (640,480))
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
            x, y, w, h = map(int, face[0:4])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)

    cv2.imshow('Face detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()