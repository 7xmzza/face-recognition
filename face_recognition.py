"""
Real-time Face Recognition and Detection

This module captures video from the webcam and performs real-time face detection
and recognition against pre-generated face encodings. Detected faces are highlighted
with rectangles and recognized faces are logged to the console.

Requirements:
    - OpenCV with ONNX support
    - Pre-generated face encodings in the 'encodings' directory
    - YuNet face detection weights
    - SFace recognition weights

Controls:
    - Press 'q' to quit the application
"""

import cv2 as cv
import numpy as np
import os

# Model paths
DETECTION_MODEL_PATH = "weights/face_detection_yunet_2026may.onnx"
RECOGNITION_MODEL_PATH = "weights/face_recognition_sface_2021dec.onnx"
ENCODINGS_DIRECTORY = "encodings"  # Directory containing pre-generated face encodings

# Detection and recognition parameters
DETECTION_INPUT_SIZE = (640, 480)
MATCH_THRESHOLD = 0.7  # Confidence threshold for face recognition matches (0-1)
MATCH_METRIC = cv.FaceRecognizerSF_FR_COSINE  # Cosine similarity metric for matching

# Visualization settings
RECTANGLE_COLOR = (0, 0, 255)  # Green color for detected face rectangles (BGR format)
RECTANGLE_THICKNESS = 2  # Line thickness for face bounding boxes

# UI configuration
WINDOW_NAME = "Face Detection and Recognition"
QUIT_KEY = ord("q")  # Keyboard key to exit application

# Initialize face detection and recognition models
detector = cv.FaceDetectorYN.create(
    DETECTION_MODEL_PATH,
    "",
    DETECTION_INPUT_SIZE
)
recognizer = cv.FaceRecognizerSF.create(
    RECOGNITION_MODEL_PATH,
    ""
)

# Open camera (index 0 is the default camera)
camera = cv.VideoCapture(0)

# Verify camera initialization
if not camera.isOpened():
    print("ERROR: Unable to open camera")
    exit(1)

print("Camera opened successfully")

# Main processing loop
try:
    while True:
        # Capture frame from camera
        success, frame = camera.read()

        # Check if frame was successfully captured
        if not success:
            print("ERROR: Failed to read frame from camera")
            break

        # Flip frame horizontally for mirror-like display
        frame = cv.flip(frame, 1)

        # Detect faces in the current frame
        retval, detected_faces = detector.detect(frame)

        # Process detected faces
        if detected_faces is not None:
            # Process each detected face
            for face_data in detected_faces:
                # Align and extract face feature vector
                aligned_face = recognizer.alignCrop(frame, face_data)
                face_feature = recognizer.feature(aligned_face)

                # Compare against all stored encodings
                for encoding_file in os.listdir(ENCODINGS_DIRECTORY):
                    # Only process .npy encoding files
                    if not encoding_file.endswith(".npy"):
                        continue

                    # Load stored face encoding
                    encoding_path = os.path.join(
                        ENCODINGS_DIRECTORY,
                        encoding_file
                    )
                    stored_encoding = np.load(encoding_path)
                    
                    # Calculate similarity score with current face
                    similarity_score = recognizer.match(
                        face_feature,
                        stored_encoding,
                        MATCH_METRIC
                    )

                    # Report match if confidence exceeds threshold
                    if similarity_score > MATCH_THRESHOLD:
                        print(f"Match found: {encoding_file} (Score: {similarity_score:.3f})")

                # Draw bounding box around detected face
                x, y, width, height = map(int, face_data[0:4])
                cv.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    RECTANGLE_COLOR,
                    RECTANGLE_THICKNESS
                )

        # Display the frame with detections
        cv.imshow(WINDOW_NAME, frame)

        # Check for exit key (q)
        if cv.waitKey(1) & 0xFF == QUIT_KEY:
            print("Application terminated by user")
            break

# Cleanup: release camera and close windows
finally:
    camera.release()
    cv.destroyAllWindows()