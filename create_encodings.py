"""
Face Encoding Generator

This module processes a directory of face images and generates face encodings
using the SFace face recognition model. These encodings are stored as .npy files
for use in real-time face recognition and matching.

Requirements:
    - OpenCV with ONNX support
    - YuNet face detection weights (face_detection_yunet_2026may.onnx)
    - SFace recognition weights (face_recognition_sface_2021dec.onnx)
"""

import cv2 as cv
import numpy as np
import os

# Directory configuration
SOURCE_DIRECTORY = "test_images"  # Directory containing face images to encode
TARGET_DIRECTORY = "encodings"  # Directory where .npy encoding files will be saved

# Model paths
DETECTION_MODEL_PATH = "weights/face_detection_yunet_2026may.onnx"
RECOGNITION_MODEL_PATH = "weights/face_recognition_sface_2021dec.onnx"

# Detection parameters
DETECTION_INPUT_SIZE = (320, 320)
DETECTION_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for face detection

# Supported image file extensions
SUPPORTED_IMAGE_FORMATS = (".jpg", ".png", ".jpeg")

# Initialize face detection and recognition models
detector = cv.FaceDetectorYN.create(
    DETECTION_MODEL_PATH,
    "",
    DETECTION_INPUT_SIZE,
    DETECTION_CONFIDENCE_THRESHOLD
)
recognizer = cv.FaceRecognizerSF.create(
    RECOGNITION_MODEL_PATH,
    ""
)

# Counter for naming encoding files sequentially
encoding_count = 0

# Process source directory if it exists
if os.path.exists(SOURCE_DIRECTORY):
    # Iterate through all files in the source directory
    for filename in os.listdir(SOURCE_DIRECTORY):
        # Only process image files
        if filename.lower().endswith(SUPPORTED_IMAGE_FORMATS):
            # Load and validate image
            image_path = os.path.join(SOURCE_DIRECTORY, filename)
            image = cv.imread(image_path)

            if image is None:
                print(f"{filename}: Failed to load image")
                continue

            # Adjust detector input size to match image dimensions
            height, width, _ = image.shape
            detector.setInputSize((width, height))

            # Detect faces in the current image
            retval, detected_faces = detector.detect(image)

            # Process detection results
            if detected_faces is None:
                print(f"{filename}: No faces detected")
            elif len(detected_faces) == 1:
                # Only process images with exactly one face
                print(f"{filename}: 1 face detected")
                aligned_face = recognizer.alignCrop(image, detected_faces[0])
                face_feature = recognizer.feature(aligned_face)
                
                # Save the face encoding
                output_path = os.path.join(
                    TARGET_DIRECTORY,
                    f"{encoding_count}.npy"
                )
                np.save(output_path, face_feature)
                encoding_count += 1
            else:
                # Skip images with multiple faces
                print(f"{filename}: {len(detected_faces)} faces detected")

else:
    print("ERROR: Source directory not found.")
    print(f"Please ensure '{SOURCE_DIRECTORY}' exists in the project root.")
    print("Place your face images (.jpg, .png, .jpeg) in this directory.")