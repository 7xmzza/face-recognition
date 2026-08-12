> [!CAUTION]
> This project is still under development. Always check the "Current state and features" section below to understand the current state.

# Face Recognition System

A summer project by Hamza ElNahtawy focused on learning through hands-on project development. The project uses DNN-based models to detect and recognize faces from a webcam feed.

## Current state and features
- Live recognition with terminal feedback
- Uses YuNet face detection and SFace face recognition DNN models through OpenCV's built-in classes
- Uses OpenCV default webcam resolution of 640x480 (I left it as default just for simplicity)
- Green bounding box drawn around recognized faces with recognition feedback printed in the terminal
- Creates encodings for images in knownFaces (images must contain only one clear face)


## Expected features
- Display recognition results directly in the webcam window
- Label recognized faces with their corresponding identity
- Improve handling of multiple faces
- Improve recognition reliability across different lighting and poses
- Store and manage multiple encodings per person

## How it works

The system is split into two main stages:

1. **Face detection**
   - YuNet detects faces in the input image or webcam frame using OpenCV's `FaceDetectorYN` class.
   - Detected faces are aligned using OpenCV's `FaceRecognizerSF` class.

2. **Face recognition**
   - SFace generates a feature embedding for each detected face.
   - Known face embeddings are stored as `.npy` files.
   - New embeddings are compared against the stored encodings to determine whether a face is recognized.

For more information, check the code for comments.

## How to use
1. Clone the repo: 
  ```git clone https://github.com/7xmzza/face-recognition.git```

2. Create and activate venv and install dependencies:
  ```
  python -m venv .venv
  .venv/Scripts/activate
  pip install -r requirements.txt
  ```

3. Save images of faces you want to recognize in the knownFaces directory. The images must contain only one face each. Only `.png` and `.jpg` are supported for now.

4. Run `create_encodings.py` then run `detect_face.py`

## This project uses
- Python
- OpenCV
  - YuNet — face detection
  - SFace — face recognition
- NumPy

## Contributions
Contributions are welcome and encouraged
