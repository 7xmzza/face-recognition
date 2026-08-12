> [!CAUTION]
> This project is still under development and may not be fully functional as described (yet probably usable).

# Face Recognition System

Learning through building a live face recognition system. The project is expected to reach the following functionality and features. Always check the status section below to learn about the project's development stage.

## Current state and features
- Live face detection, not recognition yet
- Uses YuNet face detection DNN model (instead of the previous Haar Cascade)
- Uses OpenCV default resolution 640x480
- Green box drawn around detected face


## Expected features

- Live webcam output
- Face detection & recognition
- Real-time boxes marking detected faces
- Labeling faces based on dataset encodings

## How to use
1. Clone the repo: 
  ```git clone https://github.com/7xmzza/face-recognition.git```

2. Create and activate venv and install dependencies:
```
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

3. Run `detect_face.py`
   

## This project uses

- Python
- OpenCV
- NumPy

## Contributions
Contributions are welcome and encouraged
