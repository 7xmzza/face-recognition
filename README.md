# Face Recognition System

A real-time face detection and recognition system built with **Python**, **OpenCV**, **YuNet**, and **SFace**.

The system detects faces from a webcam feed, generates facial feature embeddings, and compares them against pre-generated encodings to identify known individuals.

## ✨ Features

- 🎥 Real-time webcam face detection
- 🧠 YuNet DNN face detection
- 👤 SFace face recognition
- 📸 Multiple reference images per person
- ⚡ Encodings loaded into memory before recognition
- 🎯 Configurable recognition threshold
- 🖥️ Live bounding boxes and name labels

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Computer vision and webcam processing |
| YuNet | Face detection |
| SFace | Face recognition |
| NumPy | Facial feature storage and processing |
| ONNX | Model format |

## 🧠 How It Works

The project is divided into two main stages:

### 1. Encoding Generation

Reference images are organized into folders by person:

```text
known_faces/
├── Hamza/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── image_3.jpg
├── Ahmed/
│   ├── image_1.jpg
│   └── image_2.jpg
└── Omar/
    └── image_1.jpg
```

Each image is processed by YuNet.

If exactly one face is detected:

1. The face is detected.
2. SFace aligns the face using YuNet's facial landmarks.
3. SFace generates a facial feature vector.
4. The feature vector is saved as a `.npy` file.

The resulting encoding structure is:

```text
encodings/
├── Hamza/
│   ├── Hamza_0.npy
│   ├── Hamza_1.npy
│   └── Hamza_2.npy
├── Ahmed/
│   ├── Ahmed_0.npy
│   └── Ahmed_1.npy
└── Omar/
    └── Omar_0.npy
```

### 2. Real-Time Recognition

When the recognition program starts, all stored encodings are loaded into memory.

They are organized using a dictionary:

```python
encodings = {
    "Hamza": [encoding_1, encoding_2],
    "Ahmed": [encoding_1, encoding_2],
}
```

The webcam then continuously:

```text
Webcam Frame
      ↓
   YuNet
      ↓
 Face Detected
      ↓
  SFace Align
      ↓
 SFace Feature
      ↓
Compare Against Stored Encodings
      ↓
 Recognition Score
      ↓
   Identify Person
```

## 📊 Recognition

The project uses SFace's **cosine similarity** metric to compare the live facial feature against stored encodings.

The current recognition threshold is:

```python
score > 0.7
```

A higher score indicates greater similarity between the two facial feature vectors.

> The threshold is configurable and may need adjustment depending on the quality of the reference images, camera, lighting, and desired false-positive/false-negative tradeoff.

## 📁 Project Structure

```text
face-recognition/
│
├── weights/
│   ├── face_detection_yunet_2026may.onnx
│   └── face_recognition_sface_2021dec.onnx
│
├── known_faces/
│   ├── Person/
│   │   ├── image_1.jpg
│   │   └── image_2.jpg
│   └── ...
│
├── encodings/
│   ├── Person/
│   │   ├── Person_0.npy
│   │   └── Person_1.npy
│   └── ...
│
├── generate_encodings.py
├── face_recognition.py
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- A webcam
- OpenCV
- NumPy

### Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd face-recognition
pip install -r requirements.txt
```

### Add Reference Images

Create a folder for each person you want the system to recognize:

```text
known_faces/
├── Hamza/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── image_3.jpg
└── Ahmed/
    ├── image_1.jpg
    └── image_2.jpg
```

For best results, use clear images where the person's face is visible.

### Generate Encodings

Run the encoding-generation script:

```bash
python create_encodings.py
```

The generated facial feature vectors will be stored in the `encodings/` directory.

### Start Recognition

Run the recognition program:

```bash
python face_recognition.py
```

The webcam feed will open and detected faces will be compared against the stored encodings.

Press **Q** to exit.

## ⚙️ Configuration

The main configuration values can be changed at the beginning of the scripts.

### Model Paths

```python
DETECTOR_PATH = "weights/face_detection_yunet_2026may.onnx"
RECOGNIZER_PATH = "weights/face_recognition_sface_2021dec.onnx"
```

### Recognition Threshold

```python
if score > 0.7:
```

Increasing the threshold makes recognition more strict.

Decreasing it makes recognition more permissive.

## 🔮 Future Improvements

- [ ] Improve identity matching by aggregating scores across multiple reference images
- [ ] Add explicit `Unknown` classification
- [ ] Add face tracking between frames
- [ ] Improve recognition stability across consecutive frames
- [ ] Add configurable detection and recognition thresholds
- [ ] Add command-line configuration
- [ ] Improve dataset validation
- [ ] Optimize recognition for larger datasets

## 📜 License

This project is intended for educational and experimental purposes.

Check the `LICENSE` file to learn more