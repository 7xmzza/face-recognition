> [!CAUTION]
> This project is still under development. Always check the "Current state and features" section below to understand the current state.

# 👤 Face Recognition System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg?style=flat-square&logo=opencv)
![Status](https://img.shields.io/badge/Status-In%20Development-orange.svg?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square)

> A high-performance face detection and recognition system using deep neural networks. Built as a summer learning project with state-of-the-art DNN models for real-time face identification.

---

## ✨ Current Features

- ✅ **Live Face Detection** - Real-time face detection from webcam feed
- ✅ **Face Recognition** - Compare detected faces against known face database
- ✅ **State-of-the-art Models** - YuNet (detection) and SFace (recognition) via OpenCV
- ✅ **Visual Feedback** - Green bounding boxes around detected faces
- ✅ **Console Output** - Recognition matches logged with confidence scores
- ✅ **Automatic Encoding** - Generate face encodings from images (single face per image)
- ✅ **Clean Code** - Well-commented, PEP 8 compliant, maintainable structure

## 🎯 Planned Features

- 📍 On-screen face labels and confidence scores
- 🔄 Multi-face handling improvements
- 💡 Lighting and pose invariant recognition
- 👥 Multiple encodings per person
- 🗄️ Face database management system
- 🎨 Advanced visualization options

---

## 🔬 How It Works

The system operates in **two main stages**:

### 1️⃣ Face Encoding (Preparation Phase)
```
Images in known_faces/ → Face Detection (YuNet) → Face Alignment → 
Feature Extraction (SFace) → Save as .npy files
```

### 2️⃣ Live Recognition (Runtime Phase)
```
Webcam Feed → Frame Capture → Face Detection (YuNet) → Face Alignment → 
Feature Extraction (SFace) → Compare with Database → Match/No Match
```

**Technical Details:**
- **YuNet**: Lightweight face detection model for real-time performance
- **SFace**: Deep learning model that generates 128-D feature embeddings
- **Matching**: Cosine similarity metric with 0.7 confidence threshold
- **Storage**: NumPy `.npy` files for efficient encoding storage

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam
- 50MB+ disk space for models

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/7xmzza/face-recognition.git
   cd face-recognition
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   .venv/Scripts/activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

**Step 1: Prepare Face Database**
```bash
# Add your face images to the test_images/ folder
# Requirements: Only ONE clear face per image
# Supported formats: .jpg, .png, .jpeg
python create_encodings.py
```

**Step 2: Run Live Recognition**
```bash
python recognize_faces.py
```

Press `q` to quit the application.

---

## 📁 Project Structure

```
face-recognition/
├── create_encodings.py          # Generate face encodings from images
├── recognize_faces.py           # Real-time face recognition
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── encodings/                   # Generated face encoding files (.npy)
├── weights/                     # Pre-trained model files
│   ├── face_detection_yunet_2026may.onnx
│   └── face_recognition_sface_2021dec.onnx
├── test_images/                 # Development test images
└── known_faces/                 # Production face database (for later)
```

---

## 🛠️ Technologies

| Technology | Purpose | Version |
|-----------|---------|---------|
| **Python** | Core language | 3.8+ |
| **OpenCV** | Computer vision framework | 4.5+ |
| **YuNet** | Face detection DNN | 2026 May |
| **SFace** | Face recognition DNN | 2021 Dec |
| **NumPy** | Numerical computing | Latest |

---

## ⚙️ Configuration

Customize behavior by modifying constants in the Python files:

**In `create_encodings.py`:**
- `SOURCE_DIRECTORY` - Input images folder
- `DETECTION_CONFIDENCE_THRESHOLD` - Min confidence for face detection
- `DETECTION_INPUT_SIZE` - Face detector input dimensions

**In `recognize_faces.py`:**
- `MATCH_THRESHOLD` - Minimum score for face recognition (0.0-1.0)
- `RECTANGLE_COLOR` - Bounding box color in BGR format
- `DETECTION_INPUT_SIZE` - Webcam capture resolution

---

## 🤝 Contributing

Contributions are welcome and encouraged! Feel free to:
- Report bugs and suggest features via Issues
- Submit pull requests with improvements
- Share your use cases and results

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Hamza ElNahtawy** - Summer 2026 Learning Project

---

## 📞 Support

For issues, questions, or suggestions, please open an issue on GitHub or check the code comments for detailed documentation.
