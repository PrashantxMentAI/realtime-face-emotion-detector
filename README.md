# Real-Time Face Emotion Detection System ✨

A modern, desktop-based real-time face emotion detection dashboard built with **Python, OpenCV, and CustomTkinter**. This system captures a webcam stream, detects faces, and classifies expressions in real-time, displaying confidence distributions on a sleek glassmorphic dashboard.

---

## 🚀 Features

- **Real-Time Webcam Inference:** Smooth 30+ FPS stream showing bounding boxes and dominant expressions.
- **Deep Learning ONNX Classifier:** Uses a pre-trained **FERPlus** model via OpenCV's high-performance DNN module.
- **Modern Dark UI:** A premium desktop application designed with `customtkinter` (rounded elements, dark mode, responsive styling).
- **Interactive Analytics:**
  - **Dynamic Bar Chart:** A native custom-drawn canvas displaying confidence levels for 8 emotions: Neutral, Happy, Surprise, Sad, Angry, Disgust, Fear, and Contempt.
  - **Session Stats:** Track the number of active faces, dominant expression, and elapsed time.
  - **Transition Log:** Live-scrolling text box logging your expression changes with timestamps.
- **Snapshot capture:** Save face-bounded snapshots directly to a `snapshots/` folder with one click.
- **Dynamic Asset Loading:** If pre-trained weights are missing, the app shows a beautiful loading screen and downloads assets automatically before launching.

---

## 🛠️ Tech Stack

- **Python**: Core development language.
- **OpenCV**: High-performance webcam streaming, face detection, and DNN network execution.
- **CustomTkinter**: GUI design (Modern desktop styles).
- **Pillow (PIL)**: Imaging utilities for format conversion.
- **Requests**: Handling dynamic model weight downloads.

---

## 📦 Installation & Setup

Follow these simple steps to run the application on your computer:

### 1. Set Up Virtual Environment (Recommended)

Open your terminal or command prompt in the project root directory and run:

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Required Dependencies

Install the packages specified in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Run the Application

Start the dashboard:

```bash
python app.py
```

*Note: The application will automatically check for the pre-trained weights on launch. If they aren't found, it will download them (`~35 MB`) showing a progress bar inside the app.*

---

## 📂 Project Structure

```
├── assets/
│   └── models/
│       └── emotion-ferplus-8.onnx   # Converted ONNX model file (Auto-downloaded)
├── snapshots/                       # Folder where captured screenshots are saved
├── app.py                           # Main application interface and frame loop
├── download_assets.py               # Downloader utility for models & cascades
├── emotion_detector.py              # Face crop and ONNX model inference logic
├── requirements.txt                 # Project library requirements
└── README.md                        # Documentation
```

---

## 🧠 Model Details

The model used is **`emotion-ferplus-8`** from the **ONNX Model Zoo**.
- **Dataset:** Trained on Microsoft's **FERPlus** dataset, which corrects labeling noise in the original FER-2013 dataset.
- **Input Shape:** `(1, 1, 64, 64)` - Single channel (grayscale) image normalized to 64x64 pixels.
- **Output:** Logits for 8 classes:
  1. Neutral
  2. Happy
  3. Surprise
  4. Sad
  5. Angry
  6. Disgust
  7. Fear
  8. Contempt
