import os
import cv2
import numpy as np

class EmotionDetector:
    # Emotion categories in the order of the model outputs (FERPlus classes)
    EMOTIONS = [
        "Neutral",
        "Happy",
        "Surprise",
        "Sad",
        "Angry",
        "Disgust",
        "Fear",
        "Contempt"
    ]

    def __init__(self, model_path=None):
        """Initializes the face detector and loads the pre-trained emotion ONNX model."""
        if model_path is None:
            # Default path relative to this script
            base_dir = os.path.dirname(__file__)
            model_path = os.path.join(base_dir, "assets", "models", "emotion-ferplus-8.onnx")
        
        self.model_path = model_path
        
        # Load Haar Cascade face classifier (built into OpenCV)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise IOError("Failed to load OpenCV Haar Cascade frontal face XML classifier.")

        # Load the ONNX model using OpenCV DNN
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ONNX model file not found at: {model_path}. "
                f"Please run download_assets.py first."
            )
            
        try:
            self.net = cv2.dnn.readNetFromONNX(model_path)
        except Exception as e:
            raise RuntimeError(f"Error loading ONNX model with cv2.dnn: {e}")

    def detect_faces(self, gray_frame, scale_factor=1.1, min_neighbors=5, min_size=(60, 60)):
        """Detects faces in a grayscale frame. Returns list of bounding boxes (x, y, w, h)."""
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        return faces

    def softmax(self, x):
        """Numerically stable softmax implementation."""
        # Subtract max to prevent overflow
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def predict_emotion(self, gray_frame, face_box, pad_ratio=0.1, equalize=True):
        """Crops a face, preprocesses it, runs inference, and returns a dictionary of emotion probabilities."""
        x, y, w, h = face_box
        
        # 1. Extract face ROI with optional boundary padding to capture full facial features
        if pad_ratio > 0.0:
            x_pad = int(w * pad_ratio)
            y_pad = int(h * pad_ratio)
            x_start = max(0, x - x_pad)
            y_start = max(0, y - y_pad)
            x_end = min(gray_frame.shape[1], x + w + x_pad)
            y_end = min(gray_frame.shape[0], y + h + y_pad)
            face_roi = gray_frame[y_start:y_end, x_start:x_end]
        else:
            face_roi = gray_frame[y:y+h, x:x+w]
            
        if face_roi.size == 0:
            # Empty crop
            return None
            
        try:
            # 2. Histogram Equalization to normalize contrast under variable room lighting
            if equalize:
                face_roi = cv2.equalizeHist(face_roi)
                
            # 3. Resize to 64x64 as required by the emotion-ferplus ONNX model
            face_resized = cv2.resize(face_roi, (64, 64), interpolation=cv2.INTER_AREA)
            
            # 4. Create input blob. The model expects (1, 1, 64, 64) input shape.
            # Scaling factor is 1.0 (unscaled [0, 255] pixels), mean is 0, swapRB is False.
            blob = cv2.dnn.blobFromImage(
                face_resized, 
                scalefactor=1.0, 
                size=(64, 64), 
                mean=0, 
                swapRB=False, 
                crop=False
            )
            
            # 3. Set input and run forward pass
            self.net.setInput(blob)
            preds = self.net.forward()  # Shape: (1, 8) logits
            
            # 4. Softmax activation to convert logits to probabilities
            probs = self.softmax(preds)[0]  # Get first batch prediction
            
            # Create a dictionary mapping emotions to percentages
            emotion_probs = {self.EMOTIONS[i]: float(probs[i] * 100) for i in range(len(self.EMOTIONS))}
            
            # Get the top dominant emotion
            dominant_emotion = self.EMOTIONS[np.argmax(probs)]
            dominant_confidence = float(np.max(probs) * 100)
            
            return {
                "probabilities": emotion_probs,
                "dominant": dominant_emotion,
                "confidence": dominant_confidence
            }
        except Exception as e:
            print(f"Error during emotion inference: {e}")
            return None
