# Custom Emotion Classifier Training Pipeline 🧠

This folder contains a complete Convolutional Neural Network (CNN) training pipeline built in **PyTorch**. You can use this to train your own custom model and export it to the ONNX format required by the dashboard.

---

## 🛠️ Requirements & Setup

Before running the training script, install PyTorch, Pandas, and Openpyxl/others if needed:

```bash
# Activate your virtual environment first, then run:
pip install torch torchvision pandas openpyxl
```

---

## 🏃 How to Run Training

### 1. Fast Demo (Mock Dataset)
We have provided a mock dataset generator so you can test the training script immediately without waiting to download large files.

Run the script:
```bash
python train.py
```
*This will automatically generate a small dummy dataset `fer2013_dummy.csv`, train the CNN for 3 epochs, and save the exported `emotion-ferplus-8.onnx` file directly into your `assets/models/` folder, ready for testing on the webcam.*

---

### 2. Train on the Real FER-2013 Dataset (Full Accuracy)
To train a model that actually recognizes real expressions:

1. Download the `fer2013.csv` file from [Kaggle's FER-2013 Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
2. Move the `fer2013.csv` file into this `training/` folder.
3. Open `train.py` in your text editor and scroll to the bottom. Change the CSV filename to match:
   ```python
   csv_file = os.path.join(base_dir, "fer2013.csv")
   ```
4. Run the training script:
   ```bash
   python train.py
   ```
5. *(Optional)* To train it for higher accuracy, you can open `train.py` and increase the number of epochs (e.g. `epochs=30` or `50`).

Once the script completes, the new `.onnx` file will be generated, and your dashboard app will immediately start using your newly trained custom model!
