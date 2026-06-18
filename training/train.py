import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2

# Define the CNN architecture
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(EmotionCNN, self).__init__()
        
        # Input: 1 x 64 x 64
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 16 x 32 x 32
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 32 x 16 x 16
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 64 x 8 x 8
        
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return x

# Dataset loader
class FERDataset(Dataset):
    def __init__(self, csv_file, usage="Training"):
        self.df = pd.read_csv(csv_file)
        # Filter by usage if present (e.g. Training, PublicTest, PrivateTest)
        if "Usage" in self.df.columns:
            self.df = self.df[self.df["Usage"] == usage].reset_index(drop=True)
            
        self.labels = self.df["emotion"].values
        self.pixels = self.df["pixels"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Parse pixel values string
        pixel_array = np.fromstring(self.pixels[idx], dtype=np.uint8, sep=' ').reshape(48, 48)
        
        # Resize to 64x64 to match the dashboard's ONNX requirements
        pixel_resized = cv2.resize(pixel_array, (64, 64), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1] range and add channel dimension
        img_tensor = torch.tensor(pixel_resized, dtype=torch.float32).unsqueeze(0) / 255.0
        
        # FER2013 has 7 classes (0-6). Mapping to 8 output classes:
        # 0: Neutral (maps from 6)
        # 1: Happy (maps from 3)
        # 2: Surprise (maps from 5)
        # 3: Sad (maps from 4)
        # 4: Angry (maps from 0)
        # 5: Disgust (maps from 1)
        # 6: Fear (maps from 2)
        # 7: Contempt (empty/0 in standard FER2013)
        fer_label = self.labels[idx]
        mapping = {0: 4, 1: 5, 2: 6, 3: 1, 4: 3, 5: 2, 6: 0}
        mapped_label = mapping.get(fer_label, 0)
        
        return img_tensor, torch.tensor(mapped_label, dtype=torch.long)

def train_model(csv_path, output_onnx_path, epochs=3, batch_size=32, lr=0.001):
    print(f"Loading data from: {csv_path}")
    dataset = FERDataset(csv_path, usage="Training")
    
    if len(dataset) == 0:
        print("ERROR: Dataset is empty. Make sure you generated the dummy data or placed the real dataset.")
        return False
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = EmotionCNN(num_classes=8).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("\nStarting training loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")
        
    print("\nTraining complete! Exporting model to ONNX...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create a dummy input tensor matching input shape (1, 1, 64, 64)
    dummy_input = torch.randn(1, 1, 64, 64, device=device)
    
    # Export to ONNX
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    torch.onnx.export(
        model, 
        dummy_input, 
        output_onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"Model exported successfully to: {output_onnx_path}")
    return True

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, "fer2013_dummy.csv")
    
    # If dummy data doesn't exist, generate it
    if not os.path.exists(csv_file):
        from generate_dummy_data import generate_mock_fer2013
        generate_mock_fer2013(csv_file)
        
    onnx_output = os.path.join(base_dir, "..", "assets", "models", "emotion-ferplus-8.onnx")
    train_model(csv_file, onnx_output, epochs=3)
