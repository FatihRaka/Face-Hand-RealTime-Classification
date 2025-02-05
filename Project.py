import streamlit as st
import cv2
import numpy as np
import math
import time
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

st.title("Object Detection + Classification")
st.subheader("Aplikasi ini dibuat untuk mendeteksi objek dengan YOLO dan mengklasifikasikan gesture tangan dengan CNN")

# Model paths
model_path_yolo = "YOLOv10.pt"
model_path_cnn = "handfinal.pt"

# Initialize session state variables
if 'model' not in st.session_state:
    st.session_state.model = YOLO(model_path_yolo)  # Default model
    st.session_state.model_type = "YOLO"
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.conv2 = nn.Conv2d(10, 16, 3)
        # Menggunakan AdaptiveAvgPool2d untuk menyesuaikan ukuran output gambar
        self.pool = nn.AdaptiveAvgPool2d((6, 6))  # Hasil akhir untuk setiap gambar adalah 6x6
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.pool(x)  # Menyesuaikan ukuran output menjadi 6x6
        x = x.view(-1, self.num_flat_features(x))  # Reshaping
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # Semua dimensi kecuali batch
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# Function to load YOLO model
def load_yolo_model(model_path):
    st.session_state.model = YOLO(model_path)
    st.session_state.model_type = "YOLO"
    st.success(f"Model {model_path} berhasil dimuat sebagai YOLO!")

# Function to load CNN model
def load_cnn_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    st.session_state.model = model
    st.session_state.model_type = "CNN"
    st.success(f"Model {model_path} berhasil dimuat sebagai CNN!")

# Buttons to select model
col1, col2 = st.columns(2)
with col1:
    if st.button("Facial Landmark Detection"):
        load_yolo_model(model_path_yolo)
with col2:
    if st.button("Gesture Classification"):
        load_cnn_model(model_path_cnn)

# Transformation for CNN model
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize image to match model input
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])

# Define class labels for CNN
classes = ['Fist', 'Thumbs Up', 'Open']

# Webcam feed function
def run_webcam_feed():
    cap = cv2.VideoCapture(0)  # Start webcam
    cap.set(3, 640)  # Set width
    cap.set(4, 480)  # Set height
    stframe = st.empty()  # Placeholder for video frames

    while st.session_state.webcam_active:
        success, img = cap.read()
        if not success:
            st.warning("Failed to access webcam.")
            break

        if st.session_state.model_type == "YOLO":
            # YOLO inference
            results = st.session_state.model(img, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

                    # Confidence
                    confidence = math.ceil((box.conf[0] * 100)) / 100

                    # Class name
                    cls = int(box.cls[0])
                    class_name = st.session_state.model.names[cls]

                    # Display label
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elif st.session_state.model_type == "CNN":
            # CNN inference
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img_pil = Image.fromarray(img_rgb)              # Convert to PIL Image
            img_tensor = transform(img_pil).unsqueeze(0)    # Apply transformation and add batch dimension

            # Perform prediction
            with torch.no_grad():
                outputs = st.session_state.model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                class_label = classes[predicted.item()]

            # Display prediction
            cv2.putText(img, f"Prediction: {class_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Convert image to RGB for Streamlit display
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        stframe.image(img, channels="RGB")

        # Small delay to simulate real-time
        time.sleep(0.03)

        # Check for Streamlit rerun (which happens when session state changes)
        if not st.session_state.webcam_active:
            break

    cap.release()
    cv2.destroyAllWindows()

# Toggle webcam feed button
if st.button("Mulai/Pause Deteksi Webcam"):
    # Toggle the webcam active state
    st.session_state.webcam_active = not st.session_state.webcam_active

    # If webcam is now active, run the feed
    if st.session_state.webcam_active:
        run_webcam_feed()
