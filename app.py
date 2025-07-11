import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import base64

# ------------------------
# Page Config
# ------------------------
st.set_page_config(page_title="Deepfake Detector", layout="wide")

# ------------------------
# Background Image and Font Styling
# ------------------------
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <link href="https://fonts.googleapis.com/css2?family=Audiowide&family=Roboto+Mono&display=swap" rel="stylesheet">
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center;
            font-family: 'Roboto Mono', monospace;
            color: white;
        }}
        h1 {{
            font-family: 'Audiowide', cursive;
            font-size: 48px;
            color: white;
            text-align: center;
        }}
        h2, h3, h4, h5, h6, p, div {{
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("background.jpg")

# ------------------------
# Title with Audiowide font
# ------------------------
st.markdown("<h1>DeepCheck</h1>", unsafe_allow_html=True)
st.markdown("Upload an image to check if it's Real or Fake.")

# ------------------------
# File Upload
# ------------------------
uploaded_file = st.file_uploader("Upload a Face Image", type=["jpg", "jpeg", "png"])

# ------------------------
# Load Model
# ------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# ------------------------
# Prediction Function
# ------------------------
def predict(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()
    return predicted.item(), confidence

# ------------------------
# Show Image and Prediction
# ------------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img.thumbnail((400, 400))
    st.image(img, caption="Uploaded Image", use_column_width=False)
    pred, conf = predict(img)
    label = "🟢 Real" if pred == 0 else "🔴 Fake"
    st.markdown(f"### Prediction: {label}")
    st.markdown(f"**Confidence:** {conf * 100:.2f}%")

# ------------------------
# Visualizations
# ------------------------
st.markdown("---")
st.header("Model Insights")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Training Loss & Accuracy")
    if os.path.exists("training_metrics.png"):
        st.image("training_metrics.png", width=450)

    st.subheader("Confusion Matrix")
    if os.path.exists("confusion_matrix.png"):
        st.image("confusion_matrix.png", width=450)

with col2:
    st.subheader("ROC Curve")
    if os.path.exists("roc_curve.png"):
        st.image("roc_curve.png", width=450)

st.markdown("---")
st.caption("Made by Srishti & Misha | AI Deepfake Detection")

