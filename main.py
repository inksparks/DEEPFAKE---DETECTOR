import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
import urllib.request

# Grad-CAM utilities
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ------------------------------
# Set device (CPU or GPU)
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Load pre-trained model with correct weights
# ------------------------------
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval().to(device)

# ------------------------------
# Load and preprocess image
# ------------------------------
img_path = "sample_image.jpg"  # Make sure this image exists in your folder

if not os.path.exists(img_path):
    raise FileNotFoundError(f"{img_path} not found in the project directory.")

# Load image
image = Image.open(img_path).convert('RGB')

# Transform image for model input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
input_tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

# Prepare image as numpy array for Grad-CAM visualization
input_np = transform(image).permute(1, 2, 0).numpy()  # (H, W, C)
input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min())  # normalize to 0-1

# ------------------------------
# Generate prediction
# ------------------------------
output = model(input_tensor)
predicted_class = torch.argmax(output, dim=1).item()

# Download ImageNet class labels
url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
class_names = json.load(urllib.request.urlopen(url))

# Print actual label
print(f"Predicted Class: {class_names[predicted_class]}")


# ------------------------------
# Grad-CAM initialization
# ------------------------------
target_layer = model.layer4[-1]  # Last conv block of ResNet-18
cam = GradCAM(model=model, target_layers=[target_layer])

# Generate CAM
targets = [ClassifierOutputTarget(predicted_class)]  # explain the predicted class
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # (H, W)

# ------------------------------
# Overlay CAM on image
# ------------------------------
cam_result = show_cam_on_image(input_np, grayscale_cam, use_rgb=True)

# ------------------------------
# Display result
# ------------------------------
plt.imshow(cam_result)
plt.title("Grad-CAM Heatmap")
plt.axis('off')
plt.show()

