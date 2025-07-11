import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

# ------------------------
# 🧠 Config
# ------------------------
real_path = "dataset/dataset_real"
fake_path = "dataset/dataset_fake"
num_epochs = 15
batch_size = 32
learning_rate = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# 📦 Custom Dataset
# ------------------------
class CustomDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir)]
        self.fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)]
        self.labels = [0]*len(self.real_images) + [1]*len(self.fake_images)
        self.images = self.real_images + self.fake_images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# ------------------------
# 🧪 Transformations (With Augmentation)
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# ------------------------
# 🧱 Load Data
# ------------------------
dataset = CustomDataset(real_path, fake_path, transform=transform)
print("🧾 Label Mapping: Real = 0, Fake = 1")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# ------------------------
# 🧠 Model
# ------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses, val_accuracies = [], []

# ------------------------
# 🔁 Training Loop with Timer
# ------------------------
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    val_accuracies.append(val_acc)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

# ------------------------
# 💾 Save Model
# ------------------------
torch.save(model.state_dict(), "model.pth")
print("✅ Model saved as model.pth")

# ------------------------
# 📈 Plot Training Metrics
# ------------------------
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Loss")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Loss & Accuracy")
plt.legend()
plt.savefig("training_metrics.png")
plt.close()

# ------------------------
# 📊 Confusion Matrix
# ------------------------
all_preds = []
all_labels = []
model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

cf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix.png")
plt.close()

# ------------------------
# 📉 ROC Curve
# ------------------------
probs = []
model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        probs_batch = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
        probs.extend(probs_batch.cpu().numpy())

fpr, tpr, _ = roc_curve(all_labels, probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.close()

# ------------------------
# ⏱️ Time Taken
# ------------------------
end_time = time.time()
print(f"⏱️ Total Training Time: {(end_time - start_time):.2f} seconds")



