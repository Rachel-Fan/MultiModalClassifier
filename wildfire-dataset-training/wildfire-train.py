import pandas as pd
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Step 1: Data Preparation
data = pd.read_csv(r'C:\Users\GeoFly\Documents\rfan\MultiModalClassifier\Dataset\wildfire\train\_classes.csv')  # Update the path

# Step 2: Data Loading
class CustomDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name)
        label = torch.tensor(self.dataframe.iloc[idx, 1:].tolist(), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

# Step 3: Data Augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 4: Model Definition
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)  # 3 output classes

# Step 5: Model Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

dataset = CustomDataset(data, root_dir=r'C:\Users\GeoFly\Documents\rfan\MultiModalClassifier\Dataset\wildfire\train\images', transform=transform)  # Update the path
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def train_model(model, criterion, optimizer, train_loader, test_loader, device, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels.argmax(dim=1)).sum().item()
            total_train += labels.size(0)
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc_train = correct_train / total_train
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}, Train Acc: {epoch_acc_train:.4f}")

# Train the model
train_model(model, criterion, optimizer, train_loader, test_loader, device)
