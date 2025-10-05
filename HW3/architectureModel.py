import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# setting of hyperparameters
batch_size = 512
learning_rate = 5e-3
num_epochs = 20
image_size = (128, 128)

# path settings
data_dir = r"C:\Users\allen\.conda\envs\virus_pic\gray_virus\class_5_out"
model_path = r"C:\Users\allen\.conda\envs\virus_pic\gray_virus\cnn_model.pth"

# Transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# load dataset and split
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
labels = [label for _, label in full_dataset]

train_indices, val_indices = train_test_split(
    range(len(labels)), test_size=0.3, stratify=labels, random_state=42
)

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Total images: {len(full_dataset)}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")

# define CNN model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * (image_size[0] // 4) * (image_size[1] // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# setup device, model, loss function, optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN(num_classes=len(full_dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# load pre-trained model if exists
if os.path.exists(model_path):
    print(f"Loading pre-trained model from {model_path}")
    model.load_state_dict(torch.load(model_path))
else:
    print("No pre-trained model found. Starting training from scratch.")

# validation function
def validate(model, val_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# training function
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    accuracy_list, loss_list = [], []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for images, labels in train_loader_tqdm:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        acc = validate(model, val_loader)
        accuracy_list.append(acc)
        loss_list.append(running_loss / len(train_loader))
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}, Val Acc: {acc:.2f}%")
        torch.save(model.state_dict(), model_path)
    return accuracy_list, loss_list

# train the model
accuracy_list, loss_list = train(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# draw and save metrics
def plot_metrics(accuracy_list, loss_list, acc_path, loss_path):
    plt.figure()
    plt.plot(range(1, len(accuracy_list)+1), accuracy_list, marker='o')
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.grid()
    plt.savefig(acc_path)

    plt.figure()
    plt.plot(range(1, len(loss_list)+1), loss_list, marker='o', color='red')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid()
    plt.savefig(loss_path)

plot_metrics(
    accuracy_list, loss_list,
    acc_path=r"C:\Users\allen\.conda\envs\virus_pic\gray_virus\accuracy.png",
    loss_path=r"C:\Users\allen\.conda\envs\virus_pic\gray_virus\loss.png"
)
