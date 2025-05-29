
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset

train = pd.read_csv('train.csv')

target_numpy = train.label.values
feautures_numpy = train.loc[:, train.columns != 'label'].values / 255.0
x_train, x_test, y_train, y_test = train_test_split(
    feautures_numpy, target_numpy, test_size=0.2, random_state=42

)

# Convert to PyTorch tensors

xTrain = torch.tensor(x_train, dtype=torch.float32)
yTrain = torch.tensor(y_train, dtype=torch.long)
xTest = torch.tensor(x_test, dtype=torch.float32)
yTest = torch.tensor(y_test, dtype=torch.long)

#Create pytorch dataset
train_dataset = TensorDataset(xTrain, yTrain)
test_dataset = TensorDataset(xTest, yTest)

# Create data loaders
batch_size = 100
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
# Define the neural network architecture
class logisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(logisticRegression, self).__init__()
        self.liner = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.liner(x)
        return out
#Define the model
input_size = xTrain.shape[1]
num_clases = 10
model = logisticRegression(input_size, num_clases)

#move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx,(features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)

        # Forward pass

        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Evaluate the model
model.eval() 
correct = 0
total = 0
with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f'Accuracy of the model on the test set: {accuracy:.2f}%')


import random

# Select a random sample from the test dataset
index = random.randint(0, len(xTest) - 1)
image = xTest[index].reshape(28, 28)
label = yTest[index].item()

# Get model prediction
model.eval()
with torch.no_grad():
    image_tensor = xTest[index].to(device).unsqueeze(0)
    output = model(image_tensor)
    print(output)
    _, predicted_label = torch.max(output, 1)

# Plot image
plt.imshow(image, cmap="gray")
plt.title(f"True Label: {label}, Predicted: {predicted_label.item()}")
plt.axis("off")
plt.show()
