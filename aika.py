import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the data from the csv file
data = pd.read_csv("indoor_outdoor_data.csv")

# Extract the features and labels from the data
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values

# Convert the data to PyTorch tensors and move to GPU if available
features = torch.tensor(features, dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.tensor(labels, dtype=torch.float32)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

# Define the neural network model
class IndoorOutdoorClassifier(nn.Module):
    def __init__(self):
        super(IndoorOutdoorClassifier, self).__init__()
        self.layer1 = nn.Linear(X_train.shape[1], 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

# Create an instance of the model
model = IndoorOutdoorClassifier().cuda() if torch.cuda.is_available() else IndoorOutdoorClassifier()

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# Evaluate the model on the test data
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    test_accuracy = (test_outputs.round() == y_test).sum().float() / len(y_test)
    print("Test Accuracy:", test_accuracy)
