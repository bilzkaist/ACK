import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda import is_available

# Define your transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.transformer = nn.Transformer(hidden_size, num_layers, num_heads, dropout)
        self.fc = nn.Linear(hidden_size, 2) # output layer for x and y values
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x[:,-1]) # Use the last hidden state for prediction
        return x

# Initialize the model
input_size = 10#len(vocab) # vocab is your vocabulary of input sequence
hidden_size = 512
num_layers = 6
num_heads = 8
dropout = 0.1
model = TransformerModel(input_size, hidden_size, num_layers, num_heads, dropout)

# Move model to CUDA if available
if is_available():
    model = model.cuda()

# Define your loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Define your dataset and data loader
# dataset = Your dataset for training and testing
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for x, y in dataloader:
        if is_available():
            x, y = x.cuda(), y.cuda()
        # Forward pass
        output = model(x)
        loss = criterion(output, y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Print loss and accuracy at each epoch
    print(f'Epoch: {epoch+1}, Loss: {loss.item()}, Accuracy: {accuracy(output, y)}')
    
# Testing loop
with torch.no_grad():
    for x, y in dataloader:
        if is_available():
            x = x.cuda()
        output = model(x)
        test_loss = criterion(output, y)
        # Print loss and accuracy on test set
        print(f'Test Loss: {test_loss.item()}, Test Accuracy: {accuracy(output, y)}')


def run():
    print("AIKF  Started -!!!")
    # Enter code here

    print("AIKF Finished Successfully !!!")


if __name__ == '__main__':
    print("Program  Started !!!")
    run()
    print("Program Finished Successfully !!!")
