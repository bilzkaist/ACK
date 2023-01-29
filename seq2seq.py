import torch
import torch.nn as nn

# Define the neural network model
class NNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size,dtype=torch.float32)
        self.fc2 = nn.Linear(hidden_size, output_size,dtype=torch.float32)


    def forward(self, input_sequence):
        # Cast input_sequence to float before passing to linear layer
        input_sequence = input_sequence.float()
        # Output the kth integer in the input sequence
        hidden = self.fc1(input_sequence)
        hidden = torch.relu(hidden)
        output = self.fc2(hidden)
        return output

# Define the number of integers in the sequence
N = 10

# Generate three random input sequences
input_sequence_1 = torch.randint(0, 100, (N,))
input_sequence_2 = torch.randint(0, 100, (N,))
input_sequence_3 = torch.randint(0, 100, (N,))

# Define the integer to be outputted (k)
k = 5

# Initialize the model and move to CUDA device
model = NNModel(input_size=N, hidden_size=64, output_size=1)
model = model.to("cuda")

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for epoch in range(100):
    # Move input sequences to CUDA device
    input_sequence_1 = input_sequence_1.to("cuda")
    input_sequence_2 = input_sequence_2.to("cuda")
    input_sequence_3 = input_sequence_3.to("cuda")

    # Forward pass
    output_1 = model(input_sequence_1)
    output_2 = model(input_sequence_2)
    output_3 = model(input_sequence_3)

    # Calculate loss
    loss_1 = criterion(output_1[k], input_sequence_1[k])
    loss_2 = criterion(output_2[k], input_sequence_2[k])
    loss_3 = criterion(output_3[k], input_sequence_3[k])
    loss = (loss_1 + loss_2 + loss_3) / 3

    # Zero gradients and perform backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Test the model on the input sequences
output_1 = model(input_sequence_1)
output_2 = model(input_sequence_2)
output_3 = model(input_sequence_3)

# Print the kth integer from the output
print(f'Output from input sequence 1: {output_1[k]}')
print(f'Output from input sequence 2: {output_2[k]}')
print(f'Output from input sequence 3: {output_3[k]}')
