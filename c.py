import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import softmax

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
    
    def forward(self, x):
        x = self.embedding(x) # (batch_size, sequence_length, d_model)
        x = self.transformer(x) # (batch_size, sequence_length, d_model)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, output_dim):
        super(TransformerDecoder, self).__init__()
        self.linear = nn.Linear(d_model, output_dim * 2)
    
    def forward(self, x):
        x = self.linear(x) # (batch_size, sequence_length, output_dim * 2)
        x = x.view(-1, 2) # (batch_size * sequence_length, 2)
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.encoder = TransformerEncoder(input_dim, d_model, nhead, num_layers).cuda()
        self.decoder = TransformerDecoder(d_model, output_dim).cuda()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the model and move it to cuda
input_dim = 1000
d_model = 512
nhead = 8
num_layers = 6
output_dim = 2

model = TransformerModel(input_dim, d_model, nhead, num_layers, output_dim)
model = model.cuda()

# Input and target
x = torch.randint(0, input_dim, (64, 20)).cuda()
y = torch.randn(64, 20, 2).cuda()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for i in range(100):
    # Forward pass
    y_pred = model(x)
    loss = criterion(y_pred, y)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    
# Define the training function
def train(model, train_data, test_data, num_epochs, batch_size, learning_rate):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for acce_data, magn_data, ahrs_data, labels in train_loader:
            optimizer.zero_grad()
            input_seq = torch.cat((acce_data, magn_data, ahrs_data), dim=1)
            input_seq = pad_sequence(input_seq, batch_first=True)
            predictions = model(input_seq)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        model.eval()
        test_loss = 0
        for acce_data, magn_data, ahrs_data, labels in test_loader:
            input_seq = torch.cat((acce_data, magn_data, ahrs_data), dim=1)
            input_seq = pad_sequence(input_seq, batch_first=True)
            predictions = model(input_seq)
            loss = criterion(predictions, labels)
            test_loss += loss.item()
        test_loss /= len(test_loader)
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# Define the testing function
# Define the testing function
def test(model, test_data, batch_size):
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    model.eval()
    test_loss = 0
    test_x_preds = []
    test_y_preds = []
    test_x_gts = []
    test_y_gts = []
    with torch.no_grad():
        for acce_data, magn_data, ahrs_data, x_gt, y_gt in test_loader:
            acce_data = acce_data.to(device)
            magn_data = magn_data.to(device)
            ahrs_data = ahrs_data.to(device)
            x_gt = x_gt.to(device)
            y_gt = y_gt.to(device)
            x_pred, y_pred = model(acce_data, magn_data, ahrs_data)
            test_loss += F.mse_loss(x_pred, x_gt) + F.mse_loss(y_pred, y_gt)
            test_x_preds.append(x_pred.cpu().numpy())
            test_y_preds.append(y_pred.cpu().numpy())
            test_x_gts.append(x_gt.cpu().numpy())
            test_y_gts.append(y_gt.cpu().numpy())
    test_x_preds = np.concatenate(test_x_preds)
    test_y_preds = np.concatenate(test_y_preds)
    test_x_gts = np.concatenate(test_x_gts)
    test_y_gts = np.concatenate(test_y_gts)
    return test_loss / len(test_data), test_x_preds, test_y_preds, test_x_gts, test_y_gts


def run():
    print("Run  Started -!!!")
    # Enter code here
    # Define the hyperparameters
    input_size = 3
    hidden_size = 64
    num_layers = 2
    num_heads = 8
    batch_size = 16
    num_epochs = 10
    learning_rate = 0.01
    # Load the data
    train_data, test_data = load_data()

    # Create the dataset and data loader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = Transformer(input_size, hidden_size, num_layers, num_heads).to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer)
        test_loss, test_x_preds, test_y_preds, test_x_gts, test_y_gts = test(model, test_data, batch_size)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'checkpoint_{epoch}.pth')
    
    #Plot the train and test loss
    plt.plot(range(num_epochs), train_losses, label='Training Loss')
    plt.plot(range(num_epochs), test_losses, label='Testing Loss')
    plt.legend()
    plt.show()

    #Compare the prediction with the ground truth
    pred = model(test_data)
    ground_truth = test_data.y
    plt.scatter(pred[:, 0], pred[:, 1], label='Prediction')
    plt.scatter(ground_truth[:, 0], ground_truth[:, 1], label='Ground Truth')
    plt.legend()
    plt.show()

    print(f'Final Test Loss: {test_losses[-1]:.4f}, Final Test Accuracy: {test_acc:.4f}')

    #Save the trained model
    torch.save(model.state_dict(), 'final_model.pth')

    #Print the model architecture
    print(model)

    #Close the CUDA device
    torch.cuda.empty_cache()
    torch.cuda.device_reset()
    
    print("Run Finished Successfully !!!")


if __name__ == '__main__':
    print("Program  Started !!!")
    run()
    print("Program Finished Successfully !!!") 