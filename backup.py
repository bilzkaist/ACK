import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the custom dataset class
class LocationDataset(Dataset):
    def __init__(self, acce_datas, magn_datas, ahrs_datas, waypoints):
        self.acce_datas = acce_datas
        self.magn_datas = magn_datas
        self.ahrs_datas = ahrs_datas
        self.waypoints = waypoints
        
    def __len__(self):
        return len(self.acce_datas)
    
    def __getitem__(self, idx):
        acce_data = self.acce_datas[idx]
        magn_data = self.magn_datas[idx]
        ahrs_data = self.ahrs_datas[idx]
        waypoint = self.waypoints[idx]
        
        return acce_data, magn_data, ahrs_data, waypoint

# Define the transformer neural network
class TransformerNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, pf_dim):
        super(TransformerNet, self).__init__()
        
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(input_size, num_heads, pf_dim), num_layers)
        self.fc = nn.Linear(input_size, 2) # 2 for x and y
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x[:, -1, :]) # take the last output of the encoder as the final prediction
        return x

# Define the training function
def train(model, data_loader, criterion, optimizer, device):
    model.train()
    
    for acce_data, magn_data, ahrs_data, waypoint in data_loader:
        acce_data = acce_data.to(device)
        magn_data = magn_data.to(device)
        ahrs_data = ahrs_data.to(device)
        waypoint = waypoint.to(device)
        
        optimizer.zero_grad()
        
        # concatenate the input data
        input_data = torch.cat((acce_data, magn_data, ahrs_data), dim=1)
        
        output = model(input_data)
        loss = criterion(output, waypoint)
        loss.backward()
        optimizer.step()

# Define the testing function
def test(model, data_loader, criterion, device
