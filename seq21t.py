import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

random.seed(10) 

class MemorizeDataset(Dataset):
    def __init__(self, seq_len, k):
        self.seq_len = seq_len
        self.k = k
        
    def __len__(self):
        return self.seq_len
    
    def __getitem__(self, idx):
        x = torch.zeros(self.seq_len, 10)
        for i in range(self.seq_len):
            random_int = random.randint(0, 9)
            if i == self.k - 1:
                kth_int = random_int
            x[i, random_int] = 1
        
        x = x.unsqueeze(0)
        y = torch.tensor([kth_int])
        
        return x, y

class Memorize(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(Memorize, self).__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.output_layer = nn.Linear(d_model, 10)
        
    def forward(self, x):
        x = self.transformer(x)
        x = self.output_layer(x)
        
        return x

# set model parameters
d_model = 20
nhead = 8
num_layers = 6

# set dataset parameters
k = 2
min_seq_len = 3
max_seq_len = 8

# create dataset
dataset = MemorizeDataset(seq_len=max_seq_len, k=k)

# create model
model = Memorize(d_model, nhead, num_layers).cuda()

# create loss function
loss_function = nn.CrossEntropyLoss()

# create optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# create data loader
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# train model
num_epochs = 20
for epoch in range(num_epochs):
    total_loss = 0
    for x, y in data_loader:
        x, y = x.cuda(), y.cuda()
        
        optimizer.zero_grad()
        
        y_hat = model(x)
        loss = loss_function(y_hat, y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    #print("[epoch %d/%
    print("[epoch %d/20] Avg. Loss = %lf"%(epoch+1, total_loss / len(data_loader)))
