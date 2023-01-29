import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys
import random

def getSample(seqL,k,testFlag=False):
    #returns a random sequence of integers of length = seqL
    kthInt=0
    x =  torch.zeros(seqL,10)
    for i in range(0,seqL):
        randomIntegerNumber = random.randint(0,9)
        if i==k-1:
            kthInt=randomIntegerNumber
        if testFlag:
            sys.stdout.write(str(randomIntegerNumber) + ' ')
        x[i,randomIntegerNumber-1] = 1

    if testFlag:
            sys.stdout.write('--> ' + str(kthInt) + '\n')
    x=x.unsqueeze(1) #extra dimension for Batch
    y=torch.tensor([kthInt]) #target is the number at kth position in the sequence 

    return x,y

class MemorizeTransformer(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, num_heads, dropout):
        super(MemorizeTransformer, self).__init__()
        print("input_size % num_heads = ", input_size % num_heads)
        #assert input_size % num_heads == 0, "input_size must be divisible by num_heads"
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(input_size, num_heads, hidden_size, dropout), num_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(output_size, num_heads, hidden_size, dropout), num_layers)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, src, tgt):
        output = self.encoder(src)
        output = self.decoder(tgt, output)
        output = self.output_layer(output)
        return output

class RandomSequenceDataset(Dataset):
    def __init__(self, seq_length, k, num_samples):
        self.seq_length = seq_length
        self.k = k
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x, y = getSample(self.seq_length, self.k)
        return x, y

input_size = 80
output_size = 10
hidden_size = 20
num_layers = 4
num_heads = 2
dropout = 0.1
if input_size % num_heads != 0:
    input_size = (input_size // num_heads + 1) * num_heads


stateSize = 20
k = 2
minSeqLength = 3
maxSeqLength = 8
batch_size = 32
num_samples = 5000

model = MemorizeTransformer(input_size, output_size, hidden_size, num_layers, num_heads, dropout)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

dataset = RandomSequenceDataset(maxSeqLength, k, num_samples)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(20):
    total_loss = 0
    for x, y in data_loader:
        optimizer.zero_grad()
        output = model(x, y)
        loss = loss_fn(output.view(-1, output_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("[epoch %d/20] Avg. Loss = %lf"%(epoch+1, total_loss / len(data_loader)))
