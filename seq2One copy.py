import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(72)

def generate_sequence(seqlen: int):
    assert seqlen > 2, "The sequence should have a length of at least 3"
    # Randomize A/B placement in the sequence and randomly fill the rest with XYZ
    seq = random.choices(['X', 'Y', 'Z'], k=seqlen)
    label = random.choices(['A', 'B'], k=2)
    idx = sorted(random.sample(range(1, seqlen+1), 2)) # Indexes of A/B
    for i, val in enumerate(idx):
        seq[val-1] = label[i]
    return seq, label

def generate_sequence_batch(seqlen: int, nseq: int):
    # Initialize empty arrays for sequences and labels
    seqs = []
    labs = []
    # Fill arrays
    for i in range(nseq):
        seq, lab = generate_sequence(seqlen)
        seqs.append(seq)
        labs.append(lab)
    # Encode sequences using one-hot encoding and return
    seqs_encoded = [torch.eye(5)[np.array([['ABXYZ'.index(c) for c in s]], dtype='int32')] for s in seqs]
    labs_encoded = [torch.eye(4)[np.array([['AA','AB','BA','BB'].index(l) for l in lab]], dtype='int32')] for lab in labs]
    return seqs_encoded, labs_encoded



seqlen = 12
obsNum = 10000

# Train and test data sets with 10,000 observations each
Xtrain, ytrain = generate_sequence_batch(seqlen, obsNum)
Xtest, ytest = generate_sequence_batch(seqlen, obsNum)

# Define an accuracy measure
def accuracy(m, X, y):
    m.reset_parameters()
    return 100 * torch.mean((torch.argmax(m(X), dim=1) == torch.argmax(y, dim=1)).float())


# Set seed for replication
random.seed(72)

# Define sequence length
seqlen = 12

# Train and test data sets with 10'000 observations each
Xtrain, ytrain = generate_sequence_batch(seqlen, 10_000)
Xtest, ytest = generate_sequence_batch(seqlen, 10_000)

# Reshape the outputs for PyTorch
ytrain = torch.cat(ytrain, dim=1)
ytest = torch.cat(ytest, dim=1)

# Create train and test datasets for feedforward neural network 
# (each observation is a vector of length 12x5=60)
Xtrain_ffnn = torch.cat([torch.cat(x, dim=1) for x in Xtrain], dim=1)
Xtest_ffnn = torch.cat([torch.cat(x, dim=1) for x in Xtest], dim=1)

# Create feedforward neural network
ffnn = nn.Sequential(
    nn.Linear(5*seqlen, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 4)
)

# Initialize optimizer
opt_ffnn = optim.Adam(ffnn.parameters())

# Train the model for 100 epochs
epochs = 100
for epoch in range(1, epochs+1):
    # Train the model using batches of size 32
    for idx in torch.randperm(len(Xtrain_ffnn)).split(32):
        X, y = Xtrain_ffnn[idx], ytrain[idx]
        opt_ffnn.zero_grad()
        output = ffnn(X)
        loss = nn.functional.cross_entropy(output, y.argmax(1))
        loss.backward()
        opt_ffnn.step()

# Define accuracy function
def accuracy(m, X, y):
    with torch.no_grad():
        output = m(X)
        accuracy = (output.argmax(1) == y.argmax(1)).float().mean()
    return accuracy

# Compute accuracy of feedforward neural network
print(accuracy(ffnn, Xtrain_ffnn, ytrain)) # 46.16
print(accuracy(ffnn, Xtest_ffnn, ytest))   # 46.12


class Seq2One(nn.Module):
    def __init__(self,rnn,fc):
        super(Seq2One, self).__init__()
        self.rnn = rnn
        self.fc = fc
    def forward(self, X):
        # Run recurrent layers on all but final data point
        for x in X[1:end-1]:
            self.rnn(x)
        # Pass last data point through both recurrent and fully-connected layers
        self.fc(self.rnn(X[end]))

class Seq2One(nn.Module):
    def __init__(self):
        super(Seq2One, self).__init__()
        self.rnn = nn.Sequential(
            nn.RNN(5, 128, nonlinearity='relu'),
            nn.RNN(128, 128, nonlinearity='relu')
        )
        self.fc = nn.Linear(128, 4)

    def forward(self, X):
        out, _ = self.rnn(X)
        out = self.fc(out[-1])
        return out

seq2one = Seq2One()
opt_rnn = torch.optim.Adam(seq2one.parameters())
epochs = 10
for epoch in range(epochs):
    for idx in range(0, len(Xtrain), 32):
        X, y = Xtrain[idx:idx+32], ytrain[:, idx:idx+32]
        X = torch.stack([torch.cat([x[i] for x in X], dim=1) for i in range(seqlen)], dim=0)
        opt_rnn.zero_grad()
        loss = nn.functional.cross_entropy(seq2one(X), y.argmax(dim=1))
        loss.backward()
        opt_rnn.step()

Xtrain_rnn = [torch.cat([x[i] for x in Xtrain], dim=1) for i in range(seqlen)]
Xtest_rnn = [torch.cat([x[i] for x in Xtest], dim=1
   


#..............

# Define the RNN layer
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, activation):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, nonlinearity=activation)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        return out
    
# Define the sequence-to-one network
class Seq2One(nn.Module):
    def __init__(self, rnn, dense):
        super(Seq2One, self).__init__()
        self.rnn = rnn
        self.dense = dense
        
    def forward(self, x):
        out = self.rnn(x)
        out = self.dense(out[:,-1,:])
        return out

# Create the sequence-to-one network
seq2one = Seq2One(
    RNN(5, 128, 'relu'),
    nn.Linear(128, 4)
)

# Move the model to the GPU
seq2one = seq2one.to('cuda')

# Define the optimizer
opt_rnn = optim.Adam(seq2one.parameters())

# Define the loss function
loss_fn = nn.CrossEntropyLoss()

# Train the model for 10 epochs
epochs = 10
for epoch in range(epochs):
    # Train the model using batches of size 32
    for idx in Iterators.partition(shuffle(range(Xtrain.shape[0])), 32):
        # Reset the hidden state
        seq2one.hidden = None
        X, y = Xtrain[idx], ytrain[:, idx]
        X = X.transpose(1,0) # Reshape X for RNN format
        # Move data to GPU
        X = X.to('cuda')
        y = y.to('cuda')
        # Perform a forward pass and compute the loss
