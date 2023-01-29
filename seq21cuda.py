# coding: utf-8
# =============================================================================
# Make an RNN output kth integer in a sequence of N integers ( N > k)
# Sequences could be of any length

# ==============================================================================


import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import random
import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
random.seed( 10 ) # set the random seed (for reproducibility)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getSample(seqL,k,testFlag=False):
    #returns a random sequence of integers of length = seqL
    kthInt=0
    x =  torch.zeros(seqL,10).to(device)
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
    # y=torch.tensor([kthInt]) #target is the number at kth position in the sequence 
    y=torch.tensor([kthInt]).to(device)
    
    return x,y

class Memorize (nn.Module):
    def __init__(self,stateDim):
        super(Memorize, self).__init__()
        self.stateDim = stateDim
        self.inputDim = 10  # integer is represented as 1 hot vector of dimension=10
        self.outputDim = 10  # 10 nodes for 10 classes
        # currently the model uses the 'LSTM' cell. You could try
        # others like: tanh, GRU. See: https://github.com/pytorch/examples/blob/master/word_language_model/model.py#L11
        self.lstm = nn.LSTM(self.inputDim, self.stateDim )
        self.outputLayer = nn.Linear(self.stateDim, self.outputDim)
        self.softmax = nn.Softmax()
        

    def forward(self, x):
        """
        X: [L,B,inputDim(=10)] dimensional input tensor
            L: Sequence length
            B: is the "batch" dimension. As we are training on 
               single examples, B = 1 for us.
        """
        lstmOut,_ = self.lstm(x)
        L,B,D  = lstmOut.size(0),lstmOut.size(1),lstmOut.size(2) # L is seq len, B is batch size and D is feature dimension
        #lstmOut holds the outputs at all timesteps but we require  only the output at last time step (L-1)
        lstmOut_lastTimeStep = lstmOut[L-1,0,:]
        #print (lstmOut_lastTimeStep.size())
        
        #lstmOut = lstmOut.view(L*B,D)
        #outputLayerActivations = self.outputLayer(lstmOut_lastTimeStep)
        #outputLayerActivations = self.outputLayer(lstmOut_lastTimeStep.clone())
        outputLayerActivations = self.outputLayer(lstmOut_lastTimeStep.detach())


        #outputSoftMax=self.softmax(outputLayerActivations)
        # project lstm states to "output"
        
    
        return outputLayerActivations.unsqueeze(0)

# set here the size of the RNN state:
stateSize = 20
# set here the size of the binary strings to be used for training:
k=2 # we want the RNN to remember the number at 2nd position
minSeqLength = 3
maxSeqLength = 8

## sequenceLengths would be in range in range minSeqLength - maxSeqLength

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create the model:
model = Memorize(stateSize)
model = model.to(device)
print ('Model initialized')

# create the loss-function:
lossFunction = nn.CrossEntropyLoss() # or nn.CrossEntropyLoss() -- see question #2 below

learning_rate = 3e-2

# uncomment below to change the optimizers:
# optimizer = optim.SGD(model.parameters(), lr=3e-2, momentum=0.8)
optimizer = optim.Adam(model.parameters(),lr=0.01)
# optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

iterations = 500
min_epochs = 5
num_epochs,totalLoss = 0,float("inf")
lossList = []
while num_epochs < min_epochs:
    print("[epoch %d/%d] Avg. Loss for last 500 samples = %lf"%(num_epochs+1,min_epochs,totalLoss))
    num_epochs += 1
    totalLoss = 0
    for i in range(0,iterations):
        # get a new random training sample:
        sequenceLength = random.randint(minSeqLength,maxSeqLength)
        x,y = getSample(sequenceLength,k)
        x=x.to(device)
        y=y.to(device)
        model.zero_grad()

        pred = model(x)

        # compute the loss:
        loss = lossFunction(pred,y)
        totalLoss += loss.item() #totalLoss += loss.data[0]
        optimizer.zero_grad()
        optimizer.step()
        
        # perform the backward pass:
        loss.backward()
        # update the weights:
        optimizer.step()
    totalLoss=totalLoss/iterations
    lossList.append(int(totalLoss))
print('Training finished!')
epochs =  np.arange(1,len(lossList)+1)
# plot the loss over epcohs:
#print("epochs = ",epochs)
#print("lossList = ",lossList)
plt.plot(epochs,lossList)
plt.xlabel('epochs'); plt.ylabel('loss'); plt.xticks(epochs,epochs)
plt.ylim([0,5]); 
#plt.show()


testSeqL = 6
x,y = getSample(testSeqL,k,testFlag=True)
x=x.to(device)
y=y.to(device)
pred = model(x)
# print("x = ",x)
# print("pred = ",pred)
# print("y = ",y.item())
ind=  torch.argmax(pred)
print( 'number at kth', k ,' position is ',int(ind))


