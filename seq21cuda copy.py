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

def getSample(seqL,k,testFlag=False):
    #returns a random sequence of integers of length = seqL
    kthInt=0
    x =  torch.zeros(seqL,10).cuda()
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
    y = torch.tensor(kthInt, dtype=torch.float32).cuda()  #target is the number at kth position in the sequence 
    #y = torch.zeros(10, dtype=torch.float32)
    #y[kthInt] = 1
    #y = y.cuda()    
    return x,y

class Memorize (nn.Module):
    def __init__(self,stateDim):
        super(Memorize, self).__init__()
        self.stateDim = stateDim
        self.inputDim = 10  # integer is represented as 1 hot vector of dimension=10
        self.outputDim = 10  # 10 nodes for 10 classes
        # currently the model uses the 'LSTM' cell. You could try
        # others like: tanh, GRU. See: https://github.com/pytorch/examples/blob/master/word_language_model/model.py#L11
        self.lstm = nn.LSTM(self.inputDim, self.stateDim ).cuda()
        self.outputLayer = nn.Linear(self.stateDim, self.outputDim).cuda()
        self.softmax = nn.Softmax(dim=1).cuda()

        

    def forward(self, x):
        # x is of shape (batchSize, sequenceLen, inputDim)
        # the lstm layer expects the input in shape (sequenceLen, batchSize, inputDim)
        x = x.permute(1, 0, 2)
        lstmOut, _ = self.lstm(x)
        # lstmOut is of shape (sequenceLen, batchSize, stateDim)
        # we want the final output of the lstm at the last time step
        finalOut = lstmOut[-1]
        # pass the final output through the output layer
        output = self.outputLayer(finalOut)
        #output = self.softmax(output)
        return output


    def save_checkpoint(self, epoch, loss, path):
        torch.save({
        'epoch': epoch,
        'model_state_dict': self.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'loss': loss
        }, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return epoch, loss




def run():
    print("Start running... !!!")
    # Enter code here
    # set here the size of the RNN state:
    stateSize = 20

    # set here the size of the binary strings to be used for training:
    k=2 # we want the RNN to remember the number at 2nd position
    minSeqLength = 3
    maxSeqLength = 8

    # sequenceLengths would be in range in range minSeqLength - maxSeqLength
    # convert the model to run on CUDA:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Memorize(stateSize).to(device)
    print ('Model initialized')

    # create the loss-function:
    lossFunction = nn.CrossEntropyLoss().to(device) # or nn.CrossEntropyLoss() -- see question #2 below

    # uncomment below to change the optimizers:
    # optimizer = optim.SGD(model.parameters(), lr=3e-2, momentum=0.8)
    optimizer = optim.Adam(model.parameters(),lr=0.01)
    iterations = 500
    min_epochs = 20
    num_epochs,totalLoss = 0,float("inf")
    lossList = []
    
    # epoch
    while num_epochs < min_epochs:
        print("[epoch %d/%d] Avg. Loss for last 500 samples = %lf"%(num_epochs+1,min_epochs,totalLoss))
        num_epochs += 1
        totalLoss = 0
        for i in range(0,iterations):
            # get a new random training sample:
            sequenceLength = random.randint(minSeqLength,maxSeqLength)
            x,y = getSample(sequenceLength,k)
            # convert the input and target to CUDA tensors:
            x = x.to(device)
            y = y.to(device)

            model.zero_grad()

            pred = model(x)
            print("pred = ",pred)
            _, predicted = torch.max(pred.data, 1)
            y = y.view(-1)

            print("x = ",x)
            print("pred = ",pred)
            #print("pred item = ",print("pred.item() = ",pred.detach().numpy()[0].item()))
            print("y = ",y) # .item())
            print("predicted = ",predicted)
            # predicted_value = predicted[0].item()
            predicted_tensor = torch.tensor(predicted, dtype=torch.float32)
            predicted_value = predicted_tensor[0].item()
            print("predicted_value = ",predicted_value) # prints 2
            print("y = ",y)#.item())

            # compute the loss:
            try:
                loss = lossFunction(predicted_value,y)
            except:
                loss = lossFunction(pred,y)
            # loss = lossFunction(pred.view(-1, 10).float(), y.long().view(-1, 1))


            


            totalLoss += loss.item() 
            optimizer.zero_grad()
            # perform the backward pass:
            loss.backward()
            # update the weights:
            optimizer.step()
        totalLoss=totalLoss/iterations
        lossList.append(int(totalLoss))

    # epoch end 


    print('Training finished!')
    epochs = np.arange(1,21)

    # plot the loss over epcohs:
    plt.plot(epochs,lossList)
    plt.xlabel('epochs'); plt.ylabel('loss'); plt.xticks(epochs,epochs)
    plt.ylim([0,5])

    testSeqL = 6
    x,y = getSample(testSeqL,k,testFlag=True)

    # convert the input to a CUDA tensor:
    x = x.to(device)
    pred = model(x)
    # ind= torch.argmax(pred)
    _, ind = torch.max(pred, 1)
    print ( 'number at kth position is ',int(ind))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')
    print("Stop running... !!!")


if __name__ == '__main__':
    print("Program  Started !!!")
    print("Is Cuda Available : ", torch.cuda.is_available())
    run()
    print("Program Finished Successfully !!!") 