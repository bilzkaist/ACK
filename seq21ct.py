import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim

random.seed(10) # set the random seed (for reproducibility)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def createInputTensor(i, sequenceLength, k):
    inputTensor = torch.zeros(sequenceLength, 10)
    binary_repr = format(i, 'b').zfill(sequenceLength)
    for j, digit in enumerate(binary_repr):
        inputTensor[j, int(digit)] = 1
    return inputTensor

def createTargetTensor(i, sequenceLength, k):
    targetTensor = torch.zeros(1, 10)
    binary_repr = format(i, 'b').zfill(sequenceLength)
    targetTensor[0, int(binary_repr[k-1])] = 1
    return targetTensor

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
    y=torch.tensor([kthInt]).to(device)
    
    return x,y

class Memorize(nn.Module):
    def __init__(self,stateDim):
        super(Memorize, self).__init__()
        self.stateDim = stateDim
        self.inputDim = 10  # integer is represented as 1 hot vector of dimension=10
        self.outputDim = 10  # 10 nodes for 10 classes
        self.transformer = nn.Transformer(d_model=stateDim, nhead=8)
        self.outputLayer = nn.Linear(self.stateDim, self.outputDim)
        self.softmax = nn.Softmax()

    def forward(self, x):
        """
        X: [L,B,inputDim(=10)] dimensional input tensor
            L: Sequence length
            B: is the "batch" dimension. As we are training on 
               single examples, B = 1 for us.
        """
        transformerOut = self.transformer(x)
        L,B,D = transformerOut.size(0),transformerOut.size(1),transformerOut.size(2) # L is seq len, B is batch size and D is feature dimension
        transformerOut_lastTimeStep = transformerOut[L-1,0,:]
        outputLayerActivations = self.outputLayer(transformerOut_lastTimeStep.detach())
        return outputLayerActivations.unsqueeze(0)

# set here the size of the transformer state:
stateSize = 20
# set here the size of the binary strings to be used for training:
k=2 # we want the transformer to remember the number at 2nd position
minSeqLength = 3
maxSeqLength = 8

## sequenceLengths would be in range in range minSeqLength - maxSeqLength

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create an instance of the transformer model
transformerModel = Memorize(stateSize=stateSize, k=k)
transformerModel = transformerModel.to(device)

# loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(transformerModel.parameters())

# train the model
numEpochs = 100
for epoch in range(numEpochs):
    totalLoss = 0
    for sequenceLength in range(minSeqLength, maxSeqLength+1):
        for i in range(2**sequenceLength):
            # create the input tensor
            inputTensor = createInputTensor(i, sequenceLength, k)
            inputTensor = inputTensor.to(device)
            # create the target tensor
            targetTensor = createTargetTensor(i, sequenceLength, k)
            targetTensor = targetTensor.to(device)

            # forward pass
            output = transformerModel(inputTensor)

            # compute loss
            loss = criterion(output, targetTensor)

            # backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        totalLoss += loss.item()

    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{numEpochs}, Loss: {totalLoss/(maxSeqLength-minSeqLength+1)}')

print('Training complete')

# test the model
transformerModel.eval()
with torch.no_grad():
    for sequenceLength in range(minSeqLength, maxSeqLength+1):
        for i in range(2**sequenceLength):
            inputTensor = createInputTensor(i, sequenceLength, k)
            inputTensor = inputTensor.to(device)
            output = transformerModel(inputTensor)
            output = torch.sigmoid(output)
            output = output.cpu().numpy()
            target = createTargetTensor(i, sequenceLength, k)
            target = target.cpu().numpy()
        print(f'input: {i}, target: {target[0][0]}, output: {output[0][0]}')

# save the model
torch.save(transformerModel.state_dict(), 'transformerModel.pth')

# test the model with the saved weights
transformerModel2 = Memorize(stateSize=stateSize, k=k)
transformerModel2.load_state_dict(torch.load('transformerModel.pth'))
transformerModel2 = transformerModel2.to(device)
transformerModel2.eval()
with torch.no_grad():
    for sequenceLength in range(minSeqLength, maxSeqLength+1):
        for i in range(2**sequenceLength):
            inputTensor = createInputTensor(i, sequenceLength, k)
            inputTensor = inputTensor.to(device)
            output = transformerModel2(inputTensor)
            output = torch.sigmoid(output)
            #output = output.
            output = output.squeeze(0).detach().cpu().numpy()
            target = createTargetTensor(i, k)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            totalLoss += loss.item()
        print("Sequence Length: ", sequenceLength, " Loss: ", totalLoss/(2**sequenceLength))

#Now we test the model on some test inputs
testCases = [
{"input": 0b0010101, "output": 0b101},
{"input": 0b0111111, "output": 0b111},
{"input": 0b1111111, "output": 0b111},
{"input": 0b0000000, "output": 0b000},
{"input": 0b1000111, "output": 0b001},
]

for testCase in testCases:
    inputTensor = createInputTensor(testCase["input"], 7, k)
    inputTensor = inputTensor.to(device)
    output = transformerModel2(inputTensor)
    output = torch.sigmoid(output)
    output = output.squeeze(0).detach().cpu().numpy()
    output = (output > 0.5).astype(int)
print("Input: ", testCase["input"], " Output: ", output, " Target: ", testCase["output"])

#Now the model is trained and you can use it to predict the kth digit of any binary string of length greater than equal to minSeqLength and less than or equal to maxSeqLength
#you can also save the model using 
torch.save(transformerModel2.state_dict(),"modelFileName.pt") 
#and load the model using 
transformerModel2.load_state_dict(torch.load("modelFileName.pt"))

binaryString = "0101010101"
inputTensor = createInputTensor(int(binaryString,2), len(binaryString), k)
inputTensor = inputTensor.to(device)
predictedOutput = transformerModel2(inputTensor)
predictedOutput = torch.sigmoid(predictedOutput)

torch.jit.save(transformerModel2, "modelFileName.pt")
transformerModel2 = torch.jit.load("modelFileName.pt")

predictedOutput = transformerModel2(inputTensor)
predictedOutput = torch.sigmoid(predictedOutput)

predictedDigit = round(predictedOutput.item())
print(predictedDigit)