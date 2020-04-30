import numpy as np
import torch 

#randomize labels
def randomization(labelList, originalDataSet):
    randomizedDataSet = []
    for i, do_flip in enumerate(labelList):    
        #if its a 1 that means the back half won so flip
        x = originalDataSet[i] 
        if do_flip == 1:   
            x_rand = x[n_feat // 2:] + x[0:n_feat // 2]
        else:
            x_rand = x
        randomizedDataSet.append(x_rand)
    return randomizedDataSet

#turn into pytorch array
def PytorchArray (labels, randomizedDataSet):
    randomizedDataSet_np = np.array(randomizedDataSet).astype(np.float32)
    labels_np = np.array(labels)
    randomizedDataSet_pytorch = torch.from_numpy(randomizedDataSet_np)
    labels_pytorch = torch.from_numpy(labels_np)    
    return randomizedDataSet_pytorch, labels_pytorch

#load in processed data from CollegeBasketballPredictor.ipynb
def load_marchmadness():
    #features from training, test, validation sets
    featurestrainingData = torch.load('featurestraining_pytorch.pt')
    featurestestData = torch.load('featurestest_pytorch.pt')
    featuresvalidationData = torch.load('featuresvalidation_pytorch.pt')
    
    #labels from training, test, validation sets
    labelstrainingData = torch.load('labelstraining_pytorch.pt')
    labelstestData = torch.load('labelstest_pytorch.pt')
    labelsvalidationData = torch.load('labelsvalidation_pytorch.pt')
    
    return featurestrainingData, labelstrainingData, featurestestData, labelstestData, featuresvalidationData, labelsvalidationData

def build_model(input_dim, output_dim):
    # We don't need the softmax layer here since CrossEntropyLoss already
    # uses it internally.
    model = torch.nn.Sequential()
    model.add_module("linear",
                     torch.nn.Linear(input_dim, output_dim, bias=False))
    return model

def train(model, loss, optimizer, x, y):
    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model.forward(x)
    output = loss.forward(fx, y)

    # Backward
    output.backward()

    # Update parameters
    optimizer.step()

    return output.item()

def predict(model, x):
    output = model.forward(x)
    return output.data.numpy().argmax(axis=1)

