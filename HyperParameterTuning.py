#collect all 100 epochs of the number and store number into array
#use function and return the biggest value in the array

import numpy as np

import torch
from torch.autograd import Variable
from torch import optim

from UtilityFunctions import load_marchmadness
from UtilityFunctions import build_model
from UtilityFunctions import train
from UtilityFunctions import predict

#maybe its the cost
lr_list = [0.0001, 0.001, 0.01, 0.1, 1.0]

def main():
    torch.manual_seed(42)
    #X is data Y is label
    trX, trY, teX, teY, valX, valY = load_marchmadness()
    
    n_examples, n_features = list(trX.size())[0], list(trX.size())[1]
    n_classes = 2
    model = build_model(n_features, n_classes)
    loss = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    batch_size = 100

    for i in range(100):
        cost = 0.
        num_batches = n_examples // batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += train(model, loss, optimizer,
                          trX[start:end], trY[start:end])
        #use validation set
        predY = predict(model, valX)
        #changed print to return
        return("Epoch %d, cost = %f, acc = %.2f%%"
              #.cpu().numpy() turns it back into a numpy array
              % (i + 1, cost / num_batches, 100. * np.mean(predY == valY.cpu().numpy())))

result_list = []
if __name__ == "__main__":
    main()
    
    
    #for i in lr_list:
    #result = main(lr_list[i])
#result_list.append(result)
