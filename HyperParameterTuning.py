#collect all 100 epochs of the lr and determine the max accuracy
#repeat for rest of lr and return the max accuracy from a certain lr

import numpy as np

import torch
from torch.autograd import Variable
from torch import optim

from UtilityFunctions import load_marchmadness
from UtilityFunctions import build_model
from UtilityFunctions import train
from UtilityFunctions import predict

#learning rate list
lr_list = [0.0001, 0.001, 0.01, 0.1, 1.0]
result_list = []

def main(lr_value):
    torch.manual_seed(42)
    #X is data Y is label
    trX, trY, teX, teY, valX, valY = load_marchmadness()
    
    n_examples, n_features = list(trX.size())[0], list(trX.size())[1]
    n_classes = 2
    model = build_model(n_features, n_classes)
    loss = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
    optimizer = optim.SGD(model.parameters(), lr= lr_value, momentum=0.9)
    batch_size = 100

    acc = []
    max_lr = 0;
    for i in range(100):
        cost = 0.
        num_batches = n_examples // batch_size
        #use validation set
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += train(model, loss, optimizer,
                          trX[start:end], trY[start:end])
    
        predY = predict(model, valX)
        
        #collects accuracy for each epoch
        acc.append(100. * np.mean(predY == valY.cpu().numpy()))
        
        print("Epoch %d, cost = %f, acc = %.2f%%"
              #.cpu().numpy() turns it back into a numpy array
              #each refers to the printed values
              % (i + 1, cost / num_batches, 100. * np.mean(predY == valY.cpu().numpy())))

#finds maximum accuracy value and adds to result_list
    max_lr = max(acc)
    result_list.append(max_lr)

if __name__ == "__main__":
    for value in lr_list:
        main(value)

#associates learning rate and its respective maximum accuracy
#{learning rate: accuracy}
lr_acc = {}
for i in range(len(lr_list)):
    lr_acc.update({lr_list[i]:result_list[i]})
print(lr_acc)

#finds the learning rate that has the highest accuracy
optimal_lr = max(lr_acc, key=lr_acc.get)
print(optimal_lr)
