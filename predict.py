import numpy as np

import torch
from torch.autograd import Variable
from torch import optim

from UtilityFunctions import predict

from Simulator import game

if __name__== "__main__":
    optimal_lm = torch.load('optimal_learningmodel.pt')

    VCU = []
    UCF = []
    
    #Syracuse vs Memphis
    features = [[1272.0, 3.0, 26.0, 62.0, 8.0, 20.0, 10.0, 19.0, 15.0, 28.0, 16.0, 13.0, 4.0, 4.0, 18.0, 1393.0, 7.0, 24.0, 67.0, 6.0, 24.0, 9.0, 20.0, 20.0, 25.0, 7.0, 12.0, 8.0, 6.0, 16.0]]
    
    feature = game
    #makes feature vector into two dimensional array torch.size([1, 30])
    X = torch.tensor(feature)
    print(type(X))
    #0 means first team won, 1 means second team won
    print(predict(optimal_lm, X))

