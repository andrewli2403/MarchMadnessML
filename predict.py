import numpy as np

import torch
from torch.autograd import Variable
from torch import optim

from UtilityFunctions import predict


if __name__== “__main__”:
    optimal_lr = torch.load('optimal_learningmodel.pt')

    
    
    example_featurevector = []
    predict(optimal_lr, example_featurevector)
    #make feature vector
    #features taken out: [Wteam, Wfgm, Wfga, Wfgm3, Wfga3, Wftm, Wfta, Wor, Wdr, Wast, Wto, Wstl, Wblk, Wpf, Lteam, Lfgm, Lfga, Lfgm3, Lfga3, Lftm, Lfta, Lor, Ldr, Last, Lto, Lstl, Lblk, Lpf]
