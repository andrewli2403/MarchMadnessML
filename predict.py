import numpy as np

import torch
from torch.autograd import Variable
from torch import optim

from UtilityFunctions import predict


if __name__== “__main__”:
    optimal_lm = torch.load('optimal_learningmodel.pt')

    
    
    
    VCU = []
    UCF = []
    
    feature = []
    X = torch.Tensor(feature)
    predict(optimal_lm, X)
    #make feature vector
    #features taken out: [Wteam, Wfgm, Wfga, Wfgm3, Wfga3, Wftm, Wfta, Wor, Wdr, Wast, Wto, Wstl, Wblk, Wpf, Lteam, Lfgm, Lfga, Lfgm3, Lfga3, Lftm, Lfta, Lor, Ldr, Last, Lto, Lstl, Lblk, Lpf]
