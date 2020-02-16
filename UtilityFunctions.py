#
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

#
def PytorchArray (labels, randomizedDataSet):
    randomizedDataSet_np = np.array(randomizedDataSet).astype(np.float32)
    labels_np = np.array(labels)
    randomizedDataSet_pytorch = torch.from_numpy(randomizedDataSet_np)
    labels_pytorch = torch.from_numpy(labels_np)    
    return randomizedDataSet_pytorch, labels_pytorch
