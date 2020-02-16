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
    
    return featurestrainingData, labelstrainingData, featurestestData, labelstestData, featuresvalidationData, labelsvalid

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
        predY = predict(model, teX)
        print("Epoch %d, cost = %f, acc = %.2f%%"
              #.cpu().numpy() turns it back into a numpy array
              % (i + 1, cost / num_batches, 100. * np.mean(predY == teY.cpu().numpy())))

main()
