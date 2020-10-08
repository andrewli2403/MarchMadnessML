def main():
    torch.manual_seed(42)
    #X is data Y is label
    trX, trY, teX, teY, valX, valY = load_marchmadness()
    
    n_examples, n_features = list(trX.size())[0], list(trX.size())[1]
    #options of the possible outcomes: 0 first team, 1 second team
    n_classes = 2
    model = build_model(n_features, n_classes)
    loss = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
    #updating the model
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #how many samples you train at a time
    batch_size = 100

    for i in range(100):
        #loss: roughly proportional to how bad your model is
        cost = 0.
        #n_examples: your whole data set
        num_batches = n_examples // batch_size
        #guarentees you cover the whole data set
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += train(model, loss, optimizer,
                          trX[start:end], trY[start:end])
        predY = predict(model, teX)
        print("Epoch %d, cost = %f, acc = %.2f%%"
              #.cpu().numpy() turns it back into a numpy array
              % (i + 1, cost / num_batches, 100. * np.mean(predY == teY.cpu().numpy())))

main()
