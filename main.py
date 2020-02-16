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
