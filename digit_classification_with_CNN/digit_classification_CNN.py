# digit classification with CNN 

import torch
import pandas as pd
import matplotlib.pyplot as plt

# Load data(do not change)
data = pd.read_csv("src/mnist_train.csv")
train_data = data[:2000]
test_data = data[2000:2500]

trains_label = train_data['label']
trains_label = torch.tensor(trains_label)
test_label = test_data['label']
test_label = torch.tensor(test_label.to_numpy())

trains_data = train_data.drop(["label"],axis=1)
trains_data = trains_data.to_numpy()/255
test_data = test_data.drop(["label"],axis=1)
test_data = test_data.to_numpy()/255

trains_data = torch.tensor(trains_data,dtype=torch.float32).view(-1,1,28,28)
test_data = torch.tensor(test_data,dtype=torch.float32).view(-1,1,28,28)

# ----- Prepare Data ----- #
# step one: preparing your data including data normalization

# step two: transform np array to pytorch tensor

# ----- Build CNN Network ----- #
# Define your model here
class mymodel(torch.nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        # conv layers: feature extractor
        # using nn.Sequential can concate layer together and more less code
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size = 5),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(10, 20, kernel_size = 5),
            torch.nn.Dropout2d(0.5),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU()
        )
     # fc layers: classifier
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(50, 10),
        )


    def forward(self, x):
        x = self.conv_layers(x)
        # flatten the final output of conv_layers
        x = x.view(-1, 320)
        x = self.fc_layers(x)
        # softmax can output log probability of each potential classes
        return torch.nn.functional.log_softmax(x, dim = 1)

# Define our model
model = mymodel()
# Define your learning rate
learning_rate = 1e-3
# Define your optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
# Define your loss function
criterion = torch.nn.CrossEntropyLoss()

# ----- Complete PlotLearningCurve function ----- #
def PlotLearningCurve(epoch, trainingloss, testingloss):
    plt.title = "Learning Curve"
    xpixel = [i for i in range(1,101)]
    plt.plot(xpixel,trainingloss,color = 'b',linewidth=1.0,label="trainingloss")
    plt.plot(xpixel,testingloss,color = 'r',linewidth=1.0,linestyle='--', label="testingloss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc = "best")
    plt.show()
    

# ----- Main Function ----- #
trainingloss = []
testingloss = []
# Define number of iterations
epochs = 100
for epoch in range(1, epochs + 1):
    model.train()
    # step one : fit your model by using training data and get predict label
    y_pred = model(trains_data)
    # step two: calculate your training loss
    loss = criterion(y_pred, trains_label)
    # step three: calculate backpropagation
    loss.backward()
    # step four: update parameters
    optimizer.step()
    # step five: reset our optimizer
    optimizer.zero_grad()
    # step six: store your training loss
    trainingloss += loss.item(),
    # step seven: evaluation your model by using testing data and get the accuracy
    with torch.no_grad():
        model.eval()
        # predict testing data
        test_pred = model(test_data)
        # calculate your testing loss
        loss = criterion(test_pred, test_label)
        # store your testing loss
        testingloss += loss.item(),
        if epoch % 10 == 0:
            # get labels with max values
            _, test_pred = torch.max(test_pred, dim = 1)
            # calculate the accuracy
            acc = round((test_label == test_pred).type(torch.float32).mean().item() * 100, 2)
            print('Epoch:', epoch, 'Test Accuracy:', acc)
PlotLearningCurve(100,trainingloss, testingloss)