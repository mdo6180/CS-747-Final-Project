from dataset import MNISTDataset, NotMNISTDataset
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from torch import flatten
import torch
import torch.optim as optim

import os


def get_accuracy(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    
    correct_pred = 0 
    n = 0
    
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            y_true = y_true.to(device)

            _, y_prob = model(X)

            # this is equivalent to argmax
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    if torch.cuda.is_available() is True:
        return correct_pred.cpu().detach().numpy() / n
    else:
        return correct_pred.detach().numpy() / n


def generate_plot(
    x_data, y_data, x_axis_label, y_axis_label, title, filename=None
):
    if os.path.exists("./plots") is False:
        os.makedirs("./plots")

    fig = plt.figure()
    plt.plot(x_data, y_data)
    plt.title(title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    if filename is not None:
        plt.savefig(f"./plots/{filename}")
    
    plt.close()
    

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs

class LeNet(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.relu1 = nn.ReLU(),
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.relu2 = nn.ReLU(),
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc_relu1 = nn.ReLU(),
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc_relu2 = nn.ReLU(),
        self.fc3 = nn.Linear(in_features=84, out_features=n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = flatten(x, 1) # flatten all dimensions except batch
        
        x = self.fc1(x)
        x = self.fc_relu1(x)
        x = self.fc2(x)
        x = self.fc_relu2(x)

        logits = self.fc3(x)
        probs = F.softmax(logits, dim=1)

        return logits, probs

class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)),
                    ("relu1", nn.ReLU()),
                    ("maxpool1", nn.MaxPool2d(kernel_size=2)),
                    ("conv2", nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)),
                    ("relu2", nn.ReLU()),
                    ("maxpool2", nn.MaxPool2d(kernel_size=2))
                ]
            )
        )

        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(in_features=16 * 5 * 5, out_features=120)),
                    ("fc_relu1", nn.ReLU()),
                    ("fc2", nn.Linear(in_features=120, out_features=84)),
                    ("fc_relu2", nn.ReLU())
                    ("fc3", nn.Linear(in_features=84, out_features=n_classes))
                ]
            )
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs


LeNet5_transform = transforms.Compose(
        [
            transforms.ToTensor(), 
            transforms.Resize((32,32))
        ]
    )

if __name__ == "__main__":

    # how to run project
    # 1. cd into project folder
    # 2. activate environment using "$ pipenv shell"

    # how to deactivate environment:
    # - run "$ deactivate"
        
    if os.path.exists("./saved_models") is False:
        os.makedirs("./saved_models")

    if torch.cuda.is_available() is True:
        print("Training on Nvidia GPU")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    #model = LeNet5(n_classes=10)
    model.to(device)

    train_dataset = MNISTDataset(split="train", transform=LeNet5_transform) 
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=100)  

    test_dataset = MNISTDataset(split="test", transform=LeNet5_transform) 
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=100)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    losses = []
    test_accuracy_scores = []
    train_accuracy_scores = []

    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch_index, batch in enumerate(train_dataloader):
            # get the inputs; batch is a tuple of (images, labels)
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward propagation
            logits, probs = model(images)

            # calculate loss and backward propagation
            loss = criterion(logits, labels)
            loss.backward()

            # optimize
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        train_accuracy = get_accuracy(model, train_dataloader, device)
        test_accuracy = get_accuracy(model, test_dataloader, device)
        loss = round(running_loss, 4)
        print(f'epoch: {epoch + 1} | loss: {loss} | training accuracy {train_accuracy} | testing accuracy: {test_accuracy}')

        test_accuracy_scores.append(round(test_accuracy, 4))
        train_accuracy_scores.append(round(train_accuracy, 4))
        losses.append(loss)

        running_loss = 0.0

    torch.save(model.state_dict(), "./saved_models/10-classes-MNIST.pth")

    epochs = list(range(1, num_epochs+1))

    generate_plot(
        x_data=epochs, y_data=losses, x_axis_label="Epochs", title="Training Loss", 
        y_axis_label="Cross Entropy Loss", filename="10-mnist-training-loss.png",
    )

    generate_plot(
        x_data=epochs, y_data=train_accuracy_scores, x_axis_label="Epochs", title="Training Accuracy", 
        y_axis_label="Percent Accuracy", filename="10-mnist-training-accuracy.png",
    )

    generate_plot(
        x_data=epochs, y_data=test_accuracy_scores, x_axis_label="Epochs", title="Testing Accuracy", 
        y_axis_label="Percent Accuracy", filename="10-mnist-testing-accuracy.png",
    )
