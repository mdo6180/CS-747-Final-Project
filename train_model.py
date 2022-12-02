from dataset import MNISTDataset, NotMNISTDataset
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F
from torch import flatten

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
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
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(), 
            transforms.Resize((32,32))
        ]
    )

    test_dataset = MNISTDataset(split="test", classes=[0,1,2], transform=transform) 
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=20)  

    for batch_index, batch in enumerate(test_dataloader):
        images = batch[0]
        labels = batch[1]

        for image, label in zip(images, labels):
            img = image.numpy()
            img = img.squeeze()

            plt.imshow(img, interpolation='nearest')
            plt.show()

            break

        break
    """

    # how to run project
    # 1. cd into project folder
    # 2. activate environment using "$ pipenv shell"

    # how to deactivate environment:
    # - run "$ deactivate"

    net = Net()

    """
    not_mnist_dataset = NotMNISTDataset(["A", "B", "C"])
    not_mnist_dataloader = DataLoader(dataset=not_mnist_dataset, batch_size=20)

    for batch_index, batch in enumerate(not_mnist_dataloader):
        images = batch[0]
        labels = batch[1]

        for image, label in zip(images, labels):
            img = image.numpy()
            img = img.squeeze()

            plt.imshow(img, interpolation='nearest')
            plt.show()

            break

        break 
    """