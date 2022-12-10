from mlxtend.data import loadlocal_mnist
import types
import torch
import os
import numpy as np

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from matplotlib import pyplot as plt
from PIL import Image

import os


class MNISTDataset(Dataset):
    # mnist test dataset: 10,000 samples
    # class 0: 0-979
    # class 1: 980-2114
    # class 2: 2115-3146
    # class 3: 3147-4156
    # class 4: 4157-5138
    # class 5: 5139-6038
    # class 6: 6039-6988
    # class 7: 6989-8016
    # class 8: 8017-8990
    # class 9: 8991-10000

    # mnist train dataset: 60,000 samples

    def __init__(
        self, 
        split: str, 
        classes: list = None, 
        num_samples: int = None, 
        transform: types.ModuleType = None, 
        sorted: bool = False
    ):
    
        """Load mnist data.
        Parameters
        ----------
        split : str
            train or test
        classes : list, optional
            classes to include, by default None
        num_samples : int, optional
            number of samples to include, by default None
        transform : types.ModuleType, optional
            transform module, by default None
        sorted : bool, optional
            return the data sorted, by default False
        Raises
        ------
        ValueError
            if anything other than train and test is specified for split
        """
        # path to folder containing the mnist dataset
        MNIST_dir = "."

        if split == "test":
            images_path = os.path.join(MNIST_dir, "t10k-images.idx3-ubyte")
            labels_path = os.path.join(MNIST_dir, "t10k-labels.idx1-ubyte")
        elif split == "train":
            images_path = os.path.join(MNIST_dir, "train-images.idx3-ubyte")
            labels_path = os.path.join(MNIST_dir, "train-labels.idx1-ubyte")
        else:
            raise ValueError("Value entered into 'split' argument must be either 'train' or 'test'")

        X, Y = loadlocal_mnist(
            images_path=images_path,
            labels_path=labels_path,
        )

        # ---------------- add additional filters here -----------------
        # filter out undesired classes
        if classes != None:
            mask = []
            for y in Y:
                if y in classes:
                    mask.append(True)
                else:
                    mask.append(False)
            X = X[mask]
            Y = Y[mask]
        
        # sort classes from smallest to largest
        if sorted==True:
            X = X[Y.argsort()]
            Y.sort()

        # select first n samples
        if num_samples is not None:
            X = X[:num_samples, :]
            Y = Y[:num_samples]
        # ---------------- add additional filters here -----------------
        
        self.x = X
        self.y = Y
        self.num_samples = self.y.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        IMG_SHAPE = (28,28)
        img = np.reshape(self.x[index], IMG_SHAPE)
        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]
        label = torch.tensor(label)

        return img, label

    def __len__(self):
        return self.num_samples


class NotMNISTDataset(Dataset):

    def __init__(
        self, 
        classes: list = None, 
        keep_path: bool = False, 
        transform: types.ModuleType = None, 
        alphabetical_order: bool = False
    ):
        self.keep_path = keep_path
        self.transform = transform

        data_dir = "./notMNIST_small"

        self.sorted_classes = sorted([directory for directory in os.listdir(data_dir)])

        if classes == None:
            self.classes = self.sorted_classes
        else:
            self.classes = classes

        self.img_paths = []
        for path, dir, files in os.walk(data_dir):
            for filename in files:
                if filename != ".DS_Store":
                    classname = path.split("/")[-1]
                    
                    if classname in self.classes:
                        self.img_paths.append(os.path.join(path, filename))
        
        if alphabetical_order:
            self.img_paths.sort()

        self.num_samples = len(self.img_paths)

    def __getitem__(self, index):
        IMG_SHAPE = (28,28)

        img_path = self.img_paths[index]
        img = Image.open(img_path).convert("L")
        img = np.array(img)
        img = np.reshape(img, IMG_SHAPE)
        if self.transform is not None:
            img = self.transform(img)

        label = self.sorted_classes.index(img_path.split("/")[-2])
        label = torch.tensor(label)

        if self.keep_path is True:
            return img, label, img_path
        else:
            return img, label

    def __len__(self):
        return self.num_samples


def generate_plot(figname):
    year = [1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]
    unemployment_rate = [9.8, 12, 8, 7.2, 6.9, 7, 6.5, 6.2, 5.5, 6.3]
    
    plt.figure()
    plt.plot(year, unemployment_rate)
    plt.title('unemployment rate vs year')
    plt.xlabel('year')
    plt.ylabel('unemployment rate')
    plt.savefig(f"./plots/{figname}")
    

def display_image(image):
    img = image.numpy()
    img = img.squeeze()
    plt.imshow(img, interpolation='nearest')
    plt.show()

if __name__ == "__main__":
    if os.path.exists("./plots") is False:
        os.makedirs("./plots")

    generate_plot("unemployment1.png")
    generate_plot("unemployment2.png")
        