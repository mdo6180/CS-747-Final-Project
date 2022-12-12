import torch
from train_model import Net, LeNet5_transform
from torch.utils.data import ConcatDataset, DataLoader
from dataset import NotMNISTDataset, MNISTDataset 
import numpy as np
from typing import List
import os


class CombinedDataset(ConcatDataset):
    def __init__(self, datasets, in_dist_labels: bool = False):
        super().__init__(datasets)
        self.in_dist_labels = in_dist_labels

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        image = sample[0]
        label = sample[1].item()

        if self.in_dist_labels is True:
            
            # out of sample images (i.e., images from notMNIST dataset) 
            # are labeled -1 for the experiment
            if len(sample) == 3:
                return image, label, -1
            else:
                return image, label, 1
        else:
            return image, label


def extract(
    model, 
    dataloader: DataLoader, 
    layers: List[str], 
    folder_name: str, 
    flatten: bool = False
):

    model.eval()
    extractor_outputs = {key:[] for key in layers}

    def outer_hook(layer_name: str):
        def hook(feature_extractor, input, output):
            # Hook function to get activation map
            # .cpu() moves tensor from GPU to CPU,
            # .detach() removes computational graph tracking,
            # .numpy() converts tensor to numpy array
            for activation_map in output:
                extractor_outputs[layer_name].append(activation_map.cpu().detach().numpy())
        return hook

    handles = []
    for layer_name, module in model.named_modules():
        if layer_name in layers:
            handle = module.register_forward_hook(outer_hook(layer_name))
            handles.append(handle)
    
    for batch_id, batch in enumerate(dataloader):
        images = batch[0]
        model(images)

    if os.path.exists(f"./{folder_name}") is False:
        os.makedirs(f"./{folder_name}")

    def prod(tup):
        product = 1
        for element in tup:
            product *= element
        return product
    
    for layer, activation_maps in extractor_outputs.items():
        maps = np.asarray(activation_maps)

        if flatten is True:
            num_samples = maps.shape[0]
            num_columns = prod(maps.shape[1:])
            maps = np.reshape(maps, (num_samples, num_columns))
            np.save(f"./{folder_name}/{layer}-flattened-maps", maps)
        else: 
            np.save(f"./{folder_name}/{layer}-maps", maps)

    for handle in handles:
        handle.remove()
    

if __name__ == "__main__":
    model = Net()
    model.load_state_dict(torch.load("./saved_models/10-classes-MNIST.pth"))

    mnist_train_ds = MNISTDataset(split="train", transform=LeNet5_transform)
    mnist_train_dataloader = DataLoader(dataset=mnist_train_ds, batch_size=100)

    mnist_test_ds = MNISTDataset(split="test", transform=LeNet5_transform) 
    mnist_test_dataloader = DataLoader(dataset=mnist_test_ds, batch_size=100)

    notmnist_ds = NotMNISTDataset(keep_path=True, transform=LeNet5_transform)
    notmnist_dataloader = DataLoader(dataset=notmnist_ds, batch_size=100)

    print(f"number of samples in MNIST train: {len(mnist_train_ds)}")
    print(f"number of samples in MNIST test: {len(mnist_test_ds)}")
    print(f"number of samples in notMNIST: {len(notmnist_ds)}")

    combined_ds = CombinedDataset([mnist_test_ds, notmnist_ds], in_dist_labels=True)
    combined_dataloader = DataLoader(dataset=combined_ds, batch_size=100)

    extraction_layers = ["conv2", "fc1"]
    extract(model=model, dataloader=mnist_train_dataloader, layers=extraction_layers, folder_name="MNIST-train", flatten=True)
    extract(model=model, dataloader=mnist_test_dataloader, layers=extraction_layers, folder_name="MNIST-test", flatten=True)
    extract(model=model, dataloader=notmnist_dataloader, layers=extraction_layers, folder_name="notMNIST", flatten=True)
    extract(model=model, dataloader=combined_dataloader, layers=extraction_layers, folder_name="MNIST-notMNIST-combined", flatten=True)
    