import torch
from train_model import Net, LeNet5_transform
from torch.utils.data import ConcatDataset, DataLoader
from dataset import NotMNISTDataset, MNISTDataset, display_image
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
            # are labeled 0 for the experiment
            if len(sample) == 3:
                return image, label, 0
            else:
                return image, label, 1
        else:
            return image, label


def extract(model, dataloader: DataLoader, layers: List[str], save_path: str):

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

    if os.path.exists("./activation-maps") is False:
        os.makedirs("./activation-maps")

    for layer, activation_maps in extractor_outputs.items():
        maps = np.asarray(activation_maps)
        np.save(f"./activation-maps/{layer}-maps.npz", maps)

    for handle in handles:
        handle.remove()
    

if __name__ == "__main__":
    model = Net()
    model.load_state_dict(torch.load("./saved_models/10-classes-MNIST.pth"))

    mnist = MNISTDataset(split="test", transform=LeNet5_transform) 
    notmnist = NotMNISTDataset(keep_path=True, transform=LeNet5_transform)
    # notmnist_dataloader = DataLoader(dataset=notmnist, batch_size=2)
    print(f"number of samples in MNIST: {len(mnist)}")
    print(f"number of samples in MNIST: {len(notmnist)}")

    combined_ds = CombinedDataset([mnist, notmnist], in_dist_labels=True)
    combined_dataloader = DataLoader(dataset=combined_ds, batch_size=2)

    extract(model=model, dataloader=combined_dataloader, layers=["conv2", "fc1"], save_path=".")
    