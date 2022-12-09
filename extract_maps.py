import torch
from train_model import Net
from torch.utils.data import ConcatDataset, DataLoader
from dataset import NotMNISTDataset, MNISTDataset, save_image


class CombinedDataset(ConcatDataset):
    def __init__(self, datasets, in_dist_labels: bool = False):
        super().__init__(datasets)
        self.in_dist_labels = in_dist_labels

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        image = sample[0]
        label = sample[1].item()

        mnist_labels = [0,1,2,3,4,5,6,7,8,9]

        if self.in_dist_labels is True:
            if label in mnist_labels:
                return image, label, 0
            else:
                return image, label, 1
        else:
            return image, label


if __name__ == "__main__":
    model = Net()
    model.load_state_dict(torch.load("./saved_models/012-classes-MNIST.pth"))

    mnist = MNISTDataset(split="test") 
    notmnist = NotMNISTDataset(keep_path=True)
    # combined_ds = ConcatDataset([mnist, notmnist])

    combined_ds = CombinedDataset([mnist, notmnist], in_dist_labels=True)
    combined_dataloader = DataLoader(dataset=combined_ds, batch_size=1)

    for i, sample in enumerate(combined_dataloader):
        # %%
        if i == 10050:
            image = sample[0]
            label = sample[1]
            out_label = sample[2]
            save_image(image)
            break