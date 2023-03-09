import os

import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataLoader(pl.LightningDataModule):


    def __init__(self, batch_size, path):
        super().__init__()
        self.batch_size = batch_size

        self.train_ds = MNIST(path, train=True, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
        self.val_ds = MNIST(path, train=False, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,  num_workers=os.cpu_count(), pin_memory=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True,
                          prefetch_factor=2)
    def predict_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, shuffle=True, pin_memory=True,
                          prefetch_factor=2)




def main():
    datamodule = MNISTDataLoader(32, "../../Datasets/mnist")

    train_dataloader = datamodule.train_dataloader()

    examples = enumerate(train_dataloader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape, example_targets.shape)
if __name__ == "__main__":
    main()