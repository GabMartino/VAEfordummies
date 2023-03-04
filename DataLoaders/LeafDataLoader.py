import os

import pytorch_lightning as pl

import pandas as pd
import torch.utils.data
import torchvision.io
import PIL.Image as Image

from torch.utils.data import DataLoader
class LeafDataset(torch.utils.data.Dataset):

    def __init__(self, csv_path, images_path, transform= None):
        super().__init__()

        csv = pd.read_csv(csv_path)

        self.transform = transform

        self.image_path = images_path
        self.labels = []
        self.image_ids = []
        for index, row in csv.iterrows():
            v = str(row['Values']).split("-")
            entries = [ str(f) for f in range(int(v[0]), int(v[1]) + 1)]

            self.labels += [row['Label']]*len(entries)
            self.image_ids += entries



    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_path + "/" + self.image_ids[idx] + ".jpg"
        image = Image.open(img_name)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        return image, label
class LeafDataLoader(pl.LightningDataModule):

    def __init__(self, csv_path, images_path, batch_size=16,  validation_split=0.2, transform=None):
        super().__init__()
        self.batch_size = batch_size

        dataset = LeafDataset(csv_path, images_path, transform)
        train_len = int((1 - validation_split)*len(dataset))
        val_len = len(dataset) - train_len
        self.train_set, self.val_set = torch.utils.data.random_split(dataset, [train_len, val_len])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, prefetch_factor=1, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, prefetch_factor=1, num_workers=os.cpu_count())

    def predict_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, prefetch_factor=1, num_workers=os.cpu_count())




def getDataset():

    path = "../../Datasets/Flavia/dataset"
    from bs4 import BeautifulSoup

    with open(path, 'r') as f:
        data = f.read()

    Bs_data = BeautifulSoup(data, "xml")

    b_unique = Bs_data.find_all('tr')
    dataset = []
    for entry in b_unique:
        v = entry.find_all('td')
        entry = {}
        if len(v) > 0:
            entry["Label"] = str(v[2].getText()).replace('"', '')
            entry["Values"] = v[3].getText()
            dataset.append(entry)
    print(dataset)
    import pandas as pd
    df = pd.DataFrame(dataset)
    df.to_csv("dataset.csv", index=False, header=True)
    '''
    
    b_name = Bs_data.find('child', {'name': 'Frank'})

    print(b_name)

    value = b_name.get('test')

    print(value)
    
    '''


def main():
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize(size=(256, 256)),
        T.ToTensor()])
    dataloader = LeafDataLoader("../../Datasets/Flavia/dataset.csv", "../../Datasets/Flavia/Leaves", transform=transform)
    train_data = dataloader.train_dataloader()
    image_sample = train_data.dataset.__getitem__(0)[0]
    print(torch.mean(torch.mean(image_sample, dim=0)))
    print(train_data.dataset.__getitem__(0)[0].shape)
if __name__ == "__main__":
    main()