import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from local_settings import DATA_DIR
import torchvision
import multiprocessing as mp
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from skimage import io

BATCH_SIZE = 32
SIZE = (224, 224)

TRANSFORMS = torchvision.transforms.Compose([
    torchvision.transforms.Resize(SIZE),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def make_df():
    folders = os.listdir(DATA_DIR)
    data = []

    for folder in tqdm(folders, total=len(folders)):
        img_paths = os.listdir(os.path.join(DATA_DIR, folder))
        for path in img_paths:
            data.append([os.path.join(folder, path), folder])

    return pd.DataFrame(np.asarray(data), columns=['image', 'label'])


class C101Dataset(Dataset):
    def __init__(self, data, transform=TRANSFORMS):
        self.data = self.data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(DATA_DIR, self.data.iloc[idx, 0])
        image = io.imread(img_path)
        label = self.data.iloc[idx, 1]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


def load_dataset(df):
    trans_data = C101Dataset(df)
    loader = DataLoader(
        trans_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=mp.cpu_count()
    )

    return loader


full_data = make_df()
train, test = train_test_split(full_data, test_size=0.2)
train, test = load_dataset(train), load_dataset(test)
