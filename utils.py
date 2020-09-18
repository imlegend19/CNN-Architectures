import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import torch
import torchvision
from skimage import io
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from local_settings import DATA_DIR

BATCH_SIZE = 32
SIZE = (224, 224)

TRANSFORMS = torchvision.transforms.Compose([
    torchvision.transforms.Resize(SIZE),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(DATA_DIR, self.data.iloc[idx, 0])
        image = io.imread(img_path)
        label = self.data.iloc[idx, 1]
        sample = {
            'image': image,
            'label': label
        }

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


def get_datasets():
    full_data = make_df()
    classes = set(full_data['label'].tolist())
    train, temp = train_test_split(full_data, test_size=0.3)
    val, test = train_test_split(temp, test_size=0.5)

    return classes, load_dataset(train), load_dataset(val), load_dataset(test)
