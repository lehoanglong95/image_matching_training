from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ImageMatchingDataset(Dataset):

    def __init__(self, input_file, root_dir, image_key, label_key, transform=None):
        self.df = self.__read_input_file(input_file)
        self.root_dir = root_dir
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

    def __read_input_file(self, input_file):
        extension_input_file = os.path.splitext(input_file)[-1][1:]
        if "csv" in extension_input_file:
            return pd.read_csv(input_file)
        elif "parquet" in extension_input_file:
            return pd.read_parquet(input_file)
        elif "json" in extension_input_file:
            return pd.read_json(input_file)
        return pd.DataFrame()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        source_img_name = os.path.join(self.root_dir, self.df.iloc[idx][self.image_key].split("/")[-1])
        source_image = io.imread(source_img_name)
        sample = {"image": source_image}
        if self.transform:
            sample = self.transform(sample)
        return {"image": sample["image"], "label": self.df.iloc[idx][self.label_key]}