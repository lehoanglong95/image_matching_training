from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io
from torch.utils.data import Dataset
from skimage.color import gray2rgb
import requests
from io import BytesIO

class ImageComparingDataset(Dataset):

    def __init__(self, input_file, shopee_image_key, tiki_image_key, shopee_id="id", tiki_id="tiki_ids", transform=None):
        self.df = self.__read_input_file(input_file)
        # self.df = self.df.apply(lambda x: x.explode() if x.name in ["tiki_images", "tiki_ids"] else x)
        self.transform = transform
        self.shopee_image_key = shopee_image_key
        self.tiki_image_key = tiki_image_key
        self.shopee_id = shopee_id
        self.tiki_id = tiki_id

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
        first_image = io.imread(BytesIO(requests.get(self.df.iloc[idx][self.shopee_image_key]).content))
        second_image = io.imread(BytesIO(requests.get(self.df.iloc[idx][self.tiki_image_key].content)))
        shopee_id = self.df.iloc[idx][self.shopee_id]
        tiki_id = self.df.iloc[idx][self.tiki_id]
        if len(first_image.shape) < 3:
            first_image = gray2rgb(first_image)
        if first_image.shape[2] == 4:
            first_image = first_image[:, :, :3]
        if len(second_image.shape) < 3:
            second_image = gray2rgb(second_image)
        if second_image.shape[2] == 4:
            second_image = second_image[:, :, :3]
        if self.transform:
            first_image = self.transform(image=first_image)
            second_image = self.transform(image=second_image)
        return {"shopee_image": first_image, "tiki_image": second_image, "shopee_id": shopee_id, "tiki_id": tiki_id}