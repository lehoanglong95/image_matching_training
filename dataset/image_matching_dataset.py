from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io
from torch.utils.data import Dataset
from skimage.color import gray2rgb

class ImageMatchingDataset(Dataset):

    def __init__(self, input_file, root_dir, image_key, label_key, transform=None):
        self.df = self.__read_input_file(input_file)
        self.df = self.df.reset_index()
        del self.df["index"]
        # l = [4308, 92, 4177, 11083, 828]
        # train_label_di = {l[idx]: idx for idx in range(len(l))}
        # def get_train_label(val):
        #     return train_label_di[val]
        # self.df = self.df[self.df["train_label"].isin(set(l))]
        # self.df["train_label_new"] = self.df["train_label"].apply(get_train_label)
        # print(len(self.df))
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
        image = io.imread(source_img_name)
        if len(image.shape) < 3:
            image = gray2rgb(image)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        if self.transform:
            image = self.transform(image=image)
        return {"image": image["image"], "label": self.df.iloc[idx][self.label_key],
                "url_image": self.df.iloc[idx][self.image_key]}