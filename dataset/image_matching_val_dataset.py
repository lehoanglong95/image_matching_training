from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
from torch.utils import data

import torch
from test import *
from transform.rescale import Rescale
from transform.to_tensor import ToTensor
from torchvision.transforms import transforms
from skimage.color import gray2rgb

class ImageMatchingValDataset(Dataset):

    def __init__(self, input_file, root_dir, first_image_key, second_image_key, pair_key, transform=None):
        self.df = self.__read_input_file(input_file)
        self.root_dir = root_dir
        self.transform = transform
        self.first_image_key = first_image_key
        self.second_image_key = second_image_key
        self.pair_key = pair_key
        details = {"image_url": list(set(self.df[first_image_key].values).union(set(self.df[second_image_key].values)))}
        self.image_url_df = pd.DataFrame(details, columns=["image_url"])

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
        return len(self.image_url_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_url = os.path.join(self.root_dir, self.image_url_df.iloc[idx]["image_url"])
        image = io.imread(img_url)
        if len(image.shape) < 3:
            image = gray2rgb(image)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        if self.transform:
            image = self.transform(image=image)
        return {"image": image["image"], "img_url": self.image_url_df.iloc[idx]["image_url"]}

if __name__ == '__main__':
    composed = transforms.Compose([Rescale(380), ToTensor()])
    val_dataset = ImageMatchingValDataset("./input_file/val_set_pair.parquet",
                                       "/Users/lap02387/pms/image_matching/images",
                                       "first_image_file_name",
                                       "second_image_file_name",
                                       "pair_result",
                                       composed)
    valloader = data.DataLoader(val_dataset,
                                batch_size=8,
                                shuffle=False,
                                num_workers=4)
    for ii, data in enumerate(valloader):
        img_url = data["img_url"]
        image = data["image"]
        print(img_url)
        print(image.shape)
        break