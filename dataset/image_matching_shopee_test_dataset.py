from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
from torch.utils import data

import torch
from test import *
from transform.rescale import Rescale
from transform.to_tensor import ToTensor
from torchvision.transforms import transforms
from skimage.color import gray2rgb

class ImageMatchingShopeeTestDataset(Dataset):

    def __init__(self, input_file, root_dir, posting_id_key="posting_id", image_key="image", transform=None):
        self.df = self.__read_input_file(input_file)
        self.root_dir = root_dir
        self.transform = transform
        self.posting_id_key = posting_id_key
        self.image_key = image_key

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
        img_url = os.path.join(self.root_dir, self.df.iloc[idx][self.image_key])
        image = io.imread(img_url)
        if len(image.shape) < 3:
            image = gray2rgb(image)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        if self.transform:
            image = self.transform(image=image)
        return {"image": image["image"], "img_url": self.df.iloc[idx][self.image_key],
                "posting_id": self.df.iloc[idx][self.posting_id_key]}

# if __name__ == '__main__':
    # composed = transforms.Compose([Rescale(380), ToTensor()])
    # val_dataset = ImageMatchingValDataset("./input_file/val_set_pair.parquet",
    #                                    "/Users/lap02387/pms/image_matching/images",
    #                                    "first_image_file_name",
    #                                    "second_image_file_name",
    #                                    "pair_result",
    #                                    composed)
    # valloader = data.DataLoader(val_dataset,
    #                             batch_size=8,
    #                             shuffle=False,
    #                             num_workers=4)
    # for ii, data in enumerate(valloader):
    #     img_url = data["img_url"]
    #     image = data["image"]
    #     print(img_url)
    #     print(image.shape)
    #     break