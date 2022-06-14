import torch
from torch.utils.data import DataLoader
from dataset.image_matching_dataset import ImageMatchingDataset
from transform.rescale import Rescale
from transform.to_tensor import ToTensor
from torchvision.transforms import transforms
import pandas as pd
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
import numpy as np
from collections import defaultdict

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

class SimpleModel(nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.backbone = EfficientNet.from_pretrained("efficientnet-b2")
        self.last_layer = nn.AdaptiveAvgPool2d(1)

    def forward(self, input):
        output = self.backbone.extract_features(input)
        output = self.last_layer(output)
        return output

if __name__ == '__main__':
    composed = transforms.Compose([Rescale(512), ToTensor()])
    df = pd.read_parquet('./input_file/final_final_product_image.parquet')
    image_matching_dataset = ImageMatchingDataset(df, root_dir="/home/longle/images/images/",
                                                  source_image_key="normalized_url_image",
                                                  des_image_key="normalized_url_image", transform=composed)
    dataloader = DataLoader(image_matching_dataset, batch_size=2, shuffle=False, num_workers=2)
    model = SimpleModel()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    errors_data = []
    embeddings = []
    for i, sample_batched in enumerate(dataloader):
        print(i)
        input_image = sample_batched["image"]
        print(input_image.shape)
        try:
            if sample_batched["image"] is None:
                print("image is None")
                embeddings.append(False)
                continue
            # image = torch.unsqueeze(input_image[0], dim=0)
            source_embeds = model(input_image.type(torch.cuda.FloatTensor).to(device))
            # source_embeds = model.extract_features(input_image.type(torch.cuda.FloatTensor))
            # source_embeds = last_layer(source_embeds)
            # print(source_embeds.shape)
            # source_embeds = source_embeds.view(2, -1)
            for embed in source_embeds:
                embeddings.append(embed)
        except Exception as e:
            print(f"PROCESS {i}: {e}")
            errors_data.append(i)
            embeddings.append(False)
            continue
    neigh = NearestNeighbors(n_neighbors=50)
    neigh.fit(embeddings)
    image_distances, image_indices = neigh.kneighbors(embeddings)
    cols, rows = np.where(image_distances == 0)
    di = defaultdict(list)
    for c, r in zip(cols, rows):
        if image_indices[c][r] != c:
            di[c].append(image_indices[c][r])
    temp = [[]] * len(df)
    df["duplicated_image"] = temp
    for k, v in di.items():
        df.at[k, "duplicated_image"] = v
    df.to_parquet("./input_file/final_product_image_with_duplicated_image.parquet")