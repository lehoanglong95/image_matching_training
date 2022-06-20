import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
from dataset.image_matching_dataset import ImageMatchingDataset
from transform.train_transform import get_val_transform
from torch.utils import data
import torch

if __name__ == '__main__':
    neigh = NearestNeighbors(n_neighbors=2, metric='cosine')
    dataset = ImageMatchingDataset("./input_file/final_product_image_with_label_test_set.parquet",
                                   "/data/long.le3/image_matching/images",
                                   "normalized_url_image", "label", transform=get_val_transform())
    data_loader = data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    model = torch.load("./checkpoints/efficientnet-b4_0.pth")
    embeddings = []
    for ii, data in enumerate(data_loader):
        data_input, label = data["image"], data["label"]
        data_input = data_input.type(torch.cuda.FloatTensor)
        label = label.type(torch.cuda.LongTensor)
        feature = model(data_input)
        embeddings.append(feature)
    embeddings = np.concatenate(embeddings)
    neigh.fit(embeddings)
    image_distances, image_indices = neigh.kneighbors(embeddings)
    acc = 0
    for idx, results in enumerate(image_indices):
        for result in results:
            if result != idx:
                if dataset.df.iloc[idx]["label"] == dataset.df.iloc[result]["label"]:
                    acc += 1
    print(acc / len(dataset))


