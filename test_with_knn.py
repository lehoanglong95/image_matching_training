import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
from dataset.image_matching_dataset import ImageMatchingDataset
from transform.train_transform import get_val_transform
from torch.utils import data
import torch
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

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
    y_true = []
    y_pred = []
    for idx, results in enumerate(image_indices):
        temp_results = set(results) - {idx}
        y_true.append(dataset.df.iloc[idx]["label"])
        if idx in temp_results:
            y_pred.append(dataset.df.iloc[idx]["label"])
        else:
            y_pred.append(dataset.df.iloc[list(temp_results)[0]]["label"])
    print(precision_score(y_true, y_pred, average="macro"))
    print(recall_score(y_true, y_pred, average="macro"))
    print(f1_score(y_true, y_pred, average="macro"))

