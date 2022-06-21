import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
from dataset.image_matching_dataset import ImageMatchingDataset
from transform.train_transform import get_val_transform
from torch.utils import data
import torch
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from modules.efficient_backbone import EfficientBackbone
from utils.util import load_model_state_dict
import torch.nn as nn

torch.cuda.empty_cache()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    number_of_neighbors = 3
    neigh = NearestNeighbors(n_neighbors=(number_of_neighbors + 1), metric='cosine')
    dataset = ImageMatchingDataset("./input_file/final_product_image_with_label_test_set.parquet",
                                   "/home/longle/images/images",
                                   "normalized_url_image", "label", transform=get_val_transform())
    data_loader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    model = EfficientBackbone("efficientnet-b4", False)
    model = load_model_state_dict(model, "./checkpoints/efficientnet-b4_28.pth")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    embeddings = []
    for ii, data in enumerate(data_loader):
        data_input, label, url_image = data["image"], data["label"], data["url_image"]
        data_input = data_input.type(torch.cuda.FloatTensor)
        label = label.type(torch.cuda.LongTensor)
        feature = model(data_input)
        embeddings.append(feature.cpu().detach().numpy())
    embeddings = np.concatenate(embeddings)
    neigh.fit(embeddings)
    image_distances, image_indices = neigh.kneighbors(embeddings)
    y_true = []
    y_pred = []
    similarity_df = pd.DataFrame(columns=["normalized_url_image", "duplicated", "correct_image"])
    for idx, results in enumerate(image_indices):
        try:
            temp_df = pd.DataFrame()
            self_image = dataset.df.iloc[idx]["normalized_url_image"]
            temp_df["normalized_url_image"] = self_image
            temp_df["duplicated"] = dataset.df.loc[results]["normalized_url_image"].values
            temp_results = set(results) - {idx}
            y_true.append(dataset.df.iloc[idx]["label"])
            label = int(dataset.df.iloc[idx]["label"])
            abc = dataset.df[dataset.df["label"] == label]["normalized_url_image"].values
            correct_image = set(abc) - set([self_image])
            correct_idx = -1
            for result in temp_results:
                if dataset.df.iloc[result]["label"] == label:
                    correct_idx = result
                    break
            if correct_idx == -1:
                y_pred.append(dataset.df.iloc[list(temp_results)[0]]["label"])
            else:
                y_pred.append(dataset.df.iloc[correct_idx]["label"])
            temp_df["correct_image"] = list(correct_image)[0]
            similarity_df = pd.concat([similarity_df, temp_df])
        except:
            label = int(dataset.df.iloc[idx]["label"])
            print(label)
            break
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    error_indices = np.where(y_pred != y_true)
    np.save(f"output/error_indices_{number_of_neighbors}.npy", error_indices)
    similarity_df.to_parquet(f"output/similarity_df_{number_of_neighbors}.parquet", index=False)
    print(precision_score(y_true, y_pred, average="micro"))
    print(recall_score(y_true, y_pred, average="micro"))
    print(f1_score(y_true, y_pred, average="micro"))

