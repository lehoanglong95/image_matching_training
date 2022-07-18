import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
# from dataset.image_matching_dataset import ImageMatchingDataset
from dataset.image_matching_shopee_test_dataset import ImageMatchingShopeeTestDataset
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
    number_of_neighbors = 50
    neigh = NearestNeighbors(n_neighbors=number_of_neighbors, metric='cosine')
    dataset = ImageMatchingShopeeTestDataset("./input_file/x_test.csv",
                                   "/home/longle/shopee-product-matching/train_images", transform=get_val_transform())
    data_loader = data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    model = EfficientBackbone("efficientnet-b2", False)
    model = load_model_state_dict(model, "./checkpoints/efficientnet-b2_46.pth")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    embeddings = []
    url_l = []
    posting_ids_l = []
    for ii, data in enumerate(data_loader):
        data_input, image_url, posting_id = data["image"], data["img_url"], data["posting_id"]
        data_input = data_input.type(torch.cuda.FloatTensor)
        feature = model(data_input)
        url_l.append(image_url)
        posting_ids_l.append(posting_id)
        embeddings.append(feature.cpu().detach().numpy())
    embeddings = np.concatenate(embeddings)
    urls = np.concatenate(url_l)
    posting_ids = np.concatenate(posting_ids_l)
    neigh.fit(embeddings)
    image_distances, image_indices = neigh.kneighbors(embeddings)
    y_true = []
    y_pred = []
    similarity_df = pd.DataFrame(columns=["posting_id", "image", "predict"])
    for idx, (indices, distance) in enumerate(zip(image_indices, image_distances)):
        try:
            temp_df = pd.DataFrame()
            threshold = 0.37
            ids = np.where(distance < threshold)[0]
            while len(ids) < 2:
                threshold += 0.05
                ids = np.where(distance < threshold)[0]
            temp_df["posting_id"] = [posting_ids[idx]]
            temp_l = []
            for e in posting_ids[indices[ids]]:
                temp_l.append(str(e))
            temp_df["predict"] = [list(temp_l)]
            temp_df["image"] = [urls[idx]]
            # self_image = di[idx]
            # temp_df["normalized_url_image"] = [self_image]
            # duplicated = [di[ee] for ee in results]
            # for i in range(len(duplicated)):
            #     temp_df[f"duplicated_{i}"] = [duplicated[i]]
            # temp_results = set(results) - {idx}
            # y_true.append(labels[idx])
            # label = int(labels[idx])
            # abc = dataset.df[dataset.df["label"] == label]["normalized_url_image"].values
            # deg = [e.split("/")[-1] for e in abc]
            # correct_image = set(deg) - set([self_image])
            # correct_idx = -1
            # for result in temp_results:Ï
            #     if dataset.df.iloc[result]["label"] == label:
            #         correct_idx = result
            #         break
            # if correct_idx == -1:
            #     y_pred.append(dataset.df.iloc[list(temp_results)[0]]["label"])
            # else:
            #     y_pred.append(dataset.df.iloc[correct_idx]["label"])
            # temp_df["correct_image"] = list(correct_image)
            similarity_df = pd.concat([similarity_df, temp_df])
        except Exception as e:
            print(e)
            posting_id = posting_ids[idx]
            print(posting_id)
            break
    # y_true = np.array(y_true)
    # y_pred = np.array(y_pred)
    # error_indices = np.where(y_pred != y_true)
    # np.save(f"output/error_indices_{number_of_neighbors}.npy", error_indices)
    similarity_df.to_parquet(f"output/shopee_similarity_b2_46.parquet", index=False)
    # print(precision_score(y_true, y_pred, average="micro"))
    # print(recall_score(y_true, y_pred, average="micro"))
    # print(f1_score(y_true, y_pred, average="micro"))

