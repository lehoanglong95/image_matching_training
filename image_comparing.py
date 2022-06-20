import numpy as np
from dataset.image_comparing_dataset import ImageComparingDataset
from transform.train_transform import get_val_transform
from torch.utils import data
import torch
from modules.efficient_backbone import EfficientBackbone
from utils.util import load_model_state_dict
import torch.nn as nn
import pandas as pd

torch.cuda.empty_cache()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = ImageComparingDataset("./input_file/image_comparing_v1.parquet",
                                   "main_image_url", "tiki_images", transform=get_val_transform())
    data_loader = data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    model = EfficientBackbone("efficientnet-b4", False)
    model = load_model_state_dict(model, "./checkpoints/efficientnet-b4_28.pth")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    cos = nn.CosineSimilarity(dim=1)
    similarity_df = pd.DataFrame(columns=["shopee_id", "tiki_id", "cos_sim_score"])
    for ii, data in enumerate(data_loader):
        try:
            shopee_image, tiki_image, shopee_ids, tiki_ids = data["shopee_image"], data["tiki_image"], \
                                                           data["shopee_id"], data["tiki_id"]
            shopee_image = shopee_image.type(torch.cuda.FloatTensor)
            tiki_image = tiki_image.type(torch.cuda.FloatTensor)
            shopee_feature = model(shopee_image)
            tiki_feature = model(tiki_image)
            cosine_similarity = cos(shopee_feature, tiki_feature).cpu().detach().numpy()
            for shopee_id, tiki_id, score in zip(shopee_ids, tiki_ids, cosine_similarity):
                similarity_df = similarity_df.append({"shopee_id": shopee_id, "tiki_id": tiki_id, "cos_sim_score": score}, ignore_index=True)
        except Exception as e:
            print(e)
            similarity_df.to_parquet("./output/similarity_score_v1.parquet")
    similarity_df.to_parquet("./output/similarity_score_v1.parquet")
    # embeddings = np.concatenate(embeddings)
    # neigh.fit(embeddings)
    # image_distances, image_indices = neigh.kneighbors(embeddings)
    # y_true = []
    # y_pred = []
    # for idx, results in enumerate(image_indices):
    #     temp_results = set(results) - {idx}
    #     y_true.append(dataset.df.iloc[idx]["label"])
    #     label = int(dataset.df.iloc[idx]["label"])
    #     correct_idx = -1
    #     for result in temp_results:
    #         if dataset.df.iloc[result]["label"] == label:
    #             correct_idx = result
    #             break
    #     if correct_idx == -1:
    #         y_pred.append(dataset.df.iloc[list(temp_results)[0]]["label"])
    #     else:
    #         y_pred.append(dataset.df.iloc[correct_idx]["label"])
    # print(precision_score(y_true, y_pred, average="micro"))
    # print(recall_score(y_true, y_pred, average="micro"))
    # print(f1_score(y_true, y_pred, average="micro"))