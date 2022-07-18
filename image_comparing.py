import numpy as np
from dataset.image_comparing_dataset import ImageComparingDataset
from transform.train_transform import get_val_transform
from torch.utils import data as torch_data
import torch
from modules.efficient_backbone import EfficientBackbone
from utils.util import load_model_state_dict
import torch.nn as nn
import pandas as pd

torch.cuda.empty_cache()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EfficientBackbone("efficientnet-b2", False)
    model = load_model_state_dict(model, "./checkpoints/efficientnet-b2_46.pth")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    cos = nn.CosineSimilarity(dim=1)
    # input_files = [f"input_file/distinct_id_newest.parquet" for i in range(47, 54)]
    # input_file = "input_file/distinct_id_get_newest_multi_categories.parquet"
    # input_file = "input_file/shopee_analytic_shopee_retrival_high_demand_output_shopee_item_prior_image_score_20220704_prior=1_v2_part-00009-d3ef2fcb-3a03-43bd-a6c8-5405d9b9c7b5.c000.snappy.parquet"
    input_file = "input_file/high_demand_product_image_training_data.parquet"
    print(f"PROCESSING {input_file}")
    # try:
    dataset = ImageComparingDataset(input_file,
                                    "main_image_url", "tiki_images", shopee_id="shopeeProductId",
                                    tiki_id="tikiProductId", transform=get_val_transform())
    # except Exception as e:
    #     print(e)
    data_loader = torch_data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=20)
    similarity_df = pd.DataFrame(columns=["lazada_id", "tiki_id", "cos_sim_score"])
    le = len(data_loader)
    file_name = input_file.split("/")[-1]
    try:
        for ii, data in enumerate(data_loader):
            try:
                print(f"{ii} / {le}")
                shopee_image, tiki_image, shopee_ids, tiki_ids = data["shopee_image"], data["tiki_image"], \
                                                               data["shopee_id"], data["tiki_id"]
                error_indices = torch.where(tiki_ids == -1)[0]
                if len(error_indices) > 0:
                    shopee_ids = shopee_ids[error_indices]
                    tiki_ids = tiki_ids[tiki_ids != -1]
                    shopee_image_l = []
                    tiki_image_l = []
                    for idx in range(len(shopee_image)):
                        if idx not in error_indices:
                            shopee_image_l.append(shopee_image[idx])
                            tiki_image_l.append(tiki_image[idx])
                    shopee_image = torch.stack(shopee_image_l)
                    tiki_image = torch.stack(tiki_image_l)
                shopee_image = shopee_image.type(torch.cuda.FloatTensor)
                tiki_image = tiki_image.type(torch.cuda.FloatTensor)
                shopee_feature = model(shopee_image)
                tiki_feature = model(tiki_image)
                cosine_similarity = cos(shopee_feature, tiki_feature).cpu().detach().numpy()
                temp_df = pd.DataFrame()
                temp_df["lazada_id"] = shopee_ids
                temp_df["tiki_id"] = tiki_ids
                temp_df["cos_sim_score"] = cosine_similarity
                similarity_df = pd.concat([similarity_df, temp_df])
                if ii % 1000 == 0:
                    similarity_df.to_parquet(f"./output/similarity_score_{file_name}_from_tail.parquet")
            except Exception as e:
                print(e)
                similarity_df.to_parquet(f"./output/similarity_score_{file_name}_from_tail.parquet")
                continue
    except Exception as e:
        shopee_image, tiki_image, shopee_ids, tiki_ids = data["shopee_image"], data["tiki_image"], \
                                                       data["shopee_id"], data["tiki_id"]
        similarity_df.to_parquet(f"./output/similarity_score_{file_name}_from_tail.parquet")
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