import torch
import numpy as np

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def get_features(model, val_loader):
    di = {}
    for data in val_loader:
        img_url = data["img_url"]
        image = data["image"].type(torch.cuda.FloatTensor)
        features = model(image)
        features = features.view(image.shape[0], -1)
        for i in range(len(range(image.shape[0]))):
            di[img_url[i]] = features[i].cpu().detach().numpy()
    return di

def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(fe_dict, val_dataset):
    sims = []
    labels = []
    for idx, pair in val_dataset.df.iterrows():
        fe_1 = fe_dict[pair[val_dataset.first_image_key]]
        fe_2 = fe_dict[pair[val_dataset.second_image_key]]
        label = int(pair[val_dataset.pair_key])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th

def calculate_acc(model, val_dataset, val_loader):
    features_dict = get_features(model, val_loader)
    acc, th = test_performance(features_dict, val_dataset)
    return acc, th