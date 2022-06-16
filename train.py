from __future__ import print_function
import os
from dataset.image_matching_dataset import ImageMatchingDataset
from dataset.image_matching_val_dataset import ImageMatchingValDataset
import torch
from torch.utils import data
from modules.efficient_backbone import EfficientBackbone
from modules.metrics import *
import torchvision
import torch
import numpy as np
import random
import time
import yaml
from torch.nn import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from test import *
from transform.rescale import Rescale
from transform.to_tensor import ToTensor
from torchvision.transforms import transforms
from utils.visualizer import Visualizer
from validate import calculate_acc
from modules.focal_loss import FocalLoss

CONFIG_PATH = "./configs"
def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

if __name__ == '__main__':
    config = load_config("config.yaml")
    device = torch.device("cuda")
    if config["display"]:
        visualizer = Visualizer()
    train_config = config["train"]
    eval_config = config["evaluate"]
    dataset_config = train_config["dataset"]
    composed = transforms.Compose([Rescale(train_config["data_augmentation"]["output_shape"]), ToTensor()])
    train_dataset = ImageMatchingDataset(dataset_config["input_file"],
                                         dataset_config["root_dir"],
                                         dataset_config["image_key"],
                                         dataset_config["label_key"],
                                         composed)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=train_config["batch_size"],
                                  shuffle=train_config["shuffle"],
                                  num_workers=train_config["num_workers"])
    val_dataset = ImageMatchingValDataset(eval_config["input_file"],
                                         eval_config["root_dir"],
                                         eval_config["first_image_key"],
                                         eval_config["second_image_key"],
                                         eval_config["pair_key"],
                                         composed)
    valloader = data.DataLoader(val_dataset,
                                  batch_size=eval_config["batch_size"],
                                  shuffle=eval_config["shuffle"],
                                  num_workers=eval_config["num_workers"])
    if config["loss"] == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model_config = config["model"]
    model = EfficientBackbone(model_config["backbone_name"], model_config["pretrain"])

    if model_config["metric"] == 'add_margin':
        metric_fc = AddMarginProduct(model_config["backbone_output"], model_config["num_classes"], s=30, m=0.35)
    elif model_config["metric"] == 'arc_margin':
        metric_fc = ArcMarginProduct(model_config["backbone_output"], model_config["num_classes"], s=30, m=0.5, easy_margin=False)
    elif model_config["metric"] == 'sphere':
        metric_fc = SphereProduct(model_config["backbone_output"], model_config["num_classes"], m=4)
    else:
        metric_fc = nn.Linear(model_config["backbone_output"], model_config["num_classes"])

    print(model)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    if config["optimizer"]["name"] == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=config["optimizer"]["lr"], weight_decay=config["optimizer"]["weight_decay"])
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=config["optimizer"]["lr"], weight_decay=config["optimizer"]["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode=config["scheduler"]["mode"],
                                    factor=config["scheduler"]["factor"],
                                    patience=config["scheduler"]["patience"])

    start = time.time()
    best_acc = 0
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    for i in range(train_config["epochs"]):
        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data["image"], data["label"]
            data_input = data_input.type(torch.cuda.FloatTensor).to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            if iters % train_config["print_freq"] == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                # print(output)
                # print(label)
                acc = np.mean((output == label).astype(int))
                speed = train_config["print_freq"] / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(), acc))
                if config["display"]:
                    visualizer.display_current_results(iters, loss.item(), name='train_loss')
                    visualizer.display_current_results(iters, acc, name='train_acc')

                start = time.time()

        if i % train_config["save_interval"] == 0 or i == train_config["epochs"]:
            model.eval()
            acc, th = calculate_acc(model, val_dataset, valloader)
            scheduler.step(acc)
            print(f"epoch {i} with acc {acc} and th {th}")
            if acc >= best_acc:
                best_acc = acc
                save_model(model, config["checkpoints_path"], model_config["backbone_name"], i)
            if config["display"]:
                visualizer.display_current_results(iters, acc, name='test_acc')