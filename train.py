from __future__ import print_function
from dataset.image_matching_dataset import ImageMatchingDataset
from dataset.image_matching_val_dataset import ImageMatchingValDataset
from torch.utils import data
from modules.efficient_backbone import EfficientBackbone
from modules.metrics import *
import torch
import numpy as np
import time
import yaml
from torch.nn import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from test import *
from utils.visualizer import Visualizer
from validate import calculate_acc
from modules.focal_loss import FocalLoss
from transform.train_transform import get_train_transform, get_val_transform
from utils.util import load_model_state_dict

CONFIG_PATH = "./configs"
def save_model(model, save_path, name, iter_cnt, pretrain_cnt=0):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

if __name__ == '__main__':
    config = load_config("config.yaml")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if config["display"]:
        visualizer = Visualizer()
    train_config = config["train"]
    eval_config = config["evaluate"]
    dataset_config = train_config["dataset"]
    train_dataset = ImageMatchingDataset(dataset_config["input_file"],
                                         dataset_config["root_dir"],
                                         dataset_config["image_key"],
                                         dataset_config["label_key"],
                                         dataset_config["data_source_key"],
                                         get_train_transform())
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=train_config["batch_size"],
                                  shuffle=train_config["shuffle"],
                                  num_workers=train_config["num_workers"])
    eval_dataset_config = eval_config["dataset"]
    val_dataset = ImageMatchingValDataset(eval_dataset_config["input_file"],
                                         eval_dataset_config["root_dir"],
                                         eval_dataset_config["first_image_key"],
                                         eval_dataset_config["second_image_key"],
                                         eval_dataset_config["pair_key"],
                                         get_val_transform())
    valloader = data.DataLoader(val_dataset,
                                  batch_size=eval_config["batch_size"],
                                  shuffle=eval_config["shuffle"],
                                  num_workers=eval_config["num_workers"])
    if config["loss"] == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model_config = config["model"]
    model = EfficientBackbone(model_config["backbone_name"], model_config["backbone_output"], model_config["pretrain"])
    for idx, children_model in enumerate(model.children()):
        if idx == 0:
            for param in children_model.parameters():
                param.requires_grad = False
    if model_config["metric"] == 'add_margin':
        metric_fc = AddMarginProduct(model_config["backbone_output"], model_config["num_classes"], s=30, m=0.35)
    elif model_config["metric"] == 'arc_margin':
        metric_fc = ArcMarginProduct(model_config["backbone_output"], model_config["num_classes"], s=30, m=0.5, easy_margin=False)
    elif model_config["metric"] == 'sphere':
        metric_fc = SphereProduct(model_config["backbone_output"], model_config["num_classes"], m=4)
    else:
        metric_fc = nn.Linear(model_config["backbone_output"], model_config["num_classes"])

    # print(model)

    if config["optimizer"]["name"] == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=config["optimizer"]["lr"], weight_decay=config["optimizer"]["weight_decay"])
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=config["optimizer"]["lr"], weight_decay=float(config["optimizer"]["weight_decay"]))
    scheduler = ReduceLROnPlateau(optimizer, mode=config["scheduler"]["mode"],
                                    factor=config["scheduler"]["factor"],
                                    patience=config["scheduler"]["patience"])
    start = time.time()
    best_acc = 0
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        metric_fc = DataParallel(metric_fc)
    model.to(device)
    metric_fc.to(device)
    for i in range(train_config["epochs"]):
        if i == 5:
            for idx, children_model in enumerate(model.children()):
                if idx == 0:
                    for param in children_model.parameters():
                        param.requires_grad = True
            optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                         lr=config["optimizer"]["lr"] / 10,
                                         weight_decay=float(config["optimizer"]["weight_decay"]))
            scheduler = ReduceLROnPlateau(optimizer, mode=config["scheduler"]["mode"],
                                          factor=config["scheduler"]["factor"],
                                          patience=config["scheduler"]["patience"])
        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data["image"], data["label"]
            data_input = data_input.type(torch.cuda.FloatTensor)
            label = label.type(torch.cuda.LongTensor)
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
            if i > 4:
                scheduler.step(acc)
            print(f"epoch {i} with acc {acc} and th {th}")
            if acc >= best_acc:
                best_acc = acc
                save_model(model, config["checkpoints_path"], model_config["backbone_name"], i)
                save_model(metric_fc, config["checkpoints_path"], "metric_fc", i)
            if config["display"]:
                visualizer.display_current_results(iters, acc, name='test_acc')