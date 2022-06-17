from modules.efficient_backbone import EfficientBackbone
import torch
import yaml
import os
from transform.train_transform import get_val_transform
from dataset.image_matching_val_dataset import ImageMatchingValDataset
from torch.utils import data
from validate import calculate_acc
from utils.util import load_model_state_dict

CONFIG_PATH = "./configs"
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

if __name__ == '__main__':
    config = load_config("config.yaml")
    eval_config = config["evaluate"]
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
    model = EfficientBackbone("efficientnet-b4", False)
    model = load_model_state_dict(model, "./checkpoints/efficientnet-b4_6.pth")
    model.eval()
    acc, th = calculate_acc(model, val_dataset, valloader)
    print(acc)