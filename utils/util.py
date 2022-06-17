import torch

def load_model_state_dict(model, pretrain_path):
    pretrain_dict = torch.load(pretrain_path)
    processed_dict = {}
    for k, k1 in zip(pretrain_dict.keys(), model.state_dict().keys()):
        processed_dict[k1] = pretrain_dict[k]
    model.load_state_dict(processed_dict)
    return model