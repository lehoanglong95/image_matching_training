import torch

def load_model_state_dict(model, pretrain_path):
    pretrain_dict = torch.load(pretrain_path)
    processed_dict = {}
    for k, k1 in zip(pretrain_dict.keys(), model.state_dict().keys()):
        processed_dict[k1] = pretrain_dict[k]
    model.load_state_dict(processed_dict)
    return model

def normalize_url(url):
    TALA_URL = "https://salt.tikicdn.com/"
    PRODUCT_PREFIX = "ts/"
    ID_URL = "media/catalog/product/"
    try:
        if url.startswith('http'):
            new_url = url.replace('http://tala', 'https://salt')
            return new_url
        if url.startswith('product') or url.startswith('tmp') or url.startswith('/product') or url.startswith('/tmp'):
            if not url.startswith('/'):
                return ''.join((
                    TALA_URL,
                    PRODUCT_PREFIX,
                    url
                ))
            else:
                return ''.join((
                    TALA_URL,
                    PRODUCT_PREFIX,
                    url[1:]
                ))
        else:
            if url[0] == "/":
                return ''.join((
                    TALA_URL,
                    ID_URL,
                    url[1:]
                ))
            else:
                return ''.join((
                    TALA_URL,
                    ID_URL,
                    url
                ))
    except Exception as e:
        print(url)
        return ""