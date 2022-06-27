from efficientnet_pytorch import EfficientNet
import torch.nn as nn

class EfficientBackbone(nn.Module):

    def __init__(self, backbone_name="efficientnet-b4", pretrain=False):
        super(EfficientBackbone, self).__init__()
        if pretrain:
            self.backbone = EfficientNet.from_pretrained(backbone_name)
        else:
            self.backbone = EfficientNet.from_name(backbone_name)
        self.batchnorm = nn.BatchNorm2d(1792)
        self.last_layer = nn.AdaptiveAvgPool2d(1)

    def forward(self, input):
        output = self.backbone.extract_features(input)
        output = self.batchnorm(output)
        output = self.last_layer(output)
        output = output.view(input.shape[0], -1)
        return output

if __name__ == '__main__':
    model = EfficientBackbone(pretrain=False)
    a = []
    for idx, e in enumerate(model.children()):
        print(idx)
        a.append(e)
    print(len(a))
