from efficientnet_pytorch import EfficientNet
import torch.nn as nn

class EfficientBackbone(nn.Module):

    def __init__(self, backbone_name="efficientnet-b4", pretrain=False):
        super(EfficientBackbone, self).__init__()
        if pretrain:
            self.backbone = EfficientNet.from_pretrained(backbone_name)
        else:
            self.backbone = EfficientNet.from_name(backbone_name)
        self.last_layer = nn.AdaptiveAvgPool2d(1)
        self.batchnorm = nn.BatchNorm2d(1792)

    def forward(self, input):
        output = self.backbone.extract_features(input)
        output = self.last_layer(output)
        output = self.batchnorm(output)
        output = output.view(8, -1)
        return output