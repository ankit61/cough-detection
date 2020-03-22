from torchvision.models import resnet18
from BaseResNetRunner import BaseResNetRunner
import torch.nn as nn
import constants

class ResNetRunner(BaseResNetRunner):
    def __init__(self):
        net = resnet18(pretrained=True)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.bn1 = nn.BatchNorm2d(64)
        net.fc = nn.Linear(net.fc.in_features, constants.NUM_CLASSES)
        input_idx = 1
        super(ResNetRunner, self).__init__(net=net, input_index=input_idx)