from torchvision.models import resnet18
from BaseResNetRunner import BaseResNetRunner
import torch.nn as nn
import constants

def get_audio_model():
    net = resnet18(pretrained=True)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.bn1 = nn.BatchNorm2d(64)
    net.fc = nn.Linear(net.fc.in_features, constants.NUM_CLASSES)

    return net

class ResNetRunner(BaseResNetRunner):
    def __init__(self, load_paths=None):
        net = get_audio_model()
        input_idx = 1
        super(ResNetRunner, self).__init__(net=net, input_index=input_idx, load_paths=load_paths)