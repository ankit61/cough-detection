from torchvision.models import resnet18
from BaseResNetRunner import BaseResNetRunner
import torch.nn as nn
import constants
from MultiStreamDNN import get_audio_model

class ResNetRunner(BaseResNetRunner):
    def __init__(self, load_paths=None):
        net = get_audio_model()
        input_idx = 1
        super(ResNetRunner, self).__init__(net=net, input_index=input_idx, load_paths=load_paths)