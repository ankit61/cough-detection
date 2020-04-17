from BaseResNetRunner import BaseResNetRunner
import constants
import torch.nn as nn
from MultiStreamDNN import get_visual_model_conv3D

class ResNet3DRunner(BaseResNetRunner):
    def __init__(self, load_paths=None):
        net = get_visual_model_conv3D()
        input_idx = 0
        super(ResNet3DRunner, self).__init__(net=net, input_index=input_idx, load_paths=load_paths)
