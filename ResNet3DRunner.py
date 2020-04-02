from ResNet3D import resnet10
from BaseResNetRunner import BaseResNetRunner
import constants
import torch.nn as nn

def get_visual_model():
    from torchvision.models import resnet18

    net = resnet18(pretrained=True)
    net.conv1 = nn.Conv2d(constants.VIDEO_FPS * constants.CHUNK_SIZE * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.bn1 = nn.BatchNorm2d(64)

    return net

    #return resnet10(
    #        num_classes=constants.NUM_CLASSES, 
    #        sample_duration=constants.VIDEO_FPS, 
    #        sample_size=constants.INPUT_FRAME_WIDTH
    #    )

class ResNet3DRunner(BaseResNetRunner):
    def __init__(self, load_paths=None):
        net = get_visual_model()
        input_idx = 0
        super(ResNet3DRunner, self).__init__(net=net, input_index=input_idx, load_paths=load_paths)
