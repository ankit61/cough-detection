from ResNet3D import resnet10
from BaseResNetRunner import BaseResNetRunner
import constants

def get_visual_model():
    return resnet10(
            num_classes=constants.NUM_CLASSES, 
            sample_duration=constants.VIDEO_FPS, 
            sample_size=constants.INPUT_FRAME_WIDTH
        )

class ResNet3DRunner(BaseResNetRunner):
    def __init__(self, load_paths=None):
        net = get_visual_model()
        input_idx = 0
        super(ResNet3DRunner, self).__init__(net=net, input_index=input_idx, load_paths=load_paths)