from torchvision.models import resnet18
from ResNet3D import resnet10
import torch.nn as nn
import torch
import constants

def get_audio_model():
    net = resnet18(pretrained=True)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.bn1 = nn.BatchNorm2d(64)
    net.fc = nn.Linear(net.fc.in_features, constants.NUM_CLASSES)

    return net

def get_visual_model():
    return resnet10(
            num_classes=constants.NUM_CLASSES, 
            sample_duration=constants.VIDEO_FPS, 
            sample_size=constants.INPUT_FRAME_WIDTH
        )

class MultiStreamDNN(nn.Module):
    def __init__(self):
        super(MultiStreamDNN, self).__init__()
        self.visual_model = get_visual_model()
        self.audio_model = get_audio_model()
        
        video_features_len = self.audio_model.fc.in_features + self.visual_model.fc.in_features

        self.visual_model.fc = nn.Sequential()
        self.audio_model.fc = nn.Sequential()

        self.mlp = nn.Sequential(
            nn.Linear(video_features_len, 256),
            nn.ReLU(),
            nn.Dropout(constants.DROPOUT_PROB),
            nn.Linear(256, 1)
        )

    def forward(self, visual_stream, audio_stream):
        visual_featrues = self.visual_model(visual_stream)
        audio_featrues  = self.audio_model(audio_stream)

        video_features = torch.cat([visual_featrues, audio_featrues], dim=1)
        return self.mlp(video_features)
