from torchvision.models import resnet18, resnet34
from ResNet3D import resnet18 as resnet183D
import torch.nn as nn
import torch
import constants


def get_audio_model():
    net = resnet34()
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.bn1 = nn.BatchNorm2d(64)
    net.fc = nn.Sequential()

    return net


def get_visual_model_conv3D():
    net = resnet183D(
            num_classes=constants.NUM_CLASSES,
            sample_duration=constants.VIDEO_FPS,
            sample_size=constants.INPUT_FRAME_WIDTH
        )

    net.fc = nn.Sequential()

    return net


def get_visual_model_conv2D():
    net = resnet18()
    net.conv1 = nn.Conv2d(constants.VIDEO_FPS * constants.CHUNK_SIZE * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.bn1 = nn.BatchNorm2d(64)
    net.fc = nn.Sequential()

    return net


class MultiStreamDNN(nn.Module):
    def __init__(self, visual_model, audio_model):
        super(MultiStreamDNN, self).__init__()
        self.visual_model = visual_model
        self.audio_model = audio_model

        video_features_len = constants.HOLISTIC_FEATURES_LEN

        self.mlp = nn.Sequential(
            nn.Linear(video_features_len, 512),
            nn.ReLU(),
            nn.Dropout(constants.DROPOUT_PROB),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(constants.DROPOUT_PROB),
            nn.Linear(256, 1)
        )

    def forward(self, visual_stream, audio_stream):
        visual_featrues = self.visual_model(visual_stream)
        audio_features  = self.audio_model(audio_stream)

        video_features = torch.cat([visual_featrues, audio_features], dim=1)
        return self.mlp(video_features)
