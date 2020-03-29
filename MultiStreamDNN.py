from ResNet3DRunner import get_visual_model
from ResNetRunner import get_audio_model
import torch.nn as nn
import torch

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
            nn.Dropout(),
            nn.Linear(256, 1)
        )

    def forward(self, visual_stream, audio_stream):
        visual_featrues = self.visual_model(visual_stream)
        audio_featrues  = self.audio_model(audio_stream)

        video_features = torch.cat([visual_featrues, audio_featrues], dim=1)
        return self.mlp(video_features)
