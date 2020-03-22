from CoughDataset import CoughDataset
from ResNetRunner import ResNetRunner
from ResNet3DRunner import ResNet3DRunner
import constants
from torch.utils.data import DataLoader

TEST_AUDIO_STREAM = False

dataset = CoughDataset()

if TEST_AUDIO_STREAM:
    runner = ResNetRunner()
else:
    runner = ResNet3DRunner()

data_loader = DataLoader(dataset, batch_size=constants.BATCH_SIZE, shuffle=True)
runner.train(data_loader, constants.EPOCHS)