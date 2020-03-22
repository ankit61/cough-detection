from CoughDataset import CoughDataset
from ResNetRunner import ResNetRunner
from ResNet3DRunner import ResNet3DRunner
from MultiStreamDNNRunner import MultiStreamDNNRunner

import constants
from torch.utils.data import DataLoader

TEST_RUNNER = MultiStreamDNNRunner()

dataset = CoughDataset()

runner = TEST_RUNNER

data_loader = DataLoader(dataset, batch_size=constants.BATCH_SIZE, shuffle=True)
runner.train(data_loader, constants.EPOCHS)