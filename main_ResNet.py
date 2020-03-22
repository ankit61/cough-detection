from CoughDataset import CoughDataset
from ResNetRunner import ResNetRunner
import constants
from torch.utils.data import DataLoader

dataset = CoughDataset()
runner = ResNetRunner()
data_loader = DataLoader(dataset, batch_size=constants.BATCH_SIZE, shuffle=True)
runner.train(data_loader, constants.EPOCHS)