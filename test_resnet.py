from CoughDataset import CoughDataset
from ResNetRunner import ResNetRunner
from ResNet3DRunner import ResNet3DRunner
from MultiStreamDNNRunner import MultiStreamDNNRunner
import argparse
import constants
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser('arg parser')

parser.add_argument('--mode', '-m', default='train', choices=['train', 'test'])
parser.add_argument('--load-path', '-lp', default='')
parser.add_argument('--runner', '-r', default='multistream', choices=['multistream', 'resnet3d', 'resnet'])
parser.add_argument('--data-dir', '-d', default=constants.DATA_BASE_DIR)
args =  parser.parse_args()

dataset = CoughDataset(root_dir=args.data_dir)

if not args.load_path:
    args.load_path = None

if args.runner == 'multistream':
    runner = MultiStreamDNNRunner(load_paths=[args.load_path])
elif args.runner == 'resnet':
    runner = ResNetRunner(load_paths=[args.load_path])
else:
    runner = ResNet3DRunner(load_paths=[args.load_path])

data_loader = DataLoader(
    dataset, 
    batch_size=constants.BATCH_SIZE, 
    shuffle=(args.mode == 'train')
)
if args.mode == 'train':
    runner.train(data_loader, constants.EPOCHS)
else:
    runner.test(data_loader)