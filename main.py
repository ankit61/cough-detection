from CoughDataset import CoughDataset
from ResNetRunner import ResNetRunner
from ResNet3DRunner import ResNet3DRunner
from MultiStreamDNNRunner import MultiStreamDNNRunner
import argparse
import constants
from torch.utils.data import DataLoader
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('arg parser')

parser.add_argument('--mode', '-m', default='train', choices=['train', 'test', 'gen_result'])
parser.add_argument('--load-path', '-lp', default='')
parser.add_argument('--runner', '-r', default='multistream', choices=['multistream', 'resnet3d', 'resnet'])
parser.add_argument('--data-dir', '-d', default=constants.DATA_BASE_DIR)
args =  parser.parse_args()

dataset = CoughDataset(root_dir=args.data_dir)

if not args.load_path:
    load_paths = None
else:
    load_paths = [args.load_path]

if args.runner == 'multistream':
    runner = MultiStreamDNNRunner(load_paths=load_paths)
elif args.runner == 'resnet':
    runner = ResNetRunner(load_paths=load_paths)
else:
    runner = ResNet3DRunner(load_paths=load_paths)

data_loader = DataLoader(
    dataset, 
    batch_size=constants.BATCH_SIZE, 
    shuffle=True
)
if args.mode == 'train':
    runner.train(data_loader, constants.EPOCHS)
elif args.mode == 'test':
    runner.test(data_loader)
else:
    file_output = {}
    graph_output = {}

    for i in range(len(dataset)):
        inp = [dataset[i][0].unsqueeze(dim=0), dataset[i][1].unsqueeze(dim=0)]
        if torch.cuda.is_available():
            inp[0] = inp[0].cuda()
            inp[1] = inp[1].cuda()
        prob = runner.do_forward_pass(inp).sigmoid().item()
        print(prob)
        original_file, interval = dataset.get_meta(i)
        
        if original_file in file_output:
            if prob > 0.5:
                file_output[original_file]['coughing'].append(interval)
            graph_output[original_file].append([interval[0], prob])
        else:
            if prob > 0.5:
                file_output[original_file] = {'coughing': [interval]}
            else:
                 file_output[original_file] = {'coughing': []}
            graph_output[original_file] = [[interval[0], prob]]

    print(graph_output)
    print(file_output)

    for f in graph_output:
        out_file = f.split('.')[0] + '.json'
        json.dump(file_output[f], open(out_file, 'w'))

        graph = np.array(graph_output[f])
        plt.ioff()
        fig = plt.figure()
        plt.plot(graph[:, 0], graph[:, 1])
        plt.xlabel('time')
        plt.ylabel('probability of coughing')
        plt.ylim([0, 1])
        plt.savefig(f.split('.')[0] + '.png')
        plt.close(fig)
        