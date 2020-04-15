from CoughDataset import CoughDataset
from ResNetRunner import ResNetRunner
from ResNet3DRunner import ResNet3DRunner
from EnsembleModelRunner import EnsembleModelRunner
import argparse
import constants
from torch.utils.data import DataLoader
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

parser = argparse.ArgumentParser('arg parser')

parser.add_argument('--mode', '-m', default='train', choices=['train', 'test', 'gen_result'])
parser.add_argument('--load-path', '-lp', default='')
parser.add_argument('--data-dir', '-d', default=constants.DATA_BASE_DIR)
if os.path.exists(os.path.join(constants.DATA_BASE_DIR, 'test')):
    parser.add_argument('--test-dir', '-t', default=os.path.join(constants.DATA_BASE_DIR, 'test'))
else:
    parser.add_argument('--test-dir', '-t', default='')

args =  parser.parse_args()

dataset = CoughDataset(root_dir=args.data_dir, result_mode=(args.mode == 'gen_result'))
test_dataset = CoughDataset(root_dir=args.test_dir, result_mode=(args.mode == 'gen_result'))

if not args.load_path:
    load_paths = None
else:
    load_paths = [args.load_path]

runner = EnsembleModelRunner(load_paths=load_paths)

data_loader = DataLoader(
    dataset, 
    batch_size=constants.BATCH_SIZE, 
    shuffle=True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=constants.BATCH_SIZE, 
    shuffle=True
)

if args.mode == 'train':
    runner.train(data_loader, constants.EPOCHS, val_loader=test_loader)
elif args.mode == 'test':
    runner.test(data_loader)
else:
    runner.nets[0].eval()

    file_output = {}
    prob_graph_output = {}
    label_graph_output = {}

    for i in range(len(dataset)):
        inp = [dataset[i][0].unsqueeze(dim=0), dataset[i][1].unsqueeze(dim=0)]
        if torch.cuda.is_available():
            inp[0] = inp[0].cuda()
            inp[1] = inp[1].cuda()
        prob = runner.do_forward_pass(inp).sigmoid().item()
        original_file, interval = dataset.get_meta(i)
        
        if original_file in file_output:
            file_output[original_file]['coughing'].append([interval[0], prob])
            
            prob_graph_output[original_file].append([interval[0], prob])
            prob_graph_output[original_file].append([interval[1], prob])

            label_graph_output[original_file].append([interval[0], 1 if prob > 0.5 else 0])
            label_graph_output[original_file].append([interval[1], 1 if prob > 0.5 else 0])
        else:
            file_output[original_file] = {'coughing': [interval[0], prob]}
            
            prob_graph_output[original_file] = [[interval[0], prob], [interval[1], prob]]
            
            label_graph_output[original_file] = [[interval[0], 1 if prob > 0.5 else 0], [interval[1], 1 if prob > 0.5 else 0]]

    matplotlib.use('Agg') # to avoid displaying figure
    for f in prob_graph_output:
        #make json
        out_file = f.split('.')[0] + '.json'
        json.dump(file_output[f], open(out_file, 'w'))
        print('Saving ' + out_file + '...')

        #make probability figure
        fig_file = f.split('.')[0] + '_prob.png'
        graph = np.array(prob_graph_output[f])
        fig = plt.figure()
        plt.plot(graph[:, 0], graph[:, 1], '.-')
        plt.xlabel('time')
        plt.ylabel('probability of coughing')
        plt.ylim([0, 1.05])
        plt.savefig(fig_file)
        plt.close(fig)
        print('Saving ' + fig_file + '...')

        #make label figure
        fig_file = f.split('.')[0] + '_label.png'
        graph = np.array(label_graph_output[f])
        fig = plt.figure()
        plt.plot(graph[:, 0], graph[:, 1], '-')
        plt.xlabel('time')
        plt.ylabel('coughing label')
        plt.ylim([0, 1.05])
        plt.savefig(fig_file)
        plt.close(fig)
        print('Saving ' + fig_file + '...')