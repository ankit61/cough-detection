from BaseRunner import BaseRunner
import torch.nn as nn
import torch
import constants

class BaseResNetRunner(BaseRunner):
    def __init__(self, net, input_index=None, load_paths=None):
        self.input_idx = input_index
        loss_fn = nn.BCEWithLogitsLoss()
        if torch.cuda.is_available():
            net = net.cuda()
        optimizers = [torch.optim.SGD(net.parameters(), lr=constants.LR, momentum=constants.MOMENTUM, weight_decay=constants.WEIGHT_DECAY)]

        super(BaseResNetRunner, self).__init__(
            [net], 
            loss_fn, 
            optimizers, 
            best_metric_name='loss', 
            should_minimize_best_metric=True,
            load_paths=load_paths
        )

    def do_forward_pass(self, batch):
        return self.nets[0](batch[self.input_idx]).squeeze(dim=1)
    
    def get_metrics(self, batch):
        pred = self.do_forward_pass(batch)
        loss = self.loss_fn(pred, batch[2].float())
        acc  = self.get_accuracy(pred, batch[2])
        pred_min = pred.min().item()
        pred_max = pred.max().item()
        pred_std  = pred.std().item()
        pred_01_rate = (pred < 0).float().sum() / pred.numel()
        gt_01_rate  = (batch[2] < 1).float().sum() / batch[2].numel()

        return loss, [
            ('loss', loss.mean().item()), ('acc', acc), 
            ('pred_01_rate', pred_01_rate), ('gt_01_rate', gt_01_rate),
            ('pred_min', pred_min), ('pred_max', pred_max), ('pred_std', pred_std)
        ]

    def train_batch_and_get_metrics(self, batch):
        #forward pass
        loss, metrics = self.get_metrics(batch)

        #backward pass
        self.optimizers[0].zero_grad()
        loss.backward()
        self.output_gradient_norms(self.global_step)
        self.output_weight_norms(self.global_step)

        #step
        self.optimizers[0].step()

        return metrics

    def test_batch_and_get_metrics(self, batch):
        return self.get_metrics(batch)[1]

    def get_accuracy(self, pred, gt):
        return ((pred > 0) == gt).sum().item() / gt.shape[0]
