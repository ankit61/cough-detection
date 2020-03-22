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
        optimizers = [torch.optim.Adam(net.parameters(), lr=constants.LR, weight_decay=constants.WEIGHT_DECAY)]

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
    
    def train_batch_and_get_metrics(self, batch):
        #forward pass
        pred = self.do_forward_pass(batch)
        loss = self.loss_fn(pred, batch[2].float())
        acc  = self.get_accuracy(pred, batch[2])

        #backward pass
        self.optimizers[0].zero_grad()
        loss.backward()

        #step
        self.optimizers[0].step()

        return [('loss', loss.mean().item()), ('acc', acc)]

    def test_batch_and_get_metrics(self, batch):
        #forward pass
        pred = self.do_forward_pass(batch)
        loss = self.loss_fn(pred, batch[2])
        acc  = self.get_accuracy(pred, batch[2])

        return [('loss', loss.mean().item()), ('acc', acc)]

    def get_accuracy(self, pred, gt):
        return ((pred > 0) == gt).sum().item() / gt.shape[0]