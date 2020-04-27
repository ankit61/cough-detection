from MultiStreamDNN import get_audio_model, get_visual_model_conv3D, get_visual_model_conv2D
from MultiStreamDNN import MultiStreamDNN
from BaseRunner import BaseRunner
import torch
import constants
import torch.nn as nn


class EnsembleModelRunner(BaseRunner):
    def __init__(self, load_paths, model_type='all'):

        assert model_type in ['all', 'conv3D_MFCCs', 'conv2D_MF']

        if model_type == 'all':
            nets = [
                MultiStreamDNN(
                    get_visual_model_conv3D(),
                    get_audio_model()
                ),
                MultiStreamDNN(
                    get_visual_model_conv2D(),
                    get_audio_model()
                )
            ]

            if torch.cuda.is_available():
                for i in range(len(nets)):
                    nets[i] = nets[i].cuda()

            optimizers = [
                torch.optim.SGD(
                    nets[0].parameters(),
                    lr=constants.SGD_LR,
                    momentum=constants.SGD_MOMENTUM,
                    weight_decay=constants.SGD_WEIGHT_DECAY
                ),
                torch.optim.Adam(
                    nets[1].parameters(),
                    lr=constants.ADAM_LR,
                    weight_decay=constants.ADAM_WEIGHT_DECAY
                )
            ]
        elif model_type == 'conv3D_MFCCs':
            nets = [
                MultiStreamDNN(
                    get_visual_model_conv3D(),
                    get_audio_model()
                )
            ]

            if torch.cuda.is_available():
                for i in range(len(nets)):
                    nets[i] = nets[i].cuda()

            optimizers = [
                torch.optim.SGD(
                    nets[0].parameters(),
                    lr=constants.SGD_LR,
                    momentum=constants.SGD_MOMENTUM,
                    weight_decay=constants.SGD_WEIGHT_DECAY
                )
            ]
        elif model_type == 'conv2D_MF':
            nets = [
                MultiStreamDNN(
                    get_visual_model_conv2D(),
                    get_audio_model()
                )
            ]

            if torch.cuda.is_available():
                for i in range(len(nets)):
                    nets[i] = nets[i].cuda()

            optimizers = [
                torch.optim.Adam(
                    nets[0].parameters(), 
                    lr=constants.ADAM_LR, 
                    weight_decay=constants.ADAM_WEIGHT_DECAY
                )
            ]

        super(EnsembleModelRunner, self).__init__(
            nets,
            loss_fn=nn.BCEWithLogitsLoss(),
            optimizers=optimizers,
            best_metric_name='acc',
            should_minimize_best_metric=False,
            load_paths=load_paths
        )

    def train_batch_and_get_metrics(self, batch):
        metrics = []

        for i in range(len(self.nets)):
            # forward pass
            pred = self.nets[i](batch[2 * i], batch[2 * i + 1]).squeeze(dim=1)
            loss, m = self.get_metrics(pred, batch[-1], get_original_loss=True)

            for j in range(len(m)):
                metrics.append((m[j][0] + '_' + str(i).zfill(2), m[j][1]))

            # backward pass
            self.optimizers[i].zero_grad()
            loss.backward()

            # step
            self.optimizers[i].step()

        self.output_gradient_distributions(self.global_step)
        self.output_weight_norms(self.global_step)

        return metrics

    def do_forward_pass(self, batch):
        outputs = []
        for i in range(len(self.nets)):
            outputs.append(
                self.nets[i](batch[2 * i], batch[2 * i + 1]).squeeze(dim=1)
            )

        pred = torch.zeros_like(outputs[0])
        for i in range(len(outputs)):
            pred += outputs[i] / len(outputs)

        return pred

    def test_batch_and_get_metrics(self, batch):
        pred = self.do_forward_pass(self, batch)

        return self.get_metrics(pred, batch[-1])

    def get_metrics(self, pred, gt, get_original_loss=False):
        loss            = self.loss_fn(pred, gt.float())
        acc             = self.get_accuracy(pred, gt)
        pred_min        = pred.min().item()
        pred_max        = pred.max().item()
        pred_std        = pred.std().item()
        pred_01_rate    = (pred < 0).float().sum() / pred.numel()
        gt_01_rate      = (gt < 1).float().sum() / gt.numel()

        metrics = [
            ('loss', loss.mean().item()),
            ('acc', acc),
            ('pred_01_rate', pred_01_rate),
            ('gt_01_rate', gt_01_rate),
            ('pred_min', pred_min),
            ('pred_max', pred_max),
            ('pred_std', pred_std)
        ]

        if get_original_loss:
            return loss, metrics
        else:
            return metrics

    def get_accuracy(self, pred, gt):
        return ((pred > 0) == gt).sum().item() / gt.shape[0]
