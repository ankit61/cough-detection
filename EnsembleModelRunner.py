from MultiStreamDNN import get_audio_model, get_visual_model_conv3D, get_visual_model_conv2D
from MultiStreamDNN import MultiStreamDNN
from BaseRunner import BaseRunner
import torch
import constants
import torch.nn as nn
from torchvision.models import resnet34
from ray import tune


class AudioMF(nn.Module):
    def __init__(self):
        super(AudioMF, self).__init__()
        self.net = resnet34(num_classes=1)
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net.bn1 = nn.BatchNorm2d(64)

    def forward(self, v, a):
        return self.net(a)


class EnsembleModelRunner(BaseRunner):
    def __init__(self, load_paths, use_tune,
                 tune_config, model_type='all',):
        self.use_tune = use_tune
        self.tune_config = tune_config

        nets = [
            MultiStreamDNN(
                get_visual_model_conv3D(),
                get_audio_model()
            ),
            MultiStreamDNN(
                get_visual_model_conv3D(),
                get_audio_model()
            )
        ]

        if torch.cuda.is_available():
            for i in range(len(nets)):
                nets[i] = nets[i].cuda()
        if self.use_tune:
            optimizers = [
                torch.optim.SGD(
                    nets[0].parameters(),
                    lr=self.tune_config['conv3D_MFCCs_lr'],
                    momentum=self.tune_config['conv3D_MFCCs_momentum'],
                    weight_decay=self.tune_config['conv3D_MFCCs_weight_decay']
                ),
                torch.optim.Adam(
                    nets[1].parameters(),
                    lr=self.tune_config['conv2D_MF_lr'],
                    weight_decay=self.tune_config['conv2D_MF_weight_decay']
                )
            ]
        else:
            optimizers = [
                torch.optim.SGD(
                    nets[0].parameters(),
                    lr=constants.LRS[0],
                    momentum=constants.MOMENTUMS[0],
                    weight_decay=constants.WEIGHT_DECAYS[0]
                ),
                torch.optim.Adam(
                    nets[1].parameters(),
                    lr=constants.LRS[1],
                    weight_decay=constants.WEIGHT_DECAYS[1]
                )
            ]

        if model_type == 'conv3D_MFCCs':
            nets = [nets[0]]
            optimizers = [optimizers[0]]
        elif model_type == 'conv2D_MF':
            nets = [nets[1]]
            optimizers = [optimizers[1]]
        elif model_type != 'all':
            raise Exception('unknown model_type')

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

    def get_batch_size(self, batch):
        return batch[-1].shape[0]

    def test_batch_and_get_metrics(self, batch):
        pred = self.do_forward_pass(batch)

        return self.get_metrics(pred, batch[-1])

    def post_train_processing(self):
        if self.use_tune:
            tune.track(best_acc=self.best_metric_val)

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
