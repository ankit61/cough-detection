from MultiStreamDNN import get_audio_model, get_visual_model_conv3D, get_visual_model_conv2D
from MultiStreamDNN import MultiStreamDNN
from BaseRunner import BaseRunner
import torch

class EnsembleModelRunner(BaseRunner):
    def __init__(self, load_paths):
        nets = [
            MultiStreamDNN(
                self.get_audio_model(), 
                self.get_visual_model_conv3D()
            ),
            MultiStreamDNN(
                self.get_audio_model(), 
                self.get_visual_model_conv2D()
            )
        ]

        if torch.cuda.is_available():
            for i in range(len(nets)):
                nets[i] = nets[i].cuda()

        optimizers = [
            torch.optim.SGD(
                net.parameters(), 
                lr=ENSEMBLE_LRS[i], 
                momentum=ENSEMBLE_MOMENTUMS[i], 
                weight_decay=ENSEMBLE_WEIGHT_DECAYS[i]
            )
            for i in range(len(nets))
        ]

        super(EnsembleModelRunner, self).__init__(
            nets, 
            loss_fn = nn.BCEWithLogitsLoss(),
            optimizers=optimizers,
            best_metric_name='acc',
            should_minimize_best_metric=False,
            load_paths=load_paths
        )

    def train_batch_and_get_metrics(self, batch):
        metrics = []

        for i in range(len(self.nets)):
            #forward pass
            pred = self.nets[i](batch[2 * i], batch[2 * i + 1]).squeeze(dim=1)
            loss, m = self.get_metrics(pred_1, batch[4], get_original_loss=True)

            for j in range(len(m)):
                m[j][0] = m[j][0] + '_' + str(i).zfill(2)

            metrics += m

            #backward pass
            self.optimizers[i].zero_grad()
            loss.backward()

            #step
            self.optimizers[i].step()

        self.output_gradient_distributions(self.global_step)
        self.output_weight_norms(self.global_step)

        return metrics

    def test_batch_and_get_metrics(self, batch):
        output_1 = self.nets[0](batch[0], batch[1]).squeeze(dim=1)
        output_2 = self.nets[1](batch[2], batch[3]).squeeze(dim=1)

        pred = (output_1 + output_2) / 2

        return self.get_metrics(pred, batch[4])

    def get_metrics(self, pred, gt, get_original_loss = False):
        loss            = self.loss_fn(pred, gt.float())
        acc             = self.get_accuracy(pred, gt)
        pred_min        = pred.min().item()
        pred_max        = pred.max().item()
        pred_std        = pred.std().item()
        pred_01_rate    = (pred < 0).float().sum() / pred.numel()
        gt_01_rate      = (gt < 1).float().sum() / gt.numel()

        metrics = [
            ('loss', loss.mean().item()),
            ('acc', acc.mean().item()),
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