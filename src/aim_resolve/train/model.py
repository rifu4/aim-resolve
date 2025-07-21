import os
import torch
import lightning as pl
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss, SoftBCEWithLogitsLoss
from torch.optim import lr_scheduler

from .dataset import Dataset
from ..model.util import check_type



class SegmentationModel(pl.LightningModule):
    '''Base class for segmentation models for saving, loading, and plotting, based on pytorch lightning.'''

    def __init__(self, model, loss_fn, config):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optim_fn = torch.optim.Adam(self.parameters(), **config['optimizer'])
        self.sched_fn = lr_scheduler.CosineAnnealingLR(self.optim_fn, **config['scheduler'])
        self.config = config

        # initialize step metrics
        self.training_step_outputs = []
        self.validation_step_outputs = {}
        self.test_step_outputs = []

    @classmethod
    def build(cls, *, arch, model_args, loss, optimizer, scheduler):
        '''
        Build the model from the given parameters.

        Parameters
        ----------
        arch : str
            The arch of the model to build.
        model_args : dict
            Dictionary containing the arguments for the model.
        loss : str
            The loss function to use for training the model.
        optimizer : dict
            The optimizer to use for training the model.
        scheduler : dict
            The learning rate scheduler to use for training the model.
        '''
        check_type(arch, str)
        check_type(model_args, dict)
        check_type(loss, str)
        check_type(optimizer, dict)
        check_type(scheduler, dict)

        if arch == 'uno':
            from neuralop.models.uno import UNO
            model = UNO(**model_args)
        else:
            model = smp.create_model(arch, **model_args)

        if loss == 'bce':
            loss_fn = SoftBCEWithLogitsLoss()
        elif loss == 'dice':
            loss_fn = DiceLoss(mode='multilabel')
        elif loss == 'jaccard':
            loss_fn = JaccardLoss(mode='multilabel')
        else:
            raise ValueError(f'Unknown loss function: `{loss}`')

        config = dict(
            arch=arch,
            model_args=model_args,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        return cls(model, loss_fn, config)

    def save(self, name, odir=''):
        '''
        Save the model model, arch, and config to a `.pth` file.

        Parameters
        ----------
        name : str
            Name of the file to save the model to
        odir : str, optional
            Output directory for the file, by default ''
        '''
        os.makedirs(odir, exist_ok=True)
        if not name.endswith('.pth'):
            name_pth = name + '.pth'

        torch.save(self.config | {'state_dict': self.state_dict()}, os.path.join(odir, name_pth))

        return

    @classmethod
    def load(cls, name, odir=''):
        '''
        Load a model from a file.

        Parameters
        ----------
        name : str
            Name of the file to load the model from
        odir : str, optional
            Output directory for the file, by default ''
        '''
        if not name.endswith('.pth'):
            name_pth = name + '.pth'

        config = torch.load(os.path.join(odir, name_pth))
        state_dict = config.pop('state_dict', None)
        model = cls.build(**config)
        model.load_state_dict(state_dict)
        model.eval()

        return model
    
    def plot_predictions(self, dataset, name, odir='', n_copies=5, label=False, **kwargs):
        '''
        Plot a number of samples.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing the data to plot
        name : str
            Name of the plot
        odir : str, optional
            Output directory for the plot, by default ''
        n_copies : int, optional
            Number of samples to plot, by default 5
        label : bool, optional
            Whether to add labels ['points', 'objects', 'sky'] to the plot, by default False
        **kwargs : additional keyword arguments
            Additional keyword arguments to pass to the plotting function
        '''
        from ..plot.arrays import plot_arrays

        check_type(dataset, Dataset)

        data_loaders = {f'{name}_train': dataset.train_loader(batch_size=n_copies)}
        data_loaders |= {f'{name}_valid{k}': v for k,v in dataset.valid_loader(batch_size=n_copies).items()}

        for nm,dl in data_loaders.items():
            sample = next(iter(dl))
            x = sample['x'].detach().numpy()
            y = sample['y'].detach().numpy()
            pred = self.forward_sigmoid(sample['x'])
            pred = pred.detach().numpy()

            arrays = []
            for i in range(n_copies):
                arrays += [x[i,0], y[i,0], pred[i,0], y[i,1], pred[i,1]]
            labels = ['image', 'true points', 'predicted points', 'true objects', 'predicted boxes'] * n_copies

            [kwargs.pop(key, None) for key in ('rows', 'cols')]

            plot_arrays(
                array = arrays,
                label = labels if label else None,
                rows = n_copies,
                cols = 5,
                name = nm,
                odir = odir,
                **kwargs,
            )

        return
    
    def forward(self, image):
        return self.model.forward(image)

    def forward_sigmoid(self, image):
        pred = self.forward(image)
        pred = torch.sigmoid(pred.squeeze())
        return pred > 0.5
    
    def shared_step(self, batch, stage):
        image = batch['x']
        mask = batch['y']

        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode='multilabel')

        return {
            'loss': loss,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x['tp'] for x in outputs])
        fp = torch.cat([x['fp'] for x in outputs])
        fn = torch.cat([x['fn'] for x in outputs])
        tn = torch.cat([x['tn'] for x in outputs])

        # get average loss
        losses = torch.stack([x['loss'] for x in outputs])
        avg_loss = losses.mean()

        # calculate per image per class IoU score
        iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction=None
        )

        # get per class mIoU across all images
        cls_iou = iou.mean(dim=0)

        metrics = {
            f'{stage} loss': avg_loss,
            f'{stage} mIoU': cls_iou.mean(),
            f'{stage}_cls0_mIoU': cls_iou[0],
            f'{stage}_cls1_mIoU': cls_iou[1],
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, 'train')
        # append the metics of each step to the
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, 'train')
        # empty set output list
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Create key for each dataloader, e.g., 'valid0', 'valid1', etc.
        key = f'valid{dataloader_idx}'
        valid_loss_info = self.shared_step(batch, key)

        # Initialize list for this dataloader if not already
        if key not in self.validation_step_outputs:
            self.validation_step_outputs[key] = []
        self.validation_step_outputs[key].append(valid_loss_info)

        return valid_loss_info

    def on_validation_epoch_end(self):
        # Loop over all dataloader outputs
        for key, outputs in self.validation_step_outputs.items():
            self.shared_epoch_end(outputs, key)

        # Clear for next epoch
        self.validation_step_outputs.clear()


    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, 'test')
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, 'test')
        # empty set output list
        self.test_step_outputs.clear()
        return

    def configure_optimizers(self):
        return {
            'optimizer': self.optim_fn,
            'lr_scheduler': {
                'scheduler': self.sched_fn,
                'interval': 'step',
                'frequency': 1,
            },
        }
        return
