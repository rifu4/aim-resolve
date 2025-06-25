import os
import torch
from neuralop import Trainer
from neuralop.models import UNO
from segmentation_models_pytorch import Unet
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .dataset import Dataset
from .loss import BCELoss
from ..model.util import check_type
from ..optimize.yml import yaml_load, yaml_save



class SegmentationModel():
    '''Base class for segmentation models, providing common functionality for saving and loading.'''

    def __init__(self, mode, parameters):
        raise NotImplementedError('This method should be defined in subclasses.')

    @staticmethod
    def build(*, mode, parameters):
        '''
        Build the model from the given parameters.

        Parameters
        ----------
        mode : str
            The type of model to build ('unet' or 'uno').
        parameters : dict
            Dictionary containing the parameters for the model.
        '''
        check_type(mode, str)
        check_type(parameters, dict)

        if mode == 'unet':
            return UnetModel(mode, parameters)
        elif mode == 'uno':
            return UNOModel(mode, parameters)
        else:
            raise ValueError(f'Unknown model mode: {mode}. Supported modes are `unet` and `uno`.')

    def save(self, name, odir=''):
        '''
        Save the model the parameters to a `.yml` file and the model to a `.pth` file.
        
        Parameters
        ----------
        name : str
            Name of the file to save the model to
        odir : str, optional
            Output directory for the file, by default ''
        '''
        os.makedirs(odir, exist_ok=True)

        if not name.endswith('.yml'):
            name_yml = name + '.yml'
        yaml_save(dict(mode=self.mode, parameters=self.params), os.path.join(odir, name_yml))

        if not name.endswith('.pth'):
            name_pth = name + '.pth'
        torch.save(self.state_dict(), os.path.join(odir, name_pth))

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
        if not name.endswith('.yml'):
            name_yml = name + '.yml'
        dct = yaml_load(os.path.join(odir, name_yml))
        model = cls.build(**dct)

        if not name.endswith('.pth'):
            name_pth = name + '.pth'
        model.load_state_dict(torch.load(os.path.join(odir, name_pth)))
        model.eval()

        return model
    
    def train_model(self, train_loader, valid_loaders, optimizer, scheduler, trainer, device='cuda'):
        '''
        Train the model using the provided dataset, optimizer, and scheduler.
        
        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader for the training dataset.
        valid_loaders : dict
            Dictionary of DataLoaders for validation datasets.
        optimizer : torch.optim.Optimizer
            The optimizer to use for training.
        scheduler : torch.optim.lr_scheduler._LRScheduler
            The learning rate scheduler.
        trainer : neuralop.Trainer
            The trainer instance for training the model.
        device : str, optional
            The device to use for training, by default `cuda`.
        '''
        check_type(train_loader, DataLoader)
        check_type(valid_loaders, dict)
        check_type(optimizer, dict)
        check_type(scheduler, dict)
        check_type(trainer, dict)
        check_type(device, str)

        optimizer = AdamW(self.parameters(), **optimizer)
        print('\nOptimizer: \n', optimizer)

        scheduler = CosineAnnealingLR(optimizer, **scheduler)

        train_loss = BCELoss()
        valid_losses = {'bce': BCELoss()}
        print(f'\nLosses: \n   train: {train_loss}\n   valid: {valid_losses}')

        trainer = Trainer(model=self, device=device, **trainer)

        trainer.train(
            train_loader=train_loader,
            test_loaders=valid_loaders,
            optimizer=optimizer,
            scheduler=scheduler, 
            training_loss=train_loss,
            eval_losses=valid_losses,
        )

    def sigmoid_predict(self, x):
        pred = self(x)
        pred = torch.sigmoid(pred.squeeze())
        return pred > 0.5
    
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

        if odir:
            if not odir.endswith(('plots', 'plots/')):
                odir = os.path.join(odir, 'plots')
            os.makedirs(odir, exist_ok=True)

        for nm,dl in data_loaders.items():
            sample = next(iter(dl))
            x = sample['x'].detach().numpy()
            y = sample['y'].detach().numpy()
            pred = self.sigmoid_predict(sample['x'])
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
    


class UnetModel(Unet, SegmentationModel):
    '''Generate a U-Net model for segmentation. Use `build` function to create the model.'''

    def __init__(self, mode, parameters):
        super().__init__(**parameters)
        self.mode = mode
        self.params = parameters



class UNOModel(UNO, SegmentationModel):
    '''Generate a UNO model for segmentation. Use `build` function to create the model.'''

    def __init__(self, mode, parameters):
        super().__init__(**parameters)
        self.mode = mode
        self.params = parameters
