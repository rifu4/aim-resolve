"""CLI script for training a segmentation model for source detection."""
import os
import click

@click.command()
@click.option('--config', required=True, help='Path to the YAML config file')
@click.option('--cuda_device', default='', help='CUDA device to use (e.g. "0", "0,1", ...), default is "" for CPU')

def main(config, cuda_device):
    """Train or resume training a segmentation model from a YAML configuration.

    Parameters
    ----------
    config : str
        Path to the YAML configuration file.
    cuda_device : str
        CUDA device identifier, empty string for CPU.
    """
    if str(cuda_device) == '':
        device = 'cpu'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
        device = 'cuda'

    import lightning as pl
    from lightning.pytorch.loggers import WandbLogger
    from aim_resolve import Dataset, SegmentationModel, yaml_load

    # Load YAML config
    dct = yaml_load(config)
    name = dct.get('name')
    odir = dct.get('odir')

    # Load dataset and data loaders
    dataset = Dataset.build(**dct['dataset'])
    train_loader = dataset.train_loader(**dct['dataloader']['train'])
    valid_loaders = dataset.valid_loader(**dct['dataloader']['valid'])

    # build new model or resume from saved model
    resume = dct.get('resume', False)
    if resume:
        rname = resume if isinstance(resume, str) else name
        print('\nload model:\nname:', rname)
        model = SegmentationModel.load(rname, odir)
    else:
        print('\nbuild model:')
        model = SegmentationModel.build(**dct['model'])

    model = model.to(device)

    # setup the training
    train = dct.get('train', False)
    if train:
        # Initialize wandb if requested
        logger = train.pop('logger', None)
        if logger == 'wandb':
            logger = WandbLogger(
                project=model.config['arch'],
                name=name,
                config=dct,
            )

        # train the model
        trainer = pl.Trainer(logger=logger, **train)

        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loaders,
        )

        # save the model
        model.save(name, odir)

    # Plot predictions if requested
    if 'plot' in dct:
        print('\nplot predictions:')
        model = model.to('cpu')
        model.plot_predictions(dataset, name, **dct['plot'])
        print('done')


if __name__ == '__main__':
    main()
