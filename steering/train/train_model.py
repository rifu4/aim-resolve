import os
import click

@click.command()
@click.option('--config', required=True, help='Path to the YAML config file')
@click.option('--cuda_device', default='', help='CUDA device to use (e.g. "0", "0,1", ...), default is "" for CPU')

def main(config, cuda_device):
    if str(cuda_device) == '':
        device = 'cpu'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
        device = 'cuda'

    import wandb
    from neuralop.utils import count_model_params
    from aim_resolve import Dataset, SegmentationModel, yaml_load

    # Load YAML config
    dct = yaml_load(config)

    # Initialize wandb if requested
    if all(dct.get(k, False) for k in ['model', 'train', 'wandb']): 
        wandb.init(
            project=dct['model']['mode'],
            config=dct,
        )

    # Load dataset and data loaders
    dataset = Dataset.build(**dct['dataset'])
    train_loader = dataset.train_loader(**dct['dataloader']['train'])
    valid_loaders = dataset.valid_loader(**dct['dataloader']['valid'])

    # build new model or resume from saved model
    if 'resume' in dct:
        if isinstance(dct['resume'], str):
            print('\nload model:\nname:', dct['resume'])
            model = SegmentationModel.load(dct['resume'], dct['odir'])
        else:
            print('\nload model:\nname:', dct['name'])
            model = SegmentationModel.load(dct['name'], dct['odir'])
    elif 'model' in dct:
        print('\nbuild model:\nname:', dct['model'].get('mode', None))
        model = SegmentationModel.build(**dct['model'])
    else:
        raise ValueError('No model specified in the config file.')

    model = model.to(device)
    n_params = count_model_params(model)
    print('\nModel: \n', model)
    print('n params:', n_params)

    # Train model if requested
    if 'train' in dct:
        model.train_model(train_loader, valid_loaders, **dct['train'])
        model.save(dct['name'], dct['odir'])

    # Plot predictions if requested
    if 'plot' in dct:
        print('\nplot predictions:')
        model = model.to('cpu')
        model.plot_predictions(dataset, dct['name'], **dct['plot'])
        print('done')


if __name__ == '__main__':
    main()
