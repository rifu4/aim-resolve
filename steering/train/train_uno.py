import os
import sys
import torch
from neuralop import Trainer
from neuralop.models import UNO
from neuralop.utils import count_model_params
from torch.optim import AdamW

from aim_resolve import Dataset, BCELoss, yaml_load, yaml_save, plot_arrays



_, yfile = sys.argv[0], sys.argv[1]

dct = yaml_load(yfile)

device = dct['device']
name = dct['name']
odir = dct['odir']

yaml_save(dct, os.path.join(odir, name + '.yml'))


# load the dataset and setup the dataloaders
dataset = Dataset.build(**dct['dataset'])

train_loader = dataset.train_loader(**dct['dataloader']['train'])
valid_loaders = dataset.valid_loader(**dct['dataloader']['valid'])


# setup the UNO model
model = UNO(**dct['model'])
model = model.to(device)
n_params = count_model_params(model)
print('\nModel: \n', model)
print('n params:', n_params)


# setup optimizer and scheduler
optimizer = AdamW(model.parameters(), **dct['optimizer'])
print('\nOptimizer: \n', optimizer)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **dct['scheduler'])


# setup the loss functions
train_loss = BCELoss()
valid_losses = {'bce': BCELoss()}
print(f'\nLosses: \n   train: {train_loss}\n   valid: {valid_losses}')


# setup the trainer
trainer = Trainer(
    model=model,
    device=device,
    **dct['trainer'],
)

# train the model
trainer.train(
    train_loader=train_loader,
    test_loaders=valid_loaders,
    optimizer=optimizer,
    scheduler=scheduler, 
    regularizer=False, 
    training_loss=train_loss,
    eval_losses=valid_losses,
)


# save the model
torch.save(model.state_dict(), os.path.join(odir, name + '.pth'))


# plot the predictions for training and validation datasets

n_copies = dct['plot'].pop('n_copies', 5)

arrays = []
for i in range(n_copies):
    data = train_loader.dataset[i]
    x = torch.from_numpy(data['x'])
    pred = model(x.unsqueeze(0).to(device)).cpu()
    y = data['y']
    arrays += [x[0].detach().numpy(), y[0], pred.detach().numpy()[0,0].clip(0,1), y[1], pred.detach().numpy()[0,1].clip(0,1)]

plot_arrays(
    arrays,
    label = n_copies * ['Input', 'True Points', 'Predicted Points', 'True Objects', 'Predicted Objects'],
    name = name + '_train',
    cols = n_copies,
    **dct['plot']
)
