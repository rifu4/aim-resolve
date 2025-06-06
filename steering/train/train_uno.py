import os
# os.environ['JAX_PLATFORM_NAME'] = 'gpu'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


import torch
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F
from neuralop.models import UNO
from neuralop import Trainer
from torch.optim import AdamW
from neuralop.utils import count_model_params

import numpy as np

from aim_resolve import Dataset, BCELoss, yaml_load, plot_arrays

# torch.cuda.set_device(1)

# device = "cuda:1" if torch.cuda.is_available() else "cpu"



_, yfile = sys.argv[0], sys.argv[1]

dct = yaml_load(yfile)

dataset = Dataset.build(**dct['dataset'])

train_loader = dataset.train_loader(**dct['dataloader']['train'])
valid_loaders = dataset.valid_loader(**dct['dataloader']['valid'])


model = UNO(**dct['model'])
model = model.to(dct['device'])

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()


#Create the optimizer
optimizer = AdamW(
    model.parameters(), 
    lr=8e-3, 
    weight_decay=1e-4,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


train_loss = BCELoss()
valid_losses = {'bce': BCELoss()}


print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Valid: {valid_losses}')


# Create the trainer
trainer = Trainer(
    model=model,
    device=dct['device'],
    **dct['trainer'],
)

trainer.train(
    train_loader=train_loader,
    test_loaders=valid_loaders,
    optimizer=optimizer,
    scheduler=scheduler, 
    regularizer=False, 
    training_loss=train_loss,
    eval_losses=valid_losses,
)


# However, for practicity, we created a minimal example that
# i) fits in just a few Mb of memory
# ii) can be trained quickly on CPU
#
# In practice we would train a Neural Operator on one or multiple GPUs

# test_samples = test_loaders['128'].dataset
test_samples = train_loader.dataset

n = 5

val = []
for index in range(n):
    data = test_samples[index]
    x = data['x']
    x = torch.from_numpy(x)
    out = model(x.unsqueeze(0).to(dct['device'])).cpu()
    y = data['y']
    val += [x[0].detach().numpy(), y[0], out.detach().numpy()[0,0].clip(0,1), y[1], out.detach().numpy()[0,1].clip(0,1)]

plot_arrays(
    val,
    label = n * ['Input', 'True Points', 'Predicted Points', 'True Objects', 'Predicted Objects'],
    cols = 5,
)