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
from neuralop.training import AdamW
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader, random_split

from aim_resolve import TensorDataset, yaml_load, get_dataset, plot_arrays



torch.cuda.set_device(1)

device = "cuda:1" if torch.cuda.is_available() else "cpu"



train_data, test_data = get_dataset(
    name='train128_1k', 
    odir='/scratch/users/rfuchs/packages/tile-nifty/tests/data/files',
    dtype='float32',
    size=1000,
    coos=True,
)
print(train_data[0].shape, train_data[1].shape, train_data[0].dtype, train_data[1].dtype)
print(test_data[0].shape, test_data[1].shape, test_data[0].dtype, test_data[1].dtype)

plot_arrays(train_data[0][:10], cols=3, transpose=True)
plot_arrays(train_data[1][:10], cols=2, transpose=True)

train_data = TensorDataset(train_data)
test_data = TensorDataset(test_data)


train_loader = DataLoader(
    train_data, 
    batch_size=32, 
    num_workers=0,
    pin_memory=True,
    persistent_workers=False,
)
    
test_loaders = {}
test_loaders[128] = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    persistent_workers=False,
)

model = UNO(in_channels=3, 
            out_channels=2, 
            hidden_channels=64, 
            projection_channels=64,
            positional_embedding=None,
            uno_out_channels=[32], #,64,64,64,32],
            uno_n_modes=[[16,16]], #,[8,8],[8,8],[8,8],[16,16]],
            uno_scalings=[[1.0,1.0]], #,[0.5,0.5],[1,1],[2,2],[1,1]],
            horizontal_skips_map=None,
            channel_mlp_skip="linear",
            n_layers = 1,
            non_linearity=F.relu,
            domain_padding=0.2)

model = model.to(device)

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()