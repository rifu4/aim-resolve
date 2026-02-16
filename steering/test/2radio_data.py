#%%
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

#%%
import jax
import numpy as np
import pandas as pd
from copy import deepcopy
from aim_resolve import OptimizeKLConfig, SignalSpace, SignalGrid, signal_response, radio_data, image_data, get_builders, plot_arrays

#%%
data_dir = '/scratch/users/rfuchs/packages/aim-resolve/steering/data/runs/files/test512_10.pkl'

data = image_data(fname=data_dir, idx=1)
data.grid = SignalGrid.build(space=data.grid.space, fov=['1deg', '1deg'])
print(data)

space = SignalSpace.build(shape=data.grid.space, fov=['1deg', '1deg'])
print(space)

#%%
obs_dir = '/scratch/users/rfuchs/data/eso_986-1137mhz.npz'

obs = radio_data(fname=obs_dir)
obs = obs.get_freqs([3])
print(obs)

#%%
vis = signal_response(space, obs, True)(data.val)
print(vis.shape)

#%%
key = jax.random.PRNGKey(4)
n_std = 0.1
noise = n_std * jax.random.normal(key, vis.shape, dtype=vis.dtype)
noisy_vis = vis + noise

print(vis)
print(noisy_vis)

#%%
obs_new = deepcopy(obs)

idx = pd.MultiIndex.from_product([obs.pol, obs.freq], names=['pol', 'freq'])
vis_T = np.transpose(noisy_vis, (1,0,2)).reshape((noisy_vis.shape[1], -1))
df_vis = pd.DataFrame(vis_T, columns=idx)
obs_new._vis = df_vis
print(obs_new)

#%%
dirty = obs_new.dirty_image(data.grid)

plot_arrays(dirty, norm='log', dpi=100, rows=1, vmin=1e-7)

#%%
print(vis)
print(obs_new.vis)

#%%
obs_new.save('/scratch/users/rfuchs/packages/aim-resolve/steering/data/runs/files/exp_1062mhz')
