#%%
import os
# os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

#%%
import jax
import pickle
import numpy as np
from aim_resolve import OptimizeKLConfig, SetupKLConfig, get_builders, plot_arrays, map_signal

jax.config.update("jax_enable_x64", True)

#%%
ARCMIN2RAD = np.pi / 60 / 180
AS2RAD = ARCMIN2RAD / 60
CONV_FACTOR = 1000 * AS2RAD**2


def latest_model_file(directory):
    import os
    import re

    max_file = None
    max_it = -1
    for filename in os.listdir(directory + '/files'):
        match = re.compile(r'(\d+)_pre\.yml$').match(filename)
        if match:
            it = int(match.group(1))
            if it > max_it:
                max_it = it
                max_file = filename

    return directory + f'/files/{max_file}'



def box_markers(cfg, ps_map, grid, it):
    import numpy as np
    from aim_resolve import draw_boxes

    px, py = np.argwhere(ps_map == 1).T
    ps_mrk = dict(x=px, y=py, s=1, c='white', marker='+')
    box_map = draw_boxes(cfg.sections, grid, it)
    ox, oy = np.argwhere(box_map == 1).T
    oj_mrk = dict(x=ox, y=oy, s=0.1, c='white', marker=',')
    
    return dict(ps_mrk=ps_mrk, oj_mrk=oj_mrk)

#%%
eso_dir = '/Users/rf/Development/packages/aim-resolve/steering/runs/fast_vi_1f_1024_1z_b'


eso_yml = latest_model_file(eso_dir)
print('load:', eso_yml)

optim_cfg = OptimizeKLConfig.from_file((eso_dir + '/files/base.yml', eso_yml), get_builders, 'total')
optim_cfg.sections['data.0']['fname'] = '/Users/rf/Development/data/eso_986-1137mhz.npz'


sky_models = [sec for sec in optim_cfg.sections if sec.startswith('sky.')]
print('sky models:', sky_models)

skies_sf, samples_sf, markers_sf = [], [], []
for it,sec in enumerate(sky_models):
    sky_it = optim_cfg.instantiate_sec(sec)
    skies_sf.append(sky_it)
    with open(eso_dir + f'/opt/{it}_rec/last.pkl', "rb") as f:
        smp_it, *_ = pickle.load(f)
    samples_sf.append(smp_it)

    if it == 0:
        markers_sf += [{}]
    else:
        setup_cfg = SetupKLConfig.from_file(eso_yml)
        sky_ps = sky_it.points[0]
        ps_map = map_signal(sky_ps.points.grid, sky_ps.grid)(np.ones(sky_ps.shape))
        markers_sf.append(box_markers(setup_cfg, ps_map, sky_it.grid, it))

print('samples:', [len(s) for s in samples_sf])
print('markers:', [[f'{k}: {len(v["x"])}' for k, v in m.items()] for m in markers_sf])

#%%
sky_it = 3
min_sf = 1e-3
max_sf = 0
max_sf = np.max(samples_sf[sky_it].mean(skies_sf[sky_it]) * CONV_FACTOR)
print('vmin:', min_sf, '\nvmax:', max_sf)

plot_dict = dict(
    name = None,
    odir = None,
    norm = 'log',
    vmin = min_sf,
    vmax = max_sf,
    cmap = 'inferno',
    cbar = False,
    ticks = 0,
    dpi=300,
)

#%%
means_sf = []
for sky, smp in zip(skies_sf, samples_sf):
    means_sf += [smp.mean(sky) * CONV_FACTOR, ]


def callback(fig, axes):
    fig.text(0.049, 0.09, 'Iteration 1', fontsize=15, c='white')
    fig.text(0.353, 0.09, 'Iteration 2', fontsize=15, c='white')
    fig.text(0.657, 0.09, 'Iteration 3', fontsize=15, c='white')


print('plotting ...')
plot_arrays(
    array = means_sf[0:3], 
    marker = markers_sf[1:4], 
    rows=1, 
    callback=callback,
    grid_kwargs=dict(hspace=-0.5, wspace=-0.3, width_ratios=[1,1,1], height_ratios=[1]),
    **plot_dict,
)

#%%
zoom = 2


smp_val_sf = []
for smp in samples_sf[sky_it]:
    val = skies_sf[sky_it](smp) * CONV_FACTOR
    if zoom > 1:
        grd = skies_sf[sky_it].grid
        val = map_signal(grd, grd.update(space=grd.spc//2))(val)
    smp_val_sf.append(val)


def callback(fig, axes):
    fig.text(0.045, 0.950, 'Sample 1', fontsize=15, c='white')
    fig.text(0.51, 0.950, 'Sample 2', fontsize=15, c='white')
    fig.text(0.045, 0.485, 'Sample 3', fontsize=15, c='white')
    fig.text(0.51, 0.485, 'Sample 4', fontsize=15, c='white')


print('plotting ...')
plot_arrays(
    smp_val_sf, 
    rows=2,
    grid_kwargs=dict(hspace=-0.71, wspace=-0.7, width_ratios=[1,1], height_ratios=[1,1]),
    callback=callback,
    **plot_dict,
)

#%%
zoom = 2

mean, std = samples_sf[sky_it].mean_and_std(skies_sf[sky_it])
skyz_mean_sf = mean * CONV_FACTOR
skyz_runc_sf = std / mean
if zoom:
    grd = skies_sf[sky_it].grid
    skyz_mean_sf = map_signal(grd, grd.update(space=grd.spc//2))(skyz_mean_sf)
    skyz_runc_sf = map_signal(grd, grd.update(space=grd.spc//2))(skyz_runc_sf)


print('plotting ...')
plot_arrays(
    array = skyz_mean_sf, 
    callback = lambda fig, ax: fig.text(0.085, 0.90, 'Posterior mean', fontsize=15, c='white'),
    **plot_dict,
)
plot_arrays(
    array = skyz_runc_sf,
    contour={'array': skyz_mean_sf, 'colors': 'white', 'levels': [1e-2, 1e-1], 'linewidths': 0.5},
    callback = lambda fig, ax: fig.text(0.085, 0.90, 'Relative uncertainty', fontsize=15, c='black'),
    **plot_dict | dict(vmin=None, vmax=None),
)

#%%
sky_mean_sf = samples_sf[sky_it].mean(skies_sf[sky_it]) * CONV_FACTOR

pot_mean_sf = samples_sf[sky_it].mean(skies_sf[sky_it].points_and_objects) * CONV_FACTOR


print('plotting ...')
plot_arrays(
    array = sky_mean_sf, 
    marker = markers_sf[sky_it],
    # callback = lambda fig, ax: fig.text(0.085, 0.90, '1062 MHz', fontsize=15, c='white'),
    **plot_dict,
)
plot_arrays(
    array = sky_mean_sf, 
    # callback = lambda fig, ax: fig.text(0.085, 0.90, '1062 MHz', fontsize=15, c='white'),
    **plot_dict,
)
plot_arrays(
    array = pot_mean_sf, 
    # callback = lambda fig, ax: fig.text(0.085, 0.90, '1062 MHz', fontsize=15, c='black'),
    **plot_dict,
)

#%%
comp_mean_sf = [samples_sf[sky_it].mean(c) * CONV_FACTOR for c in [skies_sf[sky_it].objects[0], skies_sf[sky_it].objects[1]]]
    

print('plotting ...')
for c in comp_mean_sf:
    plot_arrays(
        array = c, 
        **plot_dict,
    )

#%%
sky_pt_sf = skies_sf[sky_it].copy()

pt_models = []
for m in sky_pt_sf.models:
    if m not in [skies_sf[sky_it].background, skies_sf[sky_it].objects[0], skies_sf[sky_it].objects[1]]:
        pt_models.append(m)

sky_pt_sf.models = pt_models
pt_mean_sf = samples_sf[sky_it].mean(sky_pt_sf) * CONV_FACTOR
        
        
print('plotting ...')
plot_arrays(
    array = pt_mean_sf, 
    **plot_dict,
)

#%%
