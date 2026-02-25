#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

#%%
import jax
import pickle
import numpy as np
import aim_resolve as aim

jax.config.update("jax_enable_x64", True)

#%%
def box_markers(cfg, ps_map, grid, it):
    import numpy as np
    from aim_resolve import draw_boxes

    px, py = np.argwhere(ps_map > 0).T
    ps_mrk = dict(x=px, y=py, s=10, c='white', marker='+')
    box_map = draw_boxes(cfg.sections, grid, it)
    ox, oy = np.argwhere(box_map > 0).T
    oj_mrk = dict(x=ox, y=oy, s=0.1, c='white', marker=',')

    return dict(ps_mrk=ps_mrk, oj_mrk=oj_mrk)

#%%
dir = '/scratch/users/rfuchs/packages/aim-resolve/steering/runs/fast_vi_1f_1024_1z_b'

mf_rec = '3_rec_2z_6f'
mf_it = 5

ARCMIN2RAD = np.pi / 60 / 180
AS2RAD = ARCMIN2RAD / 60
CONV_FACTOR = 1000 * AS2RAD**2


opt_yml = f'{dir}/opt/{mf_rec}/opt.yml'
print('load:', opt_yml)

optim_cfg = aim.OptimizeKLConfig.from_file(opt_yml, aim.get_builders)

sky_mf = optim_cfg.instantiate_sec(f'sky.{mf_it}')
print('sky components:', [c.prefix for c in sky_mf.models])

freq = [f'{round(f/1e6)} MHz' for f in sky_mf.freq]
print('sky freqs:', freq)
ref_freq = freq[1]
print('ref freq:', ref_freq)

with open(f'{dir}/opt/{mf_rec}/last.pkl', "rb") as f:
    samples_mf, *_ = pickle.load(f)
print('samples:', len(samples_mf))

setup_cfg = aim.SetupKLConfig.from_file(opt_yml)
sky_ps = sky_mf.points[0]
ps_map = aim.map_signal(sky_ps.points.grid, sky_ps.grid)(np.ones(sky_ps.shape)).sum(axis=0)
markers_mf = box_markers(setup_cfg, ps_map, sky_mf.grid, mf_it)
print('markers:', [f'{k}: {len(v["x"])}' for k, v in markers_mf.items()])

#%%
min_mf = 2.5e-3
max_mf = np.max(samples_mf.mean(sky_mf.ref_freq_model) * CONV_FACTOR)
print('vmin:', min_mf, '\nvmax:', max_mf)

plot_dict = dict(
    name = None,
    odir = None,
    norm = 'log',
    vmin = min_mf,
    vmax = max_mf,
    cmap = 'inferno',
    cbar = False,
    ticks = 0,
    dpi=300,
)

#%%
sky_ref_mf = samples_mf.mean(sky_mf.ref_freq_model) * CONV_FACTOR

pot_ref_mf = samples_mf.mean(sky_mf.points_and_objects.ref_freq_model) * CONV_FACTOR

bg_ref_mf = samples_mf.mean(sky_mf.background.ref_freq_model) * CONV_FACTOR

pot_alpha = samples_mf.mean(sky_mf.points_and_objects.spectral_index)
pot_alpha = np.where(pot_ref_mf > 5e-3, pot_alpha, np.nan)
amin, amax = -3, None

contours = {'array': sky_ref_mf, 'colors': 'white', 'levels': [1e-2, 1e-1], 'linewidths': 0.25}

print('plotting ...')
aim.plot_arrays(
    array = sky_ref_mf, 
    marker = markers_mf,
    callback = lambda fig, ax: fig.text(0.085, 0.90, ref_freq, fontsize=15, c='white'),
    **plot_dict,
)
aim.plot_arrays(
    array = sky_ref_mf, 
    callback = lambda fig, ax: fig.text(0.085, 0.90, ref_freq, fontsize=15, c='white'),
    **plot_dict,
)
aim.plot_arrays(
    array = bg_ref_mf, 
    callback = lambda fig, ax: fig.text(0.085, 0.90, ref_freq, fontsize=15, c='black'),
    **plot_dict,
)
aim.plot_arrays(
    array = pot_ref_mf, 
    callback = lambda fig, ax: fig.text(0.085, 0.90, ref_freq, fontsize=15, c='black'),
    **plot_dict,
)
aim.plot_arrays(
    array = pot_alpha, 
    # callback = lambda fig, ax: fig.text(0.085, 0.90, 'Spectral Index', fontsize=15, c='black'),
    contour = contours,
    **plot_dict | dict(vmin=amin, vmax=amax, norm='linear'),
)

#%%
sky_val_mf = samples_mf.mean(sky_mf) * CONV_FACTOR
grd = sky_mf.grid
sky_val = aim.map_signal(grd, grd.update(space=grd.spc//2))(sky_val_mf)

# mmin = 2*[2e-3] + 2*[4e-3]
# mmax = [5.5, 5.5, 11, 11]
# print(mmax)

# def callback(fig, axes):
#     fig.text(0.045, 0.950, '1012 MHz', fontsize=15, c='white')
#     fig.text(0.51, 0.950, '1112 MHz', fontsize=15, c='white')
#     fig.text(0.045, 0.485, '1368 MHz', fontsize=15, c='white')
#     fig.text(0.51, 0.485, '1427 MHz', fontsize=15, c='white')

print('plotting ...')
aim.plot_arrays(
    sky_val, 
    rows=2,
    grid_kwargs=dict(hspace=-0, wspace=-0, width_ratios=[1,1,1], height_ratios=[1,1]),
    # callback=callback,
    **plot_dict,
)

#%%
sky_val_mf = samples_mf.mean(sky_mf) * CONV_FACTOR
grd = sky_mf.grid
sky_val = aim.map_signal(grd, grd.update(space=grd.spc//2))(sky_val_mf)

print('plotting ...')
for i,f in enumerate(freq):
    aim.plot_arrays(
        sky_val[i], 
        callback = lambda fig, ax: fig.text(0.085, 0.90, f, fontsize=15, c='white'),
        **plot_dict,
    )

#%%
comps_mf = [sky_mf.objects[0], sky_mf.objects[1]]

comp_ref_mf = [samples_mf.mean(c.ref_freq_model) * CONV_FACTOR for c in comps_mf]
comp_alpha = [samples_mf.mean(c.spectral_index) for c in comps_mf]
min_cs, min_ca = 2.5e-3, 5e-3


print('plotting ...')
for c,a in zip(comp_ref_mf, comp_alpha):
    a = np.where(c > min_ca, a, np.nan)
    contours = {'array': c, 'colors': 'white', 'levels': [1e-2, 1e-1], 'linewidths': 0.5}
    aim.plot_arrays(
        array = c, 
        **plot_dict | dict(vmin=min_cs),
    )
    aim.plot_arrays(
        array = a,
        contour=contours,
        **plot_dict | dict(vmin=-3, vmax=0, norm='linear'),
    )

#%%
import matplotlib.pyplot as plt

min_scatter = 5e-3
print('plotting scatter ...')
for idx, (c_ref, c_alpha) in enumerate(zip(comp_ref_mf, comp_alpha), start=1):
    mask = (c_ref > min_scatter) & np.isfinite(c_alpha)
    x = c_ref[mask].ravel()
    y = c_alpha[mask].ravel()

    plt.figure()
    plt.scatter(x, y, s=2, alpha=0.35, edgecolors='none')
    plt.xlabel('Sky brightness [mJy/arcsec$^2$]')
    plt.ylabel('Spectral index')
    plt.title(f'Component {idx}')
    plt.grid(alpha=0.2)

#%%
print('plotting scatter log scale ...')
for idx, (c_ref, c_alpha) in enumerate(zip(comp_ref_mf, comp_alpha), start=1):
    mask = (c_ref > min_scatter) & np.isfinite(c_alpha)
    x = c_ref[mask].ravel()
    y = c_alpha[mask].ravel()

    plt.figure()
    plt.scatter(x, y, s=2, alpha=0.35, edgecolors='none')
    plt.xscale('log')
    plt.xlabel('Sky brightness [mJy/arcsec$^2$]')
    plt.ylabel('Spectral index')
    plt.title(f'Component {idx}')
    plt.grid(alpha=0.2)

#%%
# Identify different regions in component 1 based on spatial structure
# Using 3x3 grid regionalization

print('plotting scatter with 3x3 grid regions (log scale) ...')

colors_9 = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'magenta', 'brown']
region_names = ['top-left', 'top-center', 'top-right', 
                'mid-left', 'mid-center', 'mid-right',
                'bot-left', 'bot-center', 'bot-right']

for idx, (c_ref, c_alpha) in enumerate(zip(comp_ref_mf, comp_alpha), start=1):
    mask = (c_ref > min_scatter) & np.isfinite(c_alpha)
    
    # Get pixel coordinates where mask is True
    y_coords, x_coords = np.where(mask)
    
    # Divide into 3x3 grid
    height, width = c_ref.shape
    y_split = np.linspace(0, height, 4)
    x_split = np.linspace(0, width, 4)
    
    # Assign region ID to each pixel
    region_id = np.zeros(c_ref.shape, dtype=int)
    for i in range(3):
        for j in range(3):
            grid_mask = ((y_coords >= y_split[i]) & (y_coords < y_split[i+1]) &
                        (x_coords >= x_split[j]) & (x_coords < x_split[j+1]))
            region_id[y_coords[grid_mask], x_coords[grid_mask]] = i * 3 + j
    
    x = c_ref[mask].ravel()
    y = c_alpha[mask].ravel()
    regions = region_id[mask].ravel()
    
    plt.figure(figsize=(10, 6))
    for region in range(9):
        region_mask = regions == region
        if np.any(region_mask):
            plt.scatter(x[region_mask], y[region_mask], s=2, alpha=0.5,
                       edgecolors='none', color=colors_9[region], 
                       label=region_names[region])
    
    plt.xscale('log')
    plt.xlabel('Sky brightness [mJy/arcsec$^2$]')
    plt.ylabel('Spectral index')
    plt.title(f'Component {idx} (3x3 grid regions, log scale)')
    plt.legend(loc='best', fontsize=7, ncol=3)
    plt.grid(alpha=0.2)

#%%
# Create 2D image colored by 3x3 grid regions (same coloring as scatter plot)

print('plotting 2D grid region maps ...')

colors_9_rgb = np.array([
    [1, 0, 0],        # red
    [1, 0.647, 0],    # orange
    [1, 1, 0],        # yellow
    [0, 0.502, 0],    # green
    [0, 1, 1],        # cyan
    [0, 0, 1],        # blue
    [0.627, 0, 1],    # purple
    [1, 0, 1],        # magenta
    [0.647, 0.165, 0] # brown
])

region_names = ['top-left', 'top-center', 'top-right', 
                'mid-left', 'mid-center', 'mid-right',
                'bot-left', 'bot-center', 'bot-right']

for idx, (c_ref, c_alpha) in enumerate(zip(comp_ref_mf, comp_alpha), start=1):
    mask = (c_ref > min_scatter) & np.isfinite(c_alpha)
    
    # Get pixel coordinates where mask is True
    y_coords, x_coords = np.where(mask)
    
    # Divide into 3x3 grid
    height, width = c_ref.shape
    y_split = np.linspace(0, height, 4)
    x_split = np.linspace(0, width, 4)
    
    # Assign region ID to each pixel
    region_id = np.full(c_ref.shape, -1, dtype=int)  # -1 for masked out pixels
    for i in range(3):
        for j in range(3):
            grid_mask = ((y_coords >= y_split[i]) & (y_coords < y_split[i+1]) &
                        (x_coords >= x_split[j]) & (x_coords < x_split[j+1]))
            region_id[y_coords[grid_mask], x_coords[grid_mask]] = i * 3 + j
    
    # Create RGB image colored by region
    rgb_image = np.zeros((height, width, 3))
    for region in range(9):
        region_mask = region_id == region
        rgb_image[region_mask] = colors_9_rgb[region]
    
    # Pixels outside the mask stay black
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb_image.transpose(1, 0, 2), origin='lower')
    ax.set_title(f'Component {idx} (3x3 grid regions)')
    
    # Add legend for region colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_9_rgb[i], label=region_names[i]) 
                       for i in range(9)]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=7, ncol=3, 
              bbox_to_anchor=(1.05, 1), frameon=True)
    
    plt.tight_layout()

#%%
# Bin by spectral index and sum brightness values

print('plotting binned brightness vs spectral index ...')

for idx, (c_ref, c_alpha) in enumerate(zip(comp_ref_mf, comp_alpha), start=1):
    mask = (c_ref > min_scatter) & np.isfinite(c_alpha)
    x = c_ref[mask].ravel()
    y = c_alpha[mask].ravel()
    
    # Create bins for spectral index and sum brightness for each bin
    n_bins = 100
    binned_brightness, bin_edges = np.histogram(y, bins=n_bins, weights=x)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Filter out empty bins
    valid_mask = binned_brightness > 0
    binned_brightness_valid = binned_brightness[valid_mask]
    bin_centers_valid = bin_centers[valid_mask]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(binned_brightness_valid, bin_centers_valid, s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
    plt.xlabel('Summed sky brightness [mJy/arcsec$^2$]')
    plt.ylabel('Spectral index')
    plt.title(f'Component {idx} (binned by spectral index)')
    plt.grid(alpha=0.2)

#%%
# Same as above but with log scale on brightness

print('plotting binned brightness vs spectral index (log scale) ...')

for idx, (c_ref, c_alpha) in enumerate(zip(comp_ref_mf, comp_alpha), start=1):
    mask = (c_ref > min_scatter) & np.isfinite(c_alpha)
    x = c_ref[mask].ravel()
    y = c_alpha[mask].ravel()
    
    # Create bins for spectral index and sum brightness for each bin
    n_bins = 100
    binned_brightness, bin_edges = np.histogram(y, bins=n_bins, weights=x)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Filter out empty bins
    valid_mask = binned_brightness > 0
    binned_brightness_valid = binned_brightness[valid_mask]
    bin_centers_valid = bin_centers[valid_mask]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(binned_brightness_valid, bin_centers_valid, s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
    plt.xscale('log')
    plt.xlabel('Summed sky brightness [mJy/arcsec$^2$]')
    plt.ylabel('Spectral index')
    plt.title(f'Component {idx} (binned by spectral index, log scale)')
    plt.grid(alpha=0.2)

# %%
