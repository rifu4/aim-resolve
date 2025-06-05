import sys
import jax

from aim_resolve import OptimizeKLConfig, ImageData, Observation, get_builders, plot_arrays, domain_keys



jax.config.update("jax_enable_x64", True)

# -> select correct mode: 'total' (exp, radio) | 'major' (fast-resolve)
mode = 'total'
_, yfile = sys.argv[0], sys.argv[1]

# instantiate the optimize-config class
cfg = OptimizeKLConfig.from_file(('config/base.yml', yfile), get_builders, mode)
odir = cfg.sections['opt']['odir'] + '/plots'

# initialize all signal models for each iteration
sig_dct = {sec: cfg.instantiate_sec(sec) for sec in cfg.sections if sec.startswith('sky')}

# print and plot the data
data_dct = {sec: cfg.instantiate_sec(sec) for sec in cfg.sections if sec.startswith('data')}
for dk,dv in data_dct.items():
    print(dv, '\n')
    if isinstance(dv, ImageData):
        plot_arrays(dv.val, name=dk, odir=odir, norm='log')
    elif isinstance(dv, Observation):
        plot_arrays(dv.dirty_image(next(iter(sig_dct.values())).space), name=dk, odir=odir)

# define a callback function to plot the results of the optimization after each iteration
def callback(samples, opt_state, *_):
    for key,sig in sig_dct.items():
        if domain_keys(sig).issubset(domain_keys(samples)):
            plot_arrays(samples.mean(sig), name=f'{opt_state.nit}_{key}', odir=odir, norm='log')

# run the optimization
samples, *_ = cfg.optimize_kl(callback=callback)
