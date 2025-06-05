import os
import sys
from aim_resolve import SetupKLConfig, yaml_load, yaml_safe, merge_dicts



def main():
    _, files = sys.argv[0], sys.argv[1:]
    mdl_yml, base_yml, pipe_yml = files

    # load the basic model yaml-file into SetupKLConfig class
    cfg = SetupKLConfig.from_file(mdl_yml)

    # load the pipeline yaml-file and pop not needed keys
    pipe_dct = yaml_load(pipe_yml)
    [pipe_dct.pop(k) for k in ['n_it', 'unet']]
    
    # adjust model yaml-file (opt, lh, and data section)
    odir = pipe_dct.pop('odir')
    fun = pipe_dct['data']['fun']
    cfg.modify_sec('opt', base='base_opt', odir='->base_opt/odir + opt/0_rec')
    cfg.modify_sec('lh.0', fun=fun)
    cfg.modify_sec('data.0', **pipe_dct.pop('data'))

    # add noise scaling configuration for the likelihood
    if 'noise' in pipe_dct:
        cfg.modify_sec('lh.0', noise=pipe_dct.pop('noise'))

    # get noise level for likelihood if fun is 'exp'
    if 'max_std' in cfg.sections['data.0']:
        cfg.modify_sec('lh.0', noise=dict(max_std=cfg.sections['data.0']['max_std']))

    # get correct kernels if fast-resolve is used
    if 'radio' in fun and 'fast' in fun:
        kname = cfg.sections['data.0']['fname'].split('/')[-1].split('.')[0]
        ksize = pipe_dct['space_bg']['shape'][0]
        kfov = pipe_dct['space_bg']['fov'][0]
        cfg.modify_sec(
            sec_key = 'lh.0', 
            psf_pixels = 3000,
            response_kernel = f'test/kernel/rk_{kname}_{kfov}_{ksize}.pkl', 
            noise_kernel = f'test/kernel/nk_{kname}_{kfov}_{ksize}.pkl',
        )

    # extract callback, extra, and transition keys from pipe_dct
    callback = pipe_dct.pop('callback') if 'callback' in pipe_dct else False
    extra = pipe_dct.pop('extra') if 'extra' in pipe_dct else False
    trans = pipe_dct.pop('trans') if 'trans' in pipe_dct else False
    key = pipe_dct.pop('key') if 'key' in pipe_dct else 0
    rerun = pipe_dct.pop('rerun') if 'rerun' in pipe_dct else True

    # load and overwrite sections of the base yaml-file with pipe_dct sections (like opt, trans, i0, space,  plot, ...)
    base_dct = yaml_load(base_yml)
    base_dct = merge_dicts([dict(base_opt=dict(odir=odir, key=key, rerun=rerun)), base_dct, pipe_dct], merge_base='True')

    # create output directories
    os.makedirs(odir + '/files/', exist_ok=True)
    os.makedirs(odir + '/plots/', exist_ok=True)
    if callback:
        os.makedirs(odir + '/callback/', exist_ok=True)
    if extra:
        os.makedirs(odir + '/extra/', exist_ok=True)
    if trans:
        os.makedirs(odir + '/trans/', exist_ok=True)

    # save the new model yaml-file and base yaml-file
    cfg.to_file(odir + '/files/0_pre.yml')
    yaml_safe(base_dct, odir + '/files/base.yml')


if __name__ == '__main__':
    main()
