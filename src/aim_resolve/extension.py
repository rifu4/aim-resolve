from .optimize.set_config import SetupKLConfig
from .optimize.yml import yaml_load



def extension_func(
        mode,
        **kwargs,
):
    '''
    Versatile extension function -> performs the wanted extension specified in the 'mode' parameter
    
    Parameters:
    -----------
    mode : str
        Extension mode. Can be `zoom` and `freq`.
    kwargs : dict
        Additional keyword arguments passed to the extension functions (see extension functions).
    '''
    if mode == 'freq':
        return freq_extension(**kwargs)
    elif mode == 'zoom':
        return zoom_extension(**kwargs)
    else:
        raise TypeError(f'Unknown extension mode. Available modes are `zoom` and `freq`, but got mode `{mode}`.')



def freq_extension(*,
        odir,
        file,
        freq,
        base = 'base.yml',
        ref_freq_index = 1,
        **kwargs,
):
    '''
    Perform a multi-frequency extension on an existing optimization configuration file.
    
    Parameters:
    -----------
    odir : str
        Output directory where the existing configuration file is located.
    file : str
        Name of the existing configuration file.
    freq : list or int
        List of frequencies, a list of frequency indices or the number of frequencies.
    base : str
        Name of the base configuration file. Default is `base.yml`.
    ref_freq_index : int
        Index of the reference frequency in the `freq` list.
    kwargs : dict
        Additional options to overwrite an existing section of the initial config file.
    '''
    base = f'{odir}/files/{base}'
    base_dct = yaml_load(base)

    cfg = SetupKLConfig.from_file(f'{odir}/files/{file}')

    if not isinstance(freq, list):
        raise TypeError('The `freq` parameter must be a list of frequencies.')
    else:
        freq = sorted(freq)

    cfg.add_it(fix_keys=('data.0',), del_comp=False)

    cfg.modify_sec('opt', 
        resume=cfg.sections['opt']['odir'],
        odir=cfg.sections['opt']['odir'] + f'_{len(freq)}f',
    )

    for sec in cfg.sections:
        if 'sky_' in sec and f'.{cfg.it}' in sec:
            if 'p' in sec:
                cfg.modify_sec(sec, freq=freq, params=dict(base='params_ps', ref_freq_index=ref_freq_index))
            else:
                cfg.modify_sec(sec, freq=freq, params=dict(base='params_mf', ref_freq_index=ref_freq_index))

    pk_dir = '_'.join(cfg.sections[f'lh.{cfg.it}']['psf_kernel_fn'].split('_')[:2])
    nk_dir = '_'.join(cfg.sections[f'lh.{cfg.it}']['n_inv_kernel_fn'].split('_')[:2])
    fq_rng = f'{int(min(freq)/1e6)}mhz-{int(max(freq)/1e6)}mhz_{len(freq)}f'
    bg_fov = base_dct['grid_bg']['fov'][0]
    bg_ker = cfg.sections[f'lh.{cfg.it}']['psf_kernel_fn'].split('_')[-1].split('.')[0]
    lh_dct = {  
        'psf_kernel_fn': f'{pk_dir}_{fq_rng}_{bg_fov}_{bg_ker}.pkl',
        'n_inv_kernel_fn': f'{nk_dir}_{fq_rng}_{bg_fov}_{bg_ker}.pkl',
    }
    cfg.modify_sec(f'lh.{cfg.it}', **lh_dct)

    cfg.add_sec(f'trans.{cfg.it}',
        lh_old=f'=lh.{cfg.it-1}',
        lh_new=f'=lh.{cfg.it}',
        mode='freq',
    )
    cfg.modify_sec(f'opt.{cfg.it}', base=f'base_opt.0', transitions=f'=trans.{cfg.it}')

    for key,val in kwargs.items():
        cfg.modify_sec(key, **val)

    cfg = fun2mode(cfg)

    ext_file = f'{odir}/files/{file.split(".")[0]}_{len(freq)}f.yml'
    cfg.to_file(ext_file)

    return base, ext_file


def zoom_extension(*,
        odir,
        file,
        zoom,
        base = 'base.yml',
        **kwargs,
):
    '''
    Perform a zoom extension on an existing optimization configuration file.
    
    Parameters:
    -----------
    odir : str
        Output directory where the existing configuration file is located.
    file : str
        Name of the existing configuration file.
    zoom : int
        Zoom factor to apply.
    base : str
        Name of the base configuration file. Default is `base.yml`.
    kwargs : dict
        Additional options to overwrite an existing section of the initial config file.
    '''
    base = f'{odir}/files/{base}'
    base_dct = yaml_load(base)

    cfg = SetupKLConfig.from_file(f'{odir}/files/{file}')

    cfg.add_it(fix_keys=('data.0',), del_comp=False)

    cfg.modify_sec('opt', 
        resume=cfg.sections['opt']['odir'],
        odir=cfg.sections['opt']['odir'] + f'_{zoom}z',
    )

    for sec in cfg.sections:
        if 'sky_' in sec and f'.{cfg.it}' in sec and not 'bg' in sec:
            grid = cfg.sections[sec]['grid'] | dict(factor=zoom)
            cfg.modify_sec(sec, grid=grid)

    pkdir = '_'.join(cfg.sections[f'lh.{cfg.it}']['psf_kernel_fn'].split('_')[:-1])
    nkdir = '_'.join(cfg.sections[f'lh.{cfg.it}']['n_inv_kernel_fn'].split('_')[:-1])
    ksize = zoom * base_dct['grid_bg']['space'][0]
    cfg.modify_sec(f'lh.{cfg.it}', psf_kernel_fn=f'{pkdir}_{ksize}.pkl', n_inv_kernel_fn=f'{nkdir}_{ksize}.pkl')

    cfg.add_sec(f'trans.{cfg.it}',
        lh_old=f'=lh.{cfg.it-1}',
        lh_new=f'=lh.{cfg.it}',
        mode='zoom',
        opt_dct = dict(base='base_trans'),
        odir = f'{odir}/opt/{cfg.it-1}_rec_{zoom}z/trans',
    )
    cfg.modify_sec(f'opt.{cfg.it}', base=f'base_opt.n', transitions=f'=trans.{cfg.it}')

    for key,val in kwargs.items():
        cfg.modify_sec(key, **val)

    cfg = fun2mode(cfg)

    ext_file = f'{odir}/files/{file.split(".")[0]}_{zoom}z.yml'
    cfg.to_file(ext_file)

    return base, ext_file



def fun2mode(cfg):
    for sec in cfg.sections:
        if 'fun' in cfg.sections[sec]:
            fun = cfg.sections[sec].pop('fun')
            if 'lh' in sec:
                fun = 'fast' if 'fast' in fun else 'radio' if 'radio' in fun else 'image'
            if 'data' in sec:
                fun = 'radio' if 'radio' in fun else 'image'
            cfg.sections[sec]['mode'] = fun
    return cfg
