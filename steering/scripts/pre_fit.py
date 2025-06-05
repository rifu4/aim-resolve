import os
import jax
import pickle
import sys
from aim_resolve import OptimizeKLConfig, SetupKLConfig, get_builders, yaml_load, transition_addt, plot_mean_and_std


jax.config.update('jax_enable_x64', True)


def main():
    print(jax.devices())
    _, files = sys.argv[0], sys.argv[1:]
    mdl_yml, rec_pkl, base_yml, it = files

    # load model and base yaml-files and extract output directory and iteration number
    mdl_dct = yaml_load(mdl_yml)
    base_dct = yaml_load(base_yml)
    plt_dct = base_dct['base_plot']
    odir = base_dct['base_opt']['odir']

    # initialize the sky model using the OptimizeKLConfig class
    fun = mdl_dct[f'lh.{it}']['fun']
    cfg_mode = 'major' if 'radio' in fun and 'fast' in fun else 'total'
    cfg = OptimizeKLConfig.from_file([base_yml, mdl_yml], get_builders, cfg_mode)

    # instantiate the sky model of the current iteration
    sky_mdl = cfg.instantiate_sec(f'sky.{it}')

    # perform the transition from the old to the new sky model
    tra_pkl = f'{odir}/opt/{it}_pre/last.pkl'
    if not base_dct['base_opt']['rerun'] and os.path.isfile(tra_pkl):
        samples, offsets = pickle.load(open(tra_pkl, "rb"))
    else:
        samples, offsets = transition_addt(
            key = jax.random.PRNGKey(cfg.sections['opt']['key']),
            samples = pickle.load(open(rec_pkl, "rb"))[0],
            it = it,
            lh_old = cfg.instantiate_sec(f'lh.{int(it)-1}'),
            lh_new = cfg.instantiate_sec(f'lh.{it}'),
            sky_old = cfg.instantiate_sec(f'sky.{int(it)-1}'),
            sky_new = sky_mdl,
            offsets = True,
            opt_dct = cfg.sections['base_trans'],
            odir = f'{odir}/trans' if os.path.isdir(odir) else None,
            mask = f'{odir}/files/{it}_msk.npz',
            noise = cfg.sections[f'lh.{it}']['noise'],
            plot_dct = base_dct['base_plot'],
        )
        os.makedirs(os.path.dirname(tra_pkl), exist_ok=True)
        pickle.dump((samples, offsets), open(tra_pkl, "wb"))

    # plot the final results of the transition
    models = [sky_mdl, ] + [m for m in sky_mdl.models if len(sky_mdl.models) > 1]
    plot_mean_and_std(
        model = [md for md in models],
        samples = samples,
        mode = 'mean',
        name = f'{it}_pre.png',
        odir = f'{odir}/plots',
        **plt_dct,
    )

    # load and update the model config with the new offsets
    cfg = SetupKLConfig.from_file(mdl_yml)

    for pf,of in offsets.items():
        cfg.modify_sec(f'sky_{pf}', offset=of)

    # save the new model yaml-file
    cfg.to_file(f'{odir}/files/{it}_pre.yml')
    

if __name__ == '__main__':
    main()
