from nifty8.re import CorrelatedFieldMaker, InvGammaPrior, UniformPrior, Model, VModel

from .gaussian import gaussian_model
from .integer import integer_model
from .normal import normal_model
from .space import SignalSpace, PointSpace
from .util import check_type



CFM_KEYS = {'offset_mean', 'offset_std', 'fluctuations' ,'loglogavgslope', 'flexibility', 'asperity', 'non_parametric_kind'}
NM_KEYS = {'mean', 'std'}
IGM_KEYS = {'alpha', 'scale', 'mean', 'mode'}
GSM_KEYS = {'cov_x', 'cov_y', 'scale', 'theta'}
UM_KEYS = {'u_min', 'u_max'}
IM_KEYS = {'i_min', 'i_max', 'step'}



def prior_model(
        prefix,
        space,
        n_copies = 1,
        **i0_params,
):
    '''
    Initialize one of the prior models based on the provided parameters.
    
    Parameters
    ----------
    prefix : str
        The prefix for the model.
    space : SignalSpace or PointSpace
        The space for the model.
    n_copies : int
        The number of copies for the model. Default is 1.
    i0_params : dict
        The parameters for the model (see the specific model for details)

    Returns
    -------
    model : Model
        The initialized model.
    pspec : Callable
        The power spectrum of the correlated field model. Otherwise None.
    '''
    check_type(prefix, str)
    check_type(space, (SignalSpace, PointSpace))
    check_type(n_copies, int)

    pspec = None
    match set(i0_params.keys()):
        case k if k.issubset(CFM_KEYS):
            check_type(space, SignalSpace)
            model, pspec = correlated_field_model(
                prefix=prefix,
                shape=space.shape,
                distances=space.distances,
                n_copies=n_copies,
                **i0_params
            )
        case k if k.issubset(NM_KEYS):
            model = normal_model(
                prefix=prefix,
                shape=space.shape,
                n_copies=n_copies,
                **i0_params
            )
        case k if k.issubset(IGM_KEYS):
            model = inverse_gamma_model(
                prefix=prefix,
                shape=space.shape,
                n_copies=n_copies,
                **i0_params
            )
        case k if k.issubset(GSM_KEYS):
            check_type(space, SignalSpace)
            model = gaussian_model(
                prefix=prefix,
                shape=space.shape,
                distances=space.distances,
                n_copies=n_copies,
                **i0_params
            )
        case k if k.issubset(UM_KEYS):
            model = uniform_model(
                prefix=prefix,
                shape=space.shape,
                n_copies=n_copies,
                **i0_params
            )
        case k if k.issubset(IM_KEYS):
            model = integer_model(
                prefix=prefix,
                shape=space.shape,
                n_copies=n_copies,
                **i0_params
            )
        case _:
            print(set(i0_params.keys()))
            raise ValueError('Invalid parameters for prior model')
    return model, pspec
        


def correlated_field_model(*,
        prefix,
        shape,
        distances,
        offset_mean,
        offset_std,
        fluctuations,
        loglogavgslope,
        flexibility = None,
        asperity = None,
        non_parametric_kind = 'power',
        n_copies = 1,
):
    '''
    Initialize the correlated field model of nifty8.re (correlation model).
    
    Parameters
    ----------
    prefix : str
        The prefix for the model
    shape : tuple
        The shape of the model
    distances : tuple
        The distances for the model
    offset_mean : float
        The offset mean parameter
    offset_std : tuple
        The offset standard deviation parameter (nifty8.re.LognormalPrior)
    fluctuations : tuple
        The fluctuations parameter (nifty8.re.LognormalPrior)
    loglogavgslope : float
        The log-log average slope parameter (nifty8.re.NormalPrior)
    flexibility : float, optional
        The flexibility parameter (nifty8.re.LognormalPrior). Default is None.
    asperity : float, optional
        The asperity parameter (nifty8.re.LognormalPrior). Default is None.
    non_parametric_kind : str, optional
        Either use a power or an amplitude spectrum. Default is 'power'.
    n_copies : int, optional
        The number of copies for the model. Default is 1.

    Returns
    -------
    model : Model
        The initialized model.
    power : Model
        The power spectrum of the model.
    '''
    cfm = CorrelatedFieldMaker(prefix)
    cfm.set_amplitude_total_offset(offset_mean, offset_std)
    cfm.add_fluctuations(
        shape, 
        distances, 
        fluctuations, 
        loglogavgslope, 
        flexibility, 
        asperity,
        non_parametric_kind = non_parametric_kind,
    )  
    model = cfm.finalize()
    power = Model(cfm.power_spectrum, domain=model.domain, init=model.init)

    if n_copies > 1:
        return (VModel(model, n_copies), VModel(power, n_copies))
    else:
        return (model, power)



def inverse_gamma_model(*,
        prefix,
        shape,
        mean = None,
        mode = None,
        alpha = None,
        scale = None,
        n_copies = 1,
):
    '''
    Initialize an inverse gamma distributed prior.
    
    Parameters
    ----------
    prefix : str
        The prefix for the model
    shape : tuple
        The shape of the model
    mean : float, optional
        The mean of the model. Default is None.
    mode : float, optional
        The mode of the model. Default is None.
    alpha : float, optional
        The alpha parameter of the model. Default is None.
    scale : float, optional
        The scale parameter of the model. Default is None.
    n_copies : int, optional
        The number of copies for the model. Default is 1.

    The inverse gamma distribution usually is defined by the parameters `alpha` and `scale`.
    However, one can also define it by its `mean` and `mode`, which are related to `alpha` and `scale` via:
        alpha = 2 / (mean / mode - 1) + 1
        scale = mode * (alpha + 1)
    '''
    match (mean, mode, alpha, scale):
        case (me, mo, None, None) if isinstance(me, (int, float)) and isinstance(mo, (int, float)):
            alpha = 2 / (me / mo - 1) + 1
            scale = mo * (alpha + 1)
        case (None, None, al, sc) if isinstance(al, (int, float)) and isinstance(sc, (int, float)):
            pass
        case _:
            raise ValueError('either `mean` and `mode` or `alpha` and `scale` have to be provided')
    model = InvGammaPrior(alpha, scale, shape=shape, name=prefix)

    if n_copies > 1:
        return VModel(model, n_copies)
    else:
        return model



def uniform_model(*,
        prefix,
        shape,
        u_min,
        u_max,
        n_copies = 1,
):
    '''
    Initialize a uniform distributed prior.
    
    Parameters
    ----------
    prefix : str
        The prefix for the model
    shape : tuple
        The shape of the model
    u_min : float
        The minimum value of the uniform distribution
    u_max : float
        The maximum value of the uniform distribution
    n_copies : int, optional
        The number of copies for the model. Default is 1.
    '''
    model = UniformPrior(u_min, u_max, shape=shape, name=prefix)

    if n_copies > 1:
        return VModel(model, n_copies)
    else:
        return model
