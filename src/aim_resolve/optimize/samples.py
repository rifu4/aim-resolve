from collections.abc import Iterable
from jax import random
from jax.typing import ArrayLike
from nifty8.re import Initializer, Model, Samples, VModel, Vector, mean, mean_and_std, logger
from typing import Union



class MySamples(Samples):
    '''Extension of nifty8.re.Samples to handle components and utility lh_functions for mean and std'''

    def mean(self, model = lambda x: x):
        '''
        Calculate the mean of the samples using the model. Returns the `samples.pos` for MAP estimates.
        
        Parameters
        ----------
        model : callable
            Function to apply to the samples. Default is identity function.
        '''
        if len(self) == 0:
            return model(self.pos)
        else:
            return mean(tuple(model(s) for s in self))
        
    def mean_and_std(self, model = lambda x: x):
        '''
        Calculate the mean and standard deviation of the samples using the model.
        Returns `None` for the standard deviation if there are less than 2 samples.

        Parameters
        ----------
        model : callable
            Function to apply to the samples. Default is identity function.
        '''
        if len(self) < 2:
            return self.mean(model), 0
        else:
            return mean_and_std(tuple(model(s) for s in self)) 
        


def get_samples(key, samples, position_or_samples, lh_dict, transition=None, it=None) -> MySamples:
    '''
    Get the samples for the `optimize_kl` function and check if they are compatible with the sky model.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key if new samples need to be drawn.
    samples : MySamples
        Samples usually loaded from an old reconstruction in `optimize_kl`.
    position_or_samples : MySamples
        New sampels or position vector used as a starting point for `optimize_kl`.
    lh_dict : dict
        Dictionary containing the likelihood model and noise parameters.
    transition : callable, optional
        Transition function to update the samples to match a new sky model. Default is None.
    it : int, optional
        Current iteration number of the optimization. Default is None.

    Returns
    -------
    key : jax.random.PRNGKey
        Pass random key for the optimization.
    samples : MySamples
        Samples that shall be updated during the optimization.
    '''
    models = [v for v in lh_dict.values() if isinstance(v, Model)]
    if domain_keys(models) == set():
        raise ValueError('Check that sky and noise models in the `lh_dict` are of type `nifty8.re.Model`')

    match (domain_keys(samples), domain_keys(position_or_samples), domain_keys(models)):
    
        # if no samples are provided, draw new samples
        case (s, p, m) if s == p == set():
            print('s == p == set()')
            key, k_p = random.split(key)
            samples = random_init(k_p, models, factor=0.01)

        # if new sampels or position vector is provided, use them as starting point
        case (s, p, m) if s != m == p:
            print('s != m = p')
            if s != set():
                logger.warning('overwriting `samples` with `position_or_samples`')
            samples = position_or_samples
        
        # if samples and model have different domains, check if a transition function is provided and use it to update the samples
        case (s, p, m) if s != m:
            print('s != m')
            match transition:
                case tr if tr:
                    logger.warning('\n---\nperforming transition to update the model params')
                    key, k_t, k_p = random.split(key, 3)
                    samples = tr(k_t, samples, it)
                    logger.warning('finished transition\n---\n')
                    if domain_keys(samples) != m:
                        raise ValueError('`samples` and `likelihood` still have different domains. Check the transition function')
                case _:
                    print('s:', s)
                    print('m:', m)
                    raise ValueError('`samples` and `likelihood` have different domains and no transition is specified')

    # Turn the samples into a MySamples object
    match samples:
        case Vector():
            samples = MySamples(pos=samples, samples=None, keys=None)
        case Samples():
            samples = MySamples(pos=samples._pos, samples=samples._samples, keys=samples._keys)
    return key, samples



def domain_tree(model: Union[Model, Samples, Vector, dict, Iterable[Model, Samples, Vector, dict]], error=True) -> dict:
    '''
    Get the parameter tree of a `nifty.re` model or an iterable of those.
    
    Parameters
    ----------
    model : Model, Samples, Vector, dict, iterable
        Model, vector, dict or samples object or a iterable of those.
    error : bool, optional
        If True, raise an error if the model is not of the expected type. Otherwise return an empty dict. Default is True.
    '''
    match model:
        case None | False: 
            return {}
        case Model() | VModel(): 
            return domain_tree(model.domain, error)
        case Samples(): 
            return domain_tree(model.pos, error)
        case Vector(): 
            return domain_tree(model.tree, error)
        case dict(): 
            return model
        case Iterable() if not isinstance(model, ArrayLike):
            tree = {}
            for md in model:
                tree |= domain_tree(md, error)
            return tree
        case _:
            if error:
                print('type:', type(model))
                raise ValueError('`model` has to be an instance of `Model`, `Samples`, `Vector`, `dict` or an iterable of those')
            return {}



def domain_keys(model: Union[Model, Samples, Vector, dict], error=True) -> set[str]:
    '''Get the keys of the parameter tree (see `domain_tree` function).'''
    return set(domain_tree(model, error).keys())



def model_init(model: Union[Model, Iterable[Model]], error=True) -> Initializer:
    '''
    Get the initializer of a `nifty.re` model or a iterable of those.
    
    Parameters
    ----------
    model : Model, iterable
        Model or iterable of models.
    error : bool, optional
        If True, raise an error if the model is not of the expected type. Otherwise return an empty Initializer. Default is True.
    '''
    match model:
        case None | False: 
            return Initializer({})
        case Model() | VModel(): 
            return model.init
        case Iterable() if not isinstance(model, ArrayLike): 
            init = Initializer({})
            for md in model:
                init |= model_init(md, error)
            return init
        case _:
            if error:
                print('type:', type(model))
                raise ValueError('`model` has to be an instance of `Model` or an iterable of those')
            return Initializer({})



def random_init(key, model: Union[Iterable[Model], Model], pos: Union[Samples, Vector, dict] = {}, factor=0.01) -> Vector:
    '''
    Randomly initialize a model using the jax random key. Provide a position vector if some parameters are set already.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key to use for the initialization.
    model : Model, iterable
        Model or iterable of models to be initialized.
    pos : Samples, Vector, dict, optional
        Position vector or dictionary to set some parameters by hand. Default is {}.
    factor : float, optional
        Factor to scale the random initialization. Default is 0.01.
    '''
    mdl_tree = domain_tree(model)
    pos_tree = {k:v for k,v in domain_tree(pos).items() if k in mdl_tree}
    mdl_init = model_init(model)
    pos_init = 0.01 * Vector(mdl_init(key))
    return Vector(domain_tree(pos_init) | pos_tree)
