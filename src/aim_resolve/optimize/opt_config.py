import os
import numpy as np
from copy import deepcopy

from .fast_opt_kl import fast_optimize_kl
from .opt_kl import optimize_kl
from .samples import domain_keys
from .util import clean_dict, merge_dicts, split_its, add_dicts, clean_reps, get_it, is_or_contains_type, extend_reps, eval_string, eval_list
from .yml import yaml_load, yaml_safe



class OptimizeKLConfig:
    '''Class to initialize a nifty optimization from a single or multiple yaml configuration files.'''

    def __init__(self, sections, builders, mode='total'):
        '''
        Initialize the OptimizeKLConfig class.

        Parameters
        ----------
        sections : dict
            Configuration sections.
        builders : dict
            Dictionary of builder functions. 
        '''
        match mode:
            case 'total':
                self.total = True
            case 'major' | 'minor':
                self.total = False
            case _:
                raise ValueError('mode has to be either `total`, `major` | `minor`')
        n_dyn = ['likelihood', 'n_samples', 'draw_linear_kwargs', 'nonlinearly_update_kwargs', 'kl_kwargs', 'sample_mode']
        self.optkeys = dict(
            n_iter = ['n_total_iterations'] if self.total else ['n_major_iterations'],
            static = ['odir', 'position_or_samples', 'key', 'resume'],
            needed_dyn = n_dyn if self.total else n_dyn + ['n_minor_iterations'],
            option_dyn = ['constants', 'point_estimates', 'transitions'],
        )
        self.sections = dict(sections)
        self.interpret_base()
        self.interpret_link()
        self.interpret_reps()
        self.join_opt_stages()
        self.builders = builders(self.sections) if callable(builders) else dict(builders)


    @classmethod
    def from_file(cls, fname, builders, mode='total'):
        '''
        Import a config file and instantiate the class.

        Parameters
        ----------
        fname : str
            File name of the config file that is imported.
        builders : dict
            Dictionary of functions that are used to instantiate e.g. operators.
        '''
        sections = yaml_load(fname)

        return cls(sections, builders, mode)


    def to_file(self, fname):
        '''
        Write configuration in standardized form to file.

        Parameters
        ----------
        fname : str
            Path to which the config shall be written.
        '''
        dct = clean_dict(self.sections)
        dct['opt.0'] = clean_reps(dct['opt.0'], simplify=True)
        dct_lst = split_its(dct)

        yaml_safe(dct_lst, fname)


    def optimize_kl(self, **kwargs):
        '''
        Do the inference and save the config file to the output directory.

        Parameters
        ----------
        kwargs : dict
            Additional parameters for the `optimize_kl` function (e.g. callback).
        '''
        dct = dict(self)

        os.makedirs(dct['odir'], exist_ok=True)
        self.to_file(os.path.join(dct['odir'], 'opt.yml'))

        if self.total:
            return optimize_kl(**dct, **kwargs)
        else:
            return fast_optimize_kl(**dct, **kwargs)
    

    def interpret_base(self):
        '''Replace the `base` entries in all (sub)sections by the content of the section it points to.'''
        dct = self.sections

        for sec in dct:
            dct[sec] = get_base(dct[sec], dct)


    def interpret_link(self):
        '''Replace the `->` entries in all (sub)sections by the content (string) of the section key it points to.'''
        dct = self.sections
        
        for sec in dct:
            dct[sec] = get_link(dct[sec], dct)


    def interpret_reps(self):
        '''Expand the repetitions of all sections starting with `opt.`. Check if all necessary keys are present.'''
        dct = self.sections

        for optsec in filter(lambda x: x[:4] == 'opt.', dct.keys()):
            for key in self.optkeys['n_iter'] + self.optkeys['needed_dyn']:
                if key not in dct[optsec]:
                    raise KeyError(f'key `{key}` is missing in opt section `{optsec}`')
            for key in self.optkeys['option_dyn']:
                if key not in dct[optsec]:
                    dct[optsec][key] = None

            if self.total:
                [dct[optsec].pop(key) for key in ['n_major_iterations', 'n_minor_iterations'] if key in dct[optsec]]
                dct[optsec] = get_reps(dct[optsec], dct[optsec]['n_total_iterations'])
            else:
                [dct[optsec].pop(key) for key in ['n_total_iterations'] if key in dct[optsec]]
                n_major = dct[optsec]['n_major_iterations']
                minor_key = 'n_minor_iterations'
                n_minor = get_reps({minor_key: dct[optsec][minor_key]}, n_major)[minor_key]
                dct[optsec] = get_reps(dct[optsec], n_major, n_minor)


    def join_opt_stages(self):
        '''
        Join the repetitions for all sections starting with `opt.` to a single section called `opt.0`.

        Sort the sections in ascending order, add their leaves and clean up the `opt.` section.
        Remove the old `opt.` sections.
        '''
        dct = self.sections

        opt_keys = sorted(
            (k for k in dct.keys() if k.startswith('opt.')),
            key=lambda k: int(k.split('.')[1])
        )
        dct['opt.0'] = add_dicts(*[dct[k] for k in opt_keys])
        dct['opt.0'] = clean_reps(dct['opt.0'], simplify=False)

        for k in filter(lambda k: k != 'opt.0', opt_keys):
            del dct[k]
    

    def make_callable(self, sec, key=None):
        '''
        Turn the section repetition lists into callable functions of the iteration number.
        Instantiate all references indicated by `=` using the builders dictionary.
        '''
        def fun(it):
            val = get_it(sec, it)
            if key in ['constants', 'point_estimates']:
                val = self.get_constants_or_point_estimates(val, it)
            elif isinstance(val, str):
                if len(val) > 1 and val.startswith('='):  # is reference
                    val = self.instantiate_sec(val[1:])
            return val
        
        if is_or_contains_type(sec, list):
            return fun
        else:
            return fun(0)


    def instantiate_sec(self, sec):
        '''
        Instantiate an object that is described by a section in the config file by looking up 
        the section key in the `self._builders` dictionary and call the respective function.
        '''
        dct = deepcopy(self.sections[sec])

        # Instantiate all references (also in subsections)
        for key,val in dct.items():
            if isinstance(val, str):
                if len(val) > 1 and val[0] == '=':  # is reference
                    dct[key] = self.instantiate_sec(val[1:])

        # Plug into builders dictionary
        if sec in self.builders:
            return self.builders[sec](**dct)
        raise RuntimeError(f'Provide build routine for `{sec}` in builders dictionary')


    def get_constants_or_point_estimates(self, cpe, it):
        '''
        Get both the constants and point estimates for the current iteration. Given a model section name,
        it adds all parameter keys of that model component. For a `~` in front of the name, it includes
        all likelihood parameter keys except the ones of the model component.
        '''
        match cpe:
            case None | [None,] | (None,) | [] | ():
                return None
            case str():
                cpe = [cpe]
            case tuple():
                cpe = list(cpe)
        
        match (all('~' in c for c in cpe), any('~' in c for c in cpe)):
            case (True, _):
                neg = True
            case (False, False):
                neg = False
            case (False, True):
                raise ValueError(f'Negation `~` has to be used for all or none of the constants/point_estimates')

        if not self.total:
            minor_cs = np.cumsum(self.sections['opt.0']['n_minor_iterations'])
            it = np.searchsorted(minor_cs, it, side='right')

        lh_sec = get_it(self.sections['opt.0']['likelihood'], it)
        m_keys = domain_keys(self.instantiate_sec(lh_sec[1:])['model'])

        cpe_new = set()
        for c in cpe:
            match c.replace('=', '').strip('~'):
                case s if s in m_keys:
                    cpe_new.add(s)
                case s if s in self.sections:
                    c_keys = domain_keys(self.instantiate_sec(s))
                    cpe_new.update(k for k in c_keys if k in m_keys)
                case _:
                    raise ValueError(f'Cannot find `{c}` in sections or `{m_keys}`.')
        
        if neg:
            return tuple(m_keys - cpe_new)
        return tuple(cpe_new)


    def __iter__(self):
        '''Enable conversion to `dict` to pass everyting to the `optimize_kl` function.'''
        # static
        sopt = self.sections['opt']
        for key in self.optkeys['static']:
            if key in sopt:
                yield key, sopt[key]

        # dynamic
        sdyn = self.sections['opt.0']
        for key in self.optkeys['n_iter']:
            if key in sdyn:
                yield key, sdyn[key]
        for key in self.optkeys['needed_dyn'] + self.optkeys['option_dyn']:
            if key in sdyn:
                yield key, self.make_callable(sdyn[key], key)

        
    def __str__(self):
        s = []
        for key, val in self.sections.items():
            s += [key]
            s += [f'  {kk}: {vv}' for kk, vv in val.items()]
            s += ['']
        return '\n'.join(s)
    

    def __eq__(self, other):
        for a in 'sections', 'builders':
            if getattr(self, a) != getattr(other, a):
                return False
        return True



def get_base(sub, dct, key_lst=[]):
    '''Recursively replace the `base` entries in all (sub)sections by the content of the section it points to.'''
    for (key,val) in sub.items():
        if len(key_lst) != len(set(key_lst)):
            raise RuntimeError(f'You are trying a base-loop. Please do not do that :(')
        
        if isinstance(val, dict):
            sub[key] = merge_dicts([sub[key], get_base(val, dct, key_lst+[key])])
        
        elif key.startswith('base'):
            sec = dct.copy()
            sec_keys = []
            while '/' in val:
                pre, val = val.split('/', 2)
                if pre not in sec:
                    raise RuntimeError(f'the referred section `{pre}` does not exist in `{sec}`')
                sec = sec[pre]
                sec_keys += [pre]
            if val not in sec:
                raise RuntimeError(f'the referred section `{val}` does not exist in `{sec}')
            sub = merge_dicts([get_base(sec[val], dct, key_lst+sec_keys+[val]), sub])
        
    return sub



def get_link(sub, dct):
    '''Recursively replace the `->` entries in all (sub)sections by the content (string) of the section key it points to.'''
    for (key,val) in sub.items():
        if isinstance(val, dict):
            sub[key] = merge_dicts([sub[key], get_link(val, dct)])
        
        elif isinstance(val, str) and '->' in val:
            oldval = map(lambda x: x.strip(), val.split('+'))
            newval = ''
            for ov in oldval:
                if ov.startswith('->'):
                    ov = ov[2:].strip()
                    ov_dct = deepcopy(dct)
                    while '/' in ov:
                        pre, ov = ov.split('/', 2)
                        if pre not in ov_dct:
                            raise RuntimeError(f'the referred section `{pre}` does not exist in `{ov_dct}`')
                        ov_dct = ov_dct[pre]
                    ov = ov_dct[ov]
                    if not isinstance(ov, str):
                        raise ValueError(f'the referred section value `{ov}` has to be a string.')
                    elif '->' in ov:
                        raise ValueError('recursive links not allowed for now')
                newval = os.path.join(newval, ov.strip('/'))
            sub[key] = newval
    return sub



def get_reps(sub, total_it, minor_it=None):
    '''Recursively expand the repetitions of all sections starting with `opt.`.'''
    for (key,val) in sub.items():
        if isinstance(val, dict):
            sub[key] = get_reps(val, total_it, minor_it)
            continue
            
        elif key in ['n_total_iterations', 'n_major_iterations']:
            if not isinstance(val, int) or val < 1:
                raise TypeError(f'`{key}` has to be of type `int` and larger than 0')
            sub[key] = val
            continue

        elif isinstance(val, str):
            val = eval_string(val)

        if not isinstance(val, list) or val == []:
            val = [val]

        if isinstance(val, list):
            val = eval_list(val)
            if key in ['constants', 'point_estimates', 'transitions']:
                val = extend_reps(val, total_it, None)
            else:
                val = extend_reps(val, total_it)

            if minor_it and key not in ['n_minor_iterations', 'likelihood', 'transitions']:
                for i,mi in enumerate(minor_it):
                    vi = val[i]
                    if not isinstance(vi, list) or vi == []:
                        vi = [vi]
                    val[i] = extend_reps(vi, mi)
    
            sub[key] = val

    return sub
