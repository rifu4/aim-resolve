from .util import clean_dict, merge_dicts, split_its, update_it, pop_val
from .yml import yaml_load, yaml_safe



class SetupKLConfig:
    '''Class to update a yaml configuration file for the OptimizeKLConfig class with additional iterations and sections.'''

    def __init__(self, sections):
        '''
        Initialize the SetupKLConfig class.

        Parameters
        ----------
        sections : dict
            Configuration sections.
        '''
        self.sections = dict(sections)
        self.get_it()


    @classmethod
    def from_file(cls, fname):
        '''
        Load a configuration file and create a SetupKLConfig object.

        Parameters
        ----------
        fname : str or list of str
            File name(s) of the config file that is imported.
        '''
        sections = yaml_load(fname)

        return cls(sections)
    

    def to_file(self, fname):
        '''Write configuration in standardized form to a file.

        Parameters
        ----------
        fname : str
            Path to which the config shall be written.
        '''
        dct = clean_dict(self.sections)
        dct_lst = split_its(dct)

        yaml_safe(dct_lst, fname)


    def get_it(self):
        '''Get the current iteration number'''
        keys = sorted([k for k in filter(lambda x: 'lh' in x, self.sections)])
        self.it = int(keys[-1].split('.')[1])


    def add_it(self, fix_keys=[], del_comp=True, it=None):
        '''
        Add a new iteration to the configuration file.
        
        Parameters
        ----------
        fix_keys : list of str
            Keys that are not updated to the new iteration.
        del_comp : bool
            If True, all components are deleted from the sky model of the next iteration.
        it : int
            Current iteration number after which the new iteration is added.
        '''
        dct = self.sections
        it = int(it) if it != None else self.it
        _it = f'.{it}'

        fix_keys = set(k.split('.')[0] if '.' in k else k for k in list(fix_keys))
        fix_keys |= {'trans'}

        upd_keys = [k.split('.')[0] if '.' in k else k for k in dct]
        upd_keys = [u for i,u in enumerate(upd_keys) if u not in upd_keys[:i]]
        upd_keys = [k for k in upd_keys if k not in fix_keys and any(x in k for x in ['opt', 'lh', 'data', 'sky'])]

        n_dct = {}
        for key in upd_keys:
            k = sorted([k for k in dct if key+'.' in k])[-1]
            n_dct[key+_it] = dct[k]
        
        n_dct = update_it(n_dct, it, fix_keys)
        self.it += 1
        _it = f'.{self.it}'

        # remove components from the next iteration sky model
        if del_comp:
            n_sky = n_dct['sky'+_it]
            del_keys = [k for k in n_sky if k not in ('prefix', 'background')]
            for d in del_keys:
                n_sky.pop(d)

        dct |= n_dct
        self.sections = clean_dict(self.sections)


    def add_trans(self, it=None, mode='addt', **kwargs):
        '''
        Add a transition section to the configuration file.
        
        Parameters
        ----------
        it : int
            Iteration number to which the transition is added.
        mode : str
            Transition mode. Default is `addt`.
        kwargs : dict
            Parameters for the transition function.
        '''
        dct = self.sections
        it = int(it) if it != None else self.it
        _it = f'.{it}'

        lh_lst = sorted([int(k.split('.')[-1]) for k in self.sections if 'lh' in k])
        sky_lst = sorted([int(k.split('.')[-1]) for k in self.sections if 'sky' in k and not '_' in k])
        ll, sl = [], []
        for i in range(it+1):
            ll.append(i if i in lh_lst else ll[-1])
            sl.append(i if i in sky_lst else sl[-1])

        dct['trans'+_it] = {
            'lh_old': f'=lh.{ll[it-1]}',
            'lh_new': f'=lh.{ll[it]}',
            'sky_old': f'=sky.{sl[it-1]}',
            'sky_new': f'=sky.{sl[it]}',
            'mode': mode,
            **kwargs,
        }
        dct['opt'+_it]['transitions'] = '=trans'+_it


    def add_sec(self, sec_key, **kwargs):
        '''
        Add a new section to the configuration file.
        
        Parameters
        ----------
        key : str
            Key of the new section.
        kwargs : dict
            Parameters for the new section.
        '''
        self.modify_sec(sec_key, **kwargs)


    def modify_sec(self, sec_key, merge_base=True, **kwargs):
        '''
        Modify a section in the configuration file.

        Parameters
        ----------
        key : str
            Key of the section that is modified.
        base : bool
            If False, keys containing `base` are not merged.
        kwargs : dict
            Parameters that shall be changed in or added to the section.
        '''
        dct = self.sections

        if sec_key not in dct:
            dct[sec_key] = {}

        dct[sec_key] = merge_dicts([dct[sec_key], kwargs], merge_base)

    
    def remove_sec(self, sec_key):
        '''
        Remove a section from the configuration file.

        Parameters
        ----------
        key : str
            Key of the section that is removed.
        '''
        dct = self.sections
        dct.pop(sec_key)
        dct = pop_val(dct, sec_key)
        dct = pop_val(dct, f'={sec_key}')
        self.sections = dct
