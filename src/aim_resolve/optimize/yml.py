import os
import yaml



def yaml_load(fname):
    '''Load one or multiple yaml stream(s) from one or multiple file(s) to a single python dict
    
    Parameters
    ----------
    fname : str or list of str
        File name(s) of the yaml file that is imported.

    Returns
    -------
    dct : dict
        Python dict that contains all the yaml streams from the file
    '''
    if isinstance(fname, str):
        fname = [fname]

    dct = {}
    for fn in fname:
        if not os.path.isfile(fn):
            raise RuntimeError(f'`{fn}` not found')
        with open(fn, 'r') as f:
            yml_list = list(yaml.safe_load_all(f))
            for ll in yml_list:
                if ll != None:
                    dct |= get_vals(ll)
    return dct



def yaml_safe(dct, fname):
    '''Save a python dict as a single yaml stream or a list of dicts as separate yaml streams in a single file
    
    Parameters
    ----------
    dct: dict
        Python dict or list of dicts that shall be saved
    fname : str
        Path to which the yaml file shall be written
    '''
    dct_lst = dct if isinstance(dct, list) else [dct, ]
    if not all(isinstance(di, dict) for di in dct_lst):
        raise TypeError
    dumper = MyDumper
    dumper.add_representer(list, flow_list_rep)

    with open(fname, 'w') as f:
        yaml.dump_all(dct_lst, f, Dumper=MyDumper, sort_keys=False)



def get_vals(dct):
    '''Recursively replace spaces in the keys of a python dict with underscores'''
    new_dct = {}
    for key,val in dct.items():
        if isinstance(val, dict):
            val = get_vals(val)
        new_dct[key.replace(' ', '_')] = val
    return new_dct



class MyDumper(yaml.SafeDumper):
    '''Special yaml Dumper class that inserts extra line breaks between the first order keys of a dict'''
    def write_line_break(self, data=None):
        super().write_line_break(data)

        if len(self.indents) <= 1:
            super().write_line_break()



def flow_list_rep(dumper, data):
    '''Function to represent python lists in flow style when saving them to a yaml file'''
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
