import ast
import re
from collections.abc import Iterable
from copy import deepcopy



def clean_dict(dct):
    '''Clean up a dictionary by removing all 1. order keys that are not referenced in any value of the dictionary.'''

    def _need_key(key, dct):
        '''Check if a key is referenced in any value of the dictionary.'''
        if 'opt' in key and not 'base' in key:
            return True
        for k,v in dct.items():               
            if isinstance(v, dict):
                if _need_key(key, v):
                    return True
            elif isinstance(v, str) and key in v:
                return True
            elif isinstance(v, list) and any(key in vi for vi in v if isinstance(vi,str)):
                return True
        return False

    n_dct = deepcopy(dct)
    for key in dct:
        if not _need_key(key, n_dct):
            del n_dct[key]

    return n_dct



def merge_dicts(dcts, merge_base=False):
    '''Merge multiple dictionaries and subdictionaries into one dictionary.'''
    n_dct = {}
    for d in dcts:
        for k,v in d.items():
            if k in n_dct and all(isinstance(x, dict) for x in [n_dct[k], v]):
                n_dct[k] = merge_dicts([n_dct[k], v], merge_base)
            elif k[:4] == 'base' and not merge_base:
                continue
            else:
                n_dct[k] = v
    return n_dct



def split_its(dct):
    '''Split a dictionary into a list of dictionaries based on the iteration number.'''
    n_dct = {}
    for key,val in dct.items():
        if '.' in key:
            _, ki = key.split('.')
            it_key = 'it.' + str(int(ki))
        elif 'base' in key:
            it_key = 'm'
        elif 'opt' in key:
            it_key = 'a'
        else:
            it_key = 'z'
        if not it_key in n_dct:
            n_dct[it_key] = {}
        n_dct[it_key] |= {key: deepcopy(val)}
    return [v for k,v in sorted(n_dct.items())]



def update_it(dct, it, fix_keys=[]):
    '''Update the iteration number in the keys of a dictionary for not fixed keys (new_it = old_it + 1).'''
    n_dct = {}
    for key,val in dct.items():
        if isinstance(val, dict):
            val = update_it(val, it, fix_keys)
        n_key = key
        try:
            pre, suf = n_key.split('.')
            if int(suf) <= it:
                n_key = pre + '.' + str(it + 1)
        except:
            pass
        n_val = val
        try:
            pre, suf = n_val.split('.')
            if not any(fk in n_val for fk in fix_keys):
                if int(suf) <= it:
                    n_val = pre + '.' + str(it + 1)
        except:
            pass
        n_dct[n_key] = n_val

    return n_dct



def has_key(dct, key):
    '''Check if a key is in a dictionary or any subdictionary.'''
    if key in dct.keys():
        return True
    for v in dct.values():
        if isinstance(v, dict):
            return has_key(v, key)
    return False



def has_val(dct, val):
    '''Check if a value is in a dictionary or any subdictionary.'''
    if val in dct.values():
        return True
    for v in dct.values():
        if isinstance(v, dict):
            return has_val(v, val)
    return False



def pop_key(dct, key):
    '''Remove a key from a dictionary and all subdictionaries.'''
    new_dct={}
    for k,v in dct.items():     
        if k != key:
            new_dct[k] = v      
            if isinstance(v, dict):
                new_dct[k] = pop_key(v, key)
    return new_dct



def pop_val(dct, val):
    '''Remove a value from a dictionary and all subdictionaries.'''
    new_dct={}
    for k,v in dct.items():     
        if v != val:
            new_dct[k] = v      
            if isinstance(v, dict):
                new_dct[k] = pop_val(v, val)
    return new_dct



def add_dicts(*dicts):
    '''Add multiple dictionaries and their subdictionaries.'''
    n_dct = {}
    
    def add_1dct(dct):
        for (key,val) in dct.items():
            if isinstance(val, dict):
                n_dct[key] = add_dicts(n_dct.get(key, {}), val)
            else:
                n_dct[key] = n_dct.get(key, [] if isinstance(val, list) else 0) + val

    for d in dicts:
        add_1dct(d)
    
    return n_dct



def is_or_contains_type(dct, typ):
    '''Check if a dictionary or any subdictionary contains a certain type.'''
    if isinstance(dct, typ):
        return True
    elif isinstance(dct, dict):
        for val in dct.values():
            if isinstance(val, list):
                return True
            elif isinstance(val, dict):
                if is_or_contains_type(val, typ):
                    return True
    return False



def get_it(dct, it):
    '''Get the value of a dictionary at a certain iteration number.'''

    def _flatten_list(lst):
        lst = [_flatten_list(val) if isinstance(val, list) else val for val in lst]
        return [v for val in lst for v in (val if isinstance(val, list) else [val])]

    dct_it = {}
    match dct:
        case dict():
            for (key,val) in dct.items():
                dct_it[key] = get_it(val, it)       
        case list():
            dct_it = _flatten_list(dct)[it]
        case _:
            dct_it = dct

    return dct_it



def extend_reps(val, total_it, add_val=-1):
    '''Add repeating values to a list to reach a total iteration number.'''
    val = list(val) if isinstance(val, Iterable) else [val, ]
    add_val = val[-1] if add_val == -1 else add_val
    dif = total_it - len(val)
    if dif < 0:
        val = val[:total_it]
    else:
        val += dif * [add_val]
    return val



def clean_reps(dct, simplify=True):
    '''Clean up repeating elements in a dictionary containing lists.'''

    def _simplify_list(lst):
        expr = re.sub(r"'([^']*)'", r"\1", str(lst))
        if not any(isinstance(li, list) for li in lst):
            val = []
            for i,li in enumerate(lst + ['']):
                if i == 0:
                    f = 1
                elif li == l0 and type(li) == type(l0):
                    f += 1
                elif li != l0 or type(li) != type(l0):
                    val += [f'{f}*[{l0}]'] if f>1 else [f'[{l0}]']
                    f = 1
                l0 = li
            val = ' + '.join(val)
            val = re.sub(r"'([^']*)'", r"\1", str(val))
            if len(val) < len(expr):
                return val
        return expr

    def _clean_list(lst):
        lst = [_clean_list(li) if isinstance(li,list) else li for li in lst]
        if not any(isinstance(li, list) for li in lst) and len(set(lst)) == 1:
            return lst[0]
        elif simplify:
            return _simplify_list(lst)
        return lst

    for (key,val) in dct.items():
        match val:
            case dict():
                dct[key] = clean_reps(val, simplify)
            case list():
                newval = _clean_list(val)
                match newval:
                    case str() as nv if nv.startswith('[') and {'+', '*'} & set(nv):
                        newval = '1*' + newval
                    case str() as nv if {'[', ',', ']'} & set(nv):
                        newval = val
                if not isinstance(newval, list):
                    dct[key] = newval

    return dct



def eval_string(expr):
    '''Evaluate a string expression and return the result.'''
    expr = expr.replace(' ', '')
    expr = re.sub(r'(?<!\d)([a-zA-Z=_~][a-zA-Z0-9.=_]*)(?!\d)', r'"\1"', expr)
    expr = re.sub(r'"None"|"null"|"~"', "None", expr)
    
    def _eval(node):
        match node:
            case ast.List(elts=elts): 
                return [_eval(el) for el in elts]
            case ast.Tuple(elts=elts): 
                return tuple(_eval(el) for el in elts)
            case ast.BinOp(left=left, op=ast.Add(), right=right):
                left, right = _eval(left), _eval(right)
                if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
                    return left + right
                raise ValueError("Invalid addition")
            case ast.BinOp(left=left, op=ast.Mult(), right=right):
                left, right = _eval(left), _eval(right)
                if isinstance(left, (list, tuple)) and isinstance(right, int):
                    return left * right
                if isinstance(left, int) and isinstance(right, (list, tuple)):
                    return right * left
                raise ValueError("Invalid multiplication")
            case ast.Constant(value=value): 
                return value
            case _: 
                raise ValueError("Unsupported expression")

    return _eval(ast.parse(expr, mode='eval').body)


def eval_list(expr):
    '''Evaluate string expressions in a list and return the result.'''
    if isinstance(expr, list):
        return [eval_list(el) for el in expr]
    elif isinstance(expr, str):
        return eval_string(expr)
    return expr



def check_dict(dct, needed, optional=[]):
    '''Check if all needed keys are in the dictionary and remove wrong keys.'''
    #TODO: adjust function -> similar to check_type
    allowed = set(needed) | set(optional)
    if dct:
        dct = {key: val for key, val in dct.items() if key in allowed}
        for key in needed:
            if key not in dct:
                raise ValueError(f'key `{key}` is missing in dictionary')
    return dct
