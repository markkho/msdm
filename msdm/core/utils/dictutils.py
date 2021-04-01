import inspect
from copy import deepcopy
import collections
from functools import reduce
from itertools import combinations, product

class defaultdict2(dict):
    '''
    defaultdict2 extends dict to support a default value for unset items,
    akin to collections.defaultdict. Notably, this implementation does not store default values
    after access, a departure from defaultdict.

    The first parameter to defaultdict2 is the function used to generate values
    returned when the dict has no set value for a key.

    The defaultvalue function can either take no arguments or one argument, which corresponds
    to the key passed to dict.__getitem__.
    '''
    def __init__(self, default_value, initialize_defaults=False):
        arity = len(inspect.getfullargspec(default_value).args)
        assert arity in (0, 1), 'Default must take either 0 or 1 arguments.'
        self.defaultvalue = default_value if arity == 1 else lambda _: default_value()
        self.initialize_defaults = initialize_defaults

    def __getitem__(self, key):
        if key not in self:
            if self.initialize_defaults:
                self[key] = self.defaultvalue(key)
                return self[key]
            return self.defaultvalue(key)
        return super().__getitem__(key)

def dict_merge(dct, merge_dct, res=None):
    """ 
    Recursively merge dictionaries
    """
    if res is None:
        res = deepcopy(dct)
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.abc.Mapping)):
            dict_merge(dct[k], merge_dct[k], res[k])
        else:
            res[k] = deepcopy(merge_dct[k])
    return res

def dict_match(left, right, res=None):
    """Do two nested dictionaries match on nested values?"""
    if res is None:
        res = [True, ]
    if res[0] == False:
        return False
    for k in right.keys():
        if (k in left):
            if (isinstance(left[k], dict) and isinstance(right[k], dict)):
                dict_match(left[k], right[k], res=res)
            else:
                res[0] = res[0] and left[k] == right[k]
        if res[0] == False:
            break
    return res[0]

def natural_join(*Rs):
    """
    Naively joins lists of nested dictionaries by joining on 
    shared nested keys. Formally, for two lists, this is 
    
    R x S = {r U s | r in R and s in S and Func(r U s)}
    
    where r and s are sets of (nested) key, value pairs. Func(t)
    is true iff the relation t is a function -- ie each key maps to
    a single value. If R and S do not share any keys, then this ends up
    being the Cartesian product. 
    (see https://en.wikipedia.org/wiki/Relational_algebra#Natural_join)
    """
    for rs in product(*Rs):
        #need to test all combintions of table rows to see if they conflict
        if all([dict_match(r, s) for r, s in combinations(rs, 2)]):
            yield reduce(dict_merge, rs, {})


