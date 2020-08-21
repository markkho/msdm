import json
import inspect

def encode_item(i):
    if isinstance(i, (dict, list)):
        i = json.dumps(i, sort_keys=True)
    return i

def decode_item(i):
    try:
        i = json.loads(i)
    except json.JSONDecodeError:
        pass
    except TypeError:
        pass
    return i

class AssignmentMap(dict):
    """Dictionary that supports simple dictionaries as keys"""
    def __init__(self, *args, **kwargs):
        dict.update(self)
        if len(args) > 0:
            for k, v in args[0]:
                self[k] = v
        if len(kwargs) > 0:
            for k, v in kwargs.items():
                self[k] = v

    def __getitem__(self, key):
        return dict.__getitem__(self, encode_item(key))
    
    def get(self, key, default=None):
        return dict.get(self, encode_item(key), default)
    
    def __setitem__(self, key, val):
        dict.__setitem__(self, encode_item(key), val)
    
    def __repr__(self):
        dictrepr = dict.__repr__(self)
        return '%s(%s)' % (type(self).__name__, dictrepr)

    def __contains__(self, i):
        return dict.__contains__(self, encode_item(i))
    
    def update(self, *E, **F):
        """Updates self in place"""
        am = AssignmentMap()
        for e in E:
            for k in e.keys():
                am[k] = e[k]
        for k in F.keys():
            am[k] = F[k]
        for k, v in am.items():
            self[k] = v

    def merge(self, *E, **F):
        """Returns a new merged assignment map"""
        am = AssignmentMap()
        am.update(self)
        am.update(*E, **F)
        return am
            
    def items(self):
        for k, v in dict.items(self):
            yield decode_item(k), v
    
    def keys(self):
        for k in dict.keys(self):
            yield decode_item(k)
    
    def __iter__(self):
        for k in dict.__iter__(self):
            yield decode_item(k)

class DefaultAssignmentMap(AssignmentMap):
    '''
    DefaultAssignmentMap extends AssignmentMap to support a default value for unset items,
    akin to collections.defaultdict. Notably, this implementation does not store default values
    after access, a departure from defaultdict.

    The first parameter to DefaultAssignmentMap is the function used to generate values
    returned when the AssignmentMap has no set value for a key.

    The defaultvalue function can either take no arguments or one argument, which corresponds
    to the key passed to AssignmentMap.__getitem__.
    '''
    def __init__(self, defaultvalue, *args, **kwargs):
        super().__init__(*args, **kwargs)

        arity = len(inspect.getfullargspec(defaultvalue).args)
        assert arity in (0, 1), 'Default value function must take either 0 or 1 arguments.'
        self.defaultvalue = defaultvalue if arity == 1 else lambda _: defaultvalue()

    def __getitem__(self, key):
        if key not in self:
            return self.defaultvalue(key)
        return super().__getitem__(key)
