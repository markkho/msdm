import json
from collections import OrderedDict 

def AssignmentCache(func):
    cache = OrderedDict()
    def memoized(*args, **kwargs):
        key = []
        for arg in args:
            if isinstance(arg, dict):
                arg = json.dumps(arg, sort_keys=True)
            key.append(arg)
        key = (tuple(key), json.dumps(kwargs, sort_keys=True))
        try:
            v = cache[key]
            return v
        except KeyError:
            pass
        v = func(*args, **kwargs)
        cache[key] = v
        return v
    return memoized
