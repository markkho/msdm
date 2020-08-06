import json

class AssignmentMap(dict):
    """Dictionary that supports simple dictionaries as keys"""
    def __init__(self, *args, **kwargs):
        dict.update(self, *args, **kwargs)
    
    def __getitem__(self, key):
        if isinstance(key, dict):
            key = json.dumps(key, sort_keys=True)
        return dict.__getitem__(self, key)
    
    def get(self, key, default=None):
        if isinstance(key, dict):
            key = json.dumps(key, sort_keys=True)    
        return dict.get(self, key, default)
    
    def __setitem__(self, key, val):
        if isinstance(key, dict):
            key = json.dumps(key, sort_keys=True)
        dict.__setitem__(self, key, val)
    
    def __repr__(self):
        dictrepr = dict.__repr__(self)
        return '%s(%s)' % (type(self).__name__, dictrepr)
    
    def update(self, *args, **kwargs):
        for k, v in AssignmentMap(*args, **kwargs).items():
            self[k] = v
            
    def items(self):
        for k, v in dict.items(self):
            try:
                k = json.loads(k)
            except json.JSONDecodeError:
                pass
            yield k, v
    
    def keys(self):
        for k in dict.keys(self):
            try:
                k = json.loads(k)
            except json.JSONDecodeError:
                pass        
            yield k
    
    def __iter__(self):
        for k in dict.__iter__(self):
            try:
                k = json.loads(k)
            except json.JSONDecodeError:
                pass        
            yield k