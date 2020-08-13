import json

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

    def __contains__(self, i):
        if isinstance(i, dict):
            i = json.dumps(i, sort_keys=True)     
        return dict.__contains__(self, i)
    
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