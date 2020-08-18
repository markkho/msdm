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
    
    def encode_item(self, i):
        if isinstance(i, dict):
            i = json.dumps(i, sort_keys=True)
        return i

    def decode_item(self, i):
        try:
            i = json.loads(i)
        except json.JSONDecodeError:
            pass
        except TypeError:
            pass
        return i

    def __getitem__(self, key):
        return dict.__getitem__(self, self.encode_item(key))
    
    def get(self, key, default=None):
        return dict.get(self, self.encode_item(key), default)
    
    def __setitem__(self, key, val):
        dict.__setitem__(self, self.encode_item(key), val)
    
    def __repr__(self):
        dictrepr = dict.__repr__(self)
        return '%s(%s)' % (type(self).__name__, dictrepr)

    def __contains__(self, i):
        return dict.__contains__(self, self.encode_item(i))
    
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
            yield self.decode_item(k), v
    
    def keys(self):
        for k in dict.keys(self):
            yield self.decode_item(k)
    
    def __iter__(self):
        for k in dict.__iter__(self):
            yield self.decode_item(k)
