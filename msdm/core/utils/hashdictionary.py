import inspect

class HashDictionary(dict):
    """Dictionary with a function for hashing keys"""
    def __init__(self, hash_function=None, *args, **kwargs):
        if hash_function is None:
            hash_function = lambda x: x
        self.hash_function = hash_function
        self._encoded_keys = {} #note: this isn't doing garbage collection
        dict.update(self)
        if len(args) > 0:
            for k, v in args[0]:
                self[k] = v
        if len(kwargs) > 0:
            for k, v in kwargs.items():
                self[k] = v

    def encode_item(self, i):
        if isinstance(i, (dict, list)):
            encoded = self.hash_function(i)
            self._encoded_keys[encoded] = i
            i = encoded
        return i

    def decode_item(self, encoded_item):
        try:
            return self._encoded_keys[encoded_item]
        except KeyError:
            return encoded_item

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
        am = HashDictionary()
        for e in E:
            for k in e.keys():
                am[k] = e[k]
        for k in F.keys():
            am[k] = F[k]
        for k, v in am.items():
            self[k] = v

    def merge(self, *E, **F):
        """Returns a new merged hash dictionary"""
        am = HashDictionary()
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

class DefaultHashDictionary(HashDictionary):
    '''
    DefaultHashDictionary extends HashDictionary to support a default value for unset items,
    akin to collections.defaultdict. Notably, this implementation does not store default values
    after access, a departure from defaultdict.

    The first parameter to DefaultHashDictionary is the function used to generate values
    returned when the HashDictionary has no set value for a key.

    The defaultvalue function can either take no arguments or one argument, which corresponds
    to the key passed to HashDictionary.__getitem__.
    '''
    def __init__(self, default_value, initialize_defaults=False, hash_function=None):
        super().__init__(hash_function=hash_function)

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

