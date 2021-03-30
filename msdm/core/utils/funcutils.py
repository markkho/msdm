import inspect

def cached_property(fn):
    spec = inspect.getfullargspec(fn)
    assert len(spec.args) == 1
    assert len(spec.kwonlyargs) == 0
    assert spec.varargs is None
    assert spec.varkw is None

    key = '_cached_'+fn.__name__
    @property
    def wrapped(self):
        if not hasattr(self, key):
            setattr(self, key, fn(self))
        return getattr(self, key)
    return wrapped

def method_cache(fn):
    '''
    Avoiding functools.lru_cache since it holds onto object references
    when applied to methods of an object, creating a referenc cycle (not great for GC).
    Instead, preferring a strategy where cache storage lives on the object.
    '''
    cache_attr = '_cache_'+fn.__name__
    cache_info_attr = '_cache_info_'+fn.__name__
    def wrapped(self, *args, **kwargs):
        # Since the store is object-local, we always have to ensure
        # objects passed in have the cache initialized.
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, {})
            setattr(self, cache_info_attr, dict(hits=0, misses=0))
        cache = getattr(self, cache_attr)
        cache_info = getattr(self, cache_info_attr)

        # Now we check for this function call, and
        # run the function if it hasn't been called before.
        key = (args, frozenset(kwargs.items()) if kwargs else None)
        if key not in cache:
            cache[key] = fn(self, *args, **kwargs)
            cache_info['misses'] += 1
        cache_info['hits'] += 1
        return cache[key]
    return wrapped
