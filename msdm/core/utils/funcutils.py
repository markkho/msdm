import inspect, functools

def cached_property(fn):
    '''
    Used to decorate a function that should be a @property
    but also be cached. Worth careful consideration about
    whether we want to use this once we're fully on to Python 3.8.

    Main reason to use in place of functools.cached_property is
    this preserves the semantics of @property, as noted by the docs:
    > The mechanics of cached_property() are somewhat different from property().
    > A regular property blocks attribute writes unless a setter is defined.
    > In contrast, a cached_property allows writes.
    '''
    spec = inspect.getfullargspec(fn)
    assert spec.args == ['self']
    assert len(spec.kwonlyargs) == 0
    assert spec.varargs is None
    assert spec.varkw is None

    key = '_cached_'+fn.__name__
    @property
    @functools.wraps(fn)
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
    @functools.wraps(fn)
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
