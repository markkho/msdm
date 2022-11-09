import pytest
from msdm.core.utils.funcutils import cached_property, method_cache

def test_cached_property():
    class X(object):
        def __init__(self):
            self._call_count = 0
        @cached_property
        def expensive(self):
            self._call_count += 1
            return 1
    instance = X()

    # initial state
    assert instance._call_count == 0
    assert not hasattr(instance, '_cached_expensive')

    # state after one call
    assert instance.expensive == 1
    assert hasattr(instance, '_cached_expensive')
    assert getattr(instance, '_cached_expensive') == 1
    assert instance._call_count == 1

    # expecting caching to work for second & subsequent calls
    instance.expensive
    assert instance._call_count == 1

    # Ensure it's only usable in the right place
    with pytest.raises(AssertionError):
        class X(object):
            @cached_property
            def expensive(self, argument):
                return 1

def test_method_cache():
    class X(object):
        def __init__(self):
            self._call_count = 0
        @method_cache
        def expensive(self, argument):
            self._call_count += 1
            return argument * 2
    instance = X()
    instance.expensive(3)
    instance.expensive(3)
    assert instance._cache_expensive[((3,), None)] == 6
    assert instance._cache_info_expensive == dict(hits=2, misses=1)
    assert instance._call_count == 1

    # Can handle other entries too
    instance.expensive(4)
    assert instance._cache_expensive[((4,), None)] == 8
    assert instance._cache_info_expensive == dict(hits=3, misses=2)
    assert instance._call_count == 2

    # And we still cache appropriately
    instance.expensive(4)
    assert instance._cache_info_expensive == dict(hits=4, misses=2)
    assert instance._call_count == 2
