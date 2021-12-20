import inspect
import functools
from collections import defaultdict
import textwrap
from msdm.core.distributions.jointprobabilitytable import \
    Assignment, JointProbabilityTable
from msdm.core.exceptions import SpecificationException


def make_factor(
    variable_domains : dict = None,
    use_cache : bool = True,
    debug_mode : bool = False
):
    """
    Create a wrapper function that binds
    meta-data to a function so that it can be treated as a
    factor with named variables, call signature,
    caching, etc.

    Parameters
    ----------
    variable_domains : dict
        Dictionary with domains associated with variables
        input or output of a factor. (optional)
    use_cache : bool
    debug_mode : bool
        Checks that return variables of factor match signature
    """
    assert variable_domains is None, "Not implemented yet"
    def factor(function):
        if use_cache:
            function = cache(function)
            function.factor_cache_info = lambda : factor_cache_info(function)
        if debug_mode:
            function = debug_wrap(function)
        function._is_factor = True
        function._debug_mode = debug_mode
        function.variable_domains = variable_domains
        function.signature = get_signature(function)
        return function
    return factor

def combine(factors, use_cache=True, debug_mode=False):
    """
    Combines a list of factors into a new factor function,
    which is equivalent to a factor graph resulting from
    combining the factors. Factors need to be passed in
    in the order that they should be evaluated (i.e., topological order).

    Parameters
    ----------
    factors : list of factors or JointProbabilityTable

    use_cache : bool

    debug_mode :bool
    """
    input_variables, output_variables = get_input_output_variables(factors)
    input_str = ', '.join(input_variables)
    output_str = ', '.join([repr(o) for o in output_variables])
    input_assignment_str = ', '.join([f"{v}={v}" for v in input_variables])
    context = locals()
    # need to create a closure that references factors
    exec(textwrap.dedent(f"""
    def make_combined(factors):
        def combined({input_str}) -> [{output_str}]:
            dist = JointProbabilityTable.from_pairs([
                [dict({input_assignment_str}), 1.0]
            ])
            dist = dist.join(*factors)
            return dist
        return combined
    combined = make_combined(factors)
    """), globals(), context)
    combined = context['combined']
    wrapper = make_factor(
        variable_domains=None,
        use_cache=use_cache,
        debug_mode=debug_mode
    )
    combined = wrapper(combined)
    return combined

def get_input_output_variables(factors):
    variable_graph = defaultdict(lambda : dict(parents=set(), children=set()))
    for f in factors:
        if isinstance(f, JointProbabilityTable):
            input_variables = [None]
            output_variables = list(f.variables())
        elif hasattr(f, '_is_factor') and f._is_factor:
            sig = f.signature
            if len(sig['input_variables']) == 0:
                input_variables = [None]
            else:
                input_variables = sig['input_variables']
            output_variables = sig['output_variables']
        else:
            raise ValueError("Inputs must be a factor or JointProbabilityTable")
        for input_v in input_variables:
            for output_v in output_variables:
                variable_graph[input_v]['children'].add(output_v)
                variable_graph[output_v]['parents'].add(input_v)
    graph_inputs = []
    graph_outputs = []
    for v in variable_graph:
        if v is None:
            continue
        if len(variable_graph[v]['children']) == 0:
            graph_outputs.append(v)
        if len(variable_graph[v]['parents']) == 0:
            graph_inputs.append(v)
    return graph_inputs, graph_outputs

def debug_wrap(function):
    signature = get_signature(function)
    output_variables = set(signature['output_variables'])

    @functools.wraps(function)
    def debugged_function(*args, **kwds):
        result = function(*args, **kwds)
        if result is None:
            return result
        if not isinstance(result, JointProbabilityTable):
            raise ValueError((
                f"{function.__name__} does not return None or a JointProbabilityTable "
                f"for args={args}, kwds={kwds}; return = {result}"
            ))
        for assignment, prob in result.items():
            assert isinstance(assignment, Assignment)
            variables = set(assignment.variables())
            if variables != output_variables:
                raise InconsistentVariablesException((
                    f"{function.__name__} returns variables that do not match signature: "
                    f"Returned: {variables}; "
                    f"Expected: {output_variables}"
                ))
        return result
    return debugged_function

class InconsistentVariablesException(Exception):
    pass

def cache(user_function):
    """Simple cache"""
    cache = {}
    sentinel = object()
    hits = misses = 0
    @functools.wraps(user_function)
    def wrapper(*args, **kwds):
        nonlocal hits, misses
        key = functools._make_key(args, kwds, typed=False)
        result = cache.get(key, sentinel)
        if result is not sentinel:
            hits += 1
            return result
        misses += 1
        result = user_function(*args, **kwds)
        cache[key] = result
        return result
    def cache_info():
        return {"hits": hits, "misses": misses, "currsize": len(cache)}
    wrapper.cache_info = cache_info
    wrapper.get_cache = lambda : cache
    return wrapper

def factor_cache_summary(factor):
    cache = factor.get_cache()
    if len(cache) == 0:
        return {}
    table_sizes = [len(table) for table in cache.values() if table is not None]
    mean_table_size = np.mean(table_sizes) if table_sizes else None
    max_table_size = np.max(table_sizes) if table_sizes else None
    min_table_size = np.min(table_sizes) if table_sizes else None
    return {
        **factor.cache_info(),
        **dict(
            mean_table_size = mean_table_size,
            max_table_size = max_table_size,
            min_table_size = min_table_size
        )
    }

def get_signature(function):
    sig = inspect.signature(function)
    input_variables = list(sig.parameters.keys())
    if sig.return_annotation == inspect._empty:
        raise SpecificationException((
            f"Output variables required in call signature "
            f"for {function.__name__} "
            f"(e.g., f(input_var) -> ['my_output_var']:)"
        ))
    output_variables = list(sig.return_annotation)
    return dict(
        input_variables=tuple(input_variables),
        output_variables=tuple(output_variables)
    )
