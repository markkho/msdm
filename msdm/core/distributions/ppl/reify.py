import functools
from collections import defaultdict
import ast
from astunparse import unparse
import textwrap
import inspect
import functools
import math
from msdm.core.distributions.dictdistribution import DictDistribution
from msdm.core.distributions.ppl.interpreter import Interpreter, ASTRestorer, Context
from msdm.core.distributions.ppl.utils import strip_comments 

@functools.lru_cache()
def reify(function):
    """
    Transform the function into one that returns explicit,
    normalized `DictDistribution`s over return values.
    """
    func_ast = get_function_ast(function)
    body = ast.Module(body=func_ast.body)
    arg_extractor = arg_extractor_factory(function)

    # this way we don't have to re-register body nodes on every call
    ast_restorer = ASTRestorer()
    ast_restorer.register_children(body)

    @functools.wraps(function)
    def distribution_function(*_args, **_kwargs):
        closure_vars = function_closure(function)
        arg_vars = arg_extractor(_args, _kwargs)
        context = Context(
            context={**closure_vars, **arg_vars},
            global_context=function.__globals__,
            score=0,
            status=None
        )
        interpreter =Interpreter()
        return_contexts = interpreter.run(
            body,
            context,
            ast_restorer=ast_restorer
        )
        dist = defaultdict(lambda : 0)
        norm = 0
        for context in return_contexts:
            returnval = context.context.get(interpreter.RETURN_VAR_NAME, None)
            prob = math.exp(context.score)
            dist[returnval] += prob
            norm += prob
        dist = {e: p/norm for e, p in dist.items()}
        return DictDistribution(dist)
    return distribution_function

def arg_extractor_factory(func):
    # save the values of global variables involved in the function signature
    # HACK to get arguments of func as a dictionary - misses default values from local scope of function definition
    func_def_ast = get_function_ast(func)
    func_args_node = ast.Module(body=[
        ast.FunctionDef(
            name=func_def_ast.name,
            args=func_def_ast.args,
            body=[ast.Return(
                value=ast.Call(
                    func=ast.Name(id='locals', ctx=ast.Load()),
                    args=[],
                    keywords=[]
                ),
            )],
            decorator_list=[],
            returns=None
        ),
        ast.Assign(
            targets=[ast.Name(id='_arg_vars', ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id=func_def_ast.name, ctx=ast.Load()),
                args=[ast.Starred(value=ast.Name(id='_args', ctx=ast.Load()), ctx=ast.Load())],
                keywords=[ast.keyword(arg=None, value=ast.Name(id='_kwargs', ctx=ast.Load()))]
            )
        )
    ])
    compiled_arg_script = compile(unparse(func_args_node), "<string>", "exec")
    def arg_extractor(_args, _kwargs):
        closure_vars = function_closure(func)
        func_arg_context = {**closure_vars, '_args': _args, '_kwargs': _kwargs}
        exec(compiled_arg_script, func.__globals__, func_arg_context)
        return func_arg_context['_arg_vars']
    return arg_extractor

def function_closure(func):
    # get closure associated with func
    if func.__closure__ is not None:
        closure_keys = func.__code__.co_freevars
        closure_values = [cell.cell_contents for cell in func.__closure__]
        return dict(zip(closure_keys, closure_values))
    else:
        return {}

def get_function_ast(func):
    root = ast.parse(textwrap.dedent(strip_comments(inspect.getsource(func)))).body[0]
    return root
