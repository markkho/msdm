import inspect
import textwrap
import functools
from collections import defaultdict
import math
import ast
try:
    ast.unparse
    unparse = lambda node : ast.unparse(ast.fix_missing_locations(node))
except AttributeError:
    from astunparse import unparse

from msdm.core.distributions.ppl.interpreter import Interpreter, ASTRestorer, Context
from msdm.core.distributions.ppl.utils import strip_comments
from msdm.core.distributions.dictdistribution import DictDistribution

class FunctionReifier:
    ARG_VARS_NAME = "__arg_vars"
    def __init__(self, function):
        self.function = function
        self.func_ast = self.get_function_ast(function)
        self.func_ast_body = ast.Module(body=self.func_ast.body)
        self.ast_restorer = ASTRestorer()

        # this way we don't have to re-register body nodes on every call
        self.ast_restorer.register_children(self.func_ast)

        # Get arguments to func as a dictionary
        # HACK - this will miss non-literal default values not in global scope
        arg_string = unparse(self.func_ast.args).strip()
        self.compiled_arg_extractor_script = compile(
            source=textwrap.dedent(f"""
                def dummy_func({arg_string}):
                    return locals()
                {self.ARG_VARS_NAME} = dummy_func(*args, **kws)
            """),
            filename="<string>",
            mode="exec"
        )

        self.reified_function = self._create_reified_function()

    def extract_arg_kws(self, args, kws):
        context = {'args': args, 'kws': kws}
        exec(self.compiled_arg_extractor_script, self.function.__globals__, context)
        return context[self.ARG_VARS_NAME]

    def closure(self):
        if hasattr(self.function, "__closure__") and self.function.__closure__ is not None:
            closure_keys = self.function.__code__.co_freevars
            closure_values = [cell.cell_contents for cell in self.function.__closure__]
            return dict(zip(closure_keys, closure_values))
        else:
            return {}

    @classmethod
    def get_function_ast(cls, func):
        root = ast.parse(textwrap.dedent(strip_comments(inspect.getsource(func)))).body[0]
        return root

    def _create_reified_function(self):
        # @functools.wraps(self.function)
        def wrapper(*args, **kws):
            init_context = Context(
                context={
                    **self.closure(),
                    **self.extract_arg_kws(args=args, kws=kws)
                },
                global_context=self.function.__globals__,
                score=0,
                status=None
            )
            return_contexts = Interpreter().run(
                node=self.func_ast_body,
                context=init_context,
                ast_restorer=self.ast_restorer
            )
            dist = defaultdict(lambda : 0)
            norm = 0
            for context in return_contexts:
                returnval = context.context.get(Interpreter.RETURN_VAR_NAME, None)
                prob = math.exp(context.score)
                dist[returnval] += prob
                norm += prob
            dist = {e: p/norm for e, p in dist.items()}
            return DictDistribution(dist)
        wrapper._original_function = self.function
        return wrapper
