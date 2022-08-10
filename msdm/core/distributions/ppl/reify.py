import inspect
import copy
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

def reify(function):
    return FunctionReifier(function).reified_function

class FunctionReifier:
    ARG_VARS_NAME = "__arg_vars"
    def __init__(self, function):
        self.function = function
        self.reused_ast_handler_stack = []
        func_ast = self.get_function_ast()

        # Get arguments to func as a dictionary
        # HACK - this will miss non-literal default values not in global scope
        arg_string = unparse(func_ast.args).strip()
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

    def get_function_ast(self):
        root = ast.parse(textwrap.dedent(strip_comments(inspect.getsource(self.function)))).body[0]
        return root

    def _create_reified_function(self):
        # @functools.wraps(self.function)
        def wrapper(*args, **kws):

            if len(self.reused_ast_handler_stack) == 0:
                func_ast = self.get_function_ast()
                func_ast_body = ast.Module(body=func_ast.body)
                ast_restorer = ASTRestorer()
                # this way we don't have to re-register body nodes on every call
                ast_restorer.register_children(func_ast)
                self.reused_ast_handler_stack.append((
                    func_ast,
                    func_ast_body,
                    ast_restorer
                ))
            func_ast, func_ast_body, ast_restorer = self.reused_ast_handler_stack.pop()
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
                node=func_ast_body,
                context=init_context,
                ast_restorer=ast_restorer
            )
            dist = defaultdict(lambda : 0)
            norm = 0
            for context in return_contexts:
                returnval = context.context.get(Interpreter.RETURN_VAR_NAME, None)
                prob = math.exp(context.score)
                dist[returnval] += prob
                norm += prob
            dist = {e: p/norm for e, p in dist.items()}
            self.reused_ast_handler_stack.append((
                func_ast,
                func_ast_body,
                ast_restorer
            ))
            return DictDistribution(dist)
        wrapper._original_function = self.function
        wrapper._function_reifier = self
        return wrapper
