from collections import namedtuple, defaultdict
import ast
try:
    ast.unparse
    unparse = lambda node : ast.unparse(ast.fix_missing_locations(node))
except AttributeError:
    from astunparse import unparse
import inspect
import math
from msdm.core.distributions.distributions import Distribution

def factor(prob):
    return prob

class Interpreter(ast.NodeTransformer):
    RETURN_VAR_NAME = '__returnval'
    def run(
        self,
        node,
        context=None,
        ast_restorer=None
    ):
        if ast_restorer is None:
            ast_restorer = ASTRestorer()
            ast_restorer.register_children(node)
        self.contexts_manager = ContextsManager(context)
        self.ast_restorer = ast_restorer
        self.temp_vars = TemporaryVariableManager()
        self.visit(node)
        self.ast_restorer.restore_nodes()
        yield from self.contexts_manager.all_contexts()

    def visit_statement(self, node):
        ast.NodeTransformer.generic_visit(self, node)
        compiled_node = compile_node(node, "exec")
        def run_statement(context):
            exec(compiled_node, context.global_context, context.context)
            yield context
        self.contexts_manager.update(run_statement)
        return node

    visit_Assign = visit_statement
    visit_AugAssign = visit_statement

    def visit_Call(self, node):
        """
        This is where non-determinism resulting from `Distribution.sample`
        calls occur and also where the re-weighting of traces
        through `factor` statements happen.

        The `Call` node is replaced with a `Name` node corresponding
        to a variable that has been assigned the value resulting from
        the call.

        Note that this will *not* enter any other function to see if it has
        statements with `Distribution.sample` or `factor` inside.
        It will simply call the function like normal.
        """
        ast.NodeTransformer.generic_visit(self, node)
        compiled_node = compile_node(node, "eval")
        _result_name = self.temp_vars.new_varname("__call_res")
        def run_Call(context):
            called_name = unparse(node.func)
            called = context.get(called_name)
            if called == factor: #factor is a special function defined in this module
                new_prob = eval(compiled_node, context.global_context, context.context)
                if new_prob > 0:
                    yield context.updated_copy(score=math.log(new_prob))
            elif ( #is a sample method from a Distribution
                called.__name__ == "sample" and \
                inspect.ismethod(called) and \
                issubclass(called.__self__.__class__, Distribution)
            ):
                for val, prob in called.__self__.items():
                    if prob == 0:
                        continue
                    new_context = context.updated_copy(
                        context={_result_name: val},
                        score=math.log(prob)
                    )
                    yield new_context
            else:
                val = eval(compiled_node, context.global_context, context.context)
                new_context = context.updated_copy(
                    context={_result_name: val},
                )
                yield new_context
        self.contexts_manager.update(run_Call)
        self.ast_restorer.register_node_to_restore(node)
        return ast.Name(id=_result_name, ctx=ast.Load())

    def visit_UnaryOp(self, node):
        """
        Special ~ (__invert__) operator for sampling from a
        `Distribution` object handled here
        """
        if not isinstance(node.op, ast.Invert):
            self.generic_visit(node)
            return node
        node_operand = self.visit(node.operand)
        compiled_op = compile_node(node_operand, "eval")
        _result_name = self.temp_vars.new_varname("__invert_res")
        def run_UnaryOp(context):
            # we don't know if the operand is a Distribution until runtime
            operand = context.get(unparse(node_operand))
            if isinstance(operand, Distribution):
                for val, prob in operand.items():
                    if prob == 0:
                        continue
                    new_context = context.updated_copy(
                        context={_result_name: val},
                        score=math.log(prob)
                    )
                    yield new_context
            else:
                val = ~operand
                new_context = context.updated_copy(
                    context={_result_name: val},
                )
                yield new_context
        self.contexts_manager.update(run_UnaryOp)
        self.ast_restorer.register_node_to_restore(node)
        return ast.Name(id=_result_name, ctx=ast.Load())

    def visit_If(self, node):
        node_test = self.visit(node.test)
        compiled_test = compile_node(node_test, "eval")
        def run_If(context):
            if eval(compiled_test, context.global_context, context.context):
                yield from Interpreter().run(
                    node=ast.Module(body=node.body),
                    context=context,
                )
            else:
                yield from Interpreter().run(
                    node=ast.Module(body=node.orelse),
                    context=context,
                )
        self.contexts_manager.update(run_If)
        return node

    def visit_Return(self, node):
        ast.NodeTransformer.generic_visit(self, node)
        compiled_val = compile_node(node.value, "eval")
        def run_Return(context):
            context.context[self.RETURN_VAR_NAME] = eval(compiled_val, context.global_context, context.context)
            yield context.updated_copy(status="return")
        self.contexts_manager.update(run_Return)
        return node

    def visit_IfExp(self, node):
        node_test = self.visit(node.test)
        compiled_test = compile_node(node_test, "eval")
        _res_name = self.temp_vars.new_varname("__ifexp_val")
        node_if = ast.Assign(
            targets=[ast.Name(id=_res_name, ctx=ast.Store())],
            value=node.body
        )
        node_else = ast.Assign(
            targets=[ast.Name(id=_res_name, ctx=ast.Store())],
            value=node.orelse
        )
        def run_If(context):
            if eval(compiled_test, context.global_context, context.context):
                yield from Interpreter().run(
                    ast.Module(body=node_if),
                    context,
                )
            else:
                yield from Interpreter().run(
                    ast.Module(body=node_else),
                    context,
                )
        self.contexts_manager.update(run_If)
        self.ast_restorer.register_node_to_restore(node)
        return ast.Name(id=_res_name, ctx=ast.Load())

    def visit_not_implemented(self, node):
        raise NotImplementedError(
            f"{node.__class__.__name__} interpreter not implemented"
        )

    # likely to implement
    visit_Lambda = \
    visit_not_implemented

    # maybe supported someday
    visit_For = visit_While = visit_With = visit_Match = \
    visit_Break = visit_Continue = \
    visit_ListComp = visit_SetComp = visit_GeneratorExp = \
    visit_DictComp = \
    visit_FunctionDef = \
    visit_not_implemented

    # unlikely to be supported
    visit_Global = visit_Nonlocal = \
    visit_AsyncFunctionDef = visit_AsyncFor = AsyncWith = \
    visit_ClassDef = visit_Import = \
    visit_not_implemented

NodeRecord = namedtuple("NodeRecord", "node parent field field_is_list field_idx")
class ASTRestorer:
    """
    This saves information about the original structure of an AST,
    so that it can be "restored" later.
    """
    def __init__(self):
        self.node_records = {}
        self.to_restore = []

    def register_children(self, parent):
        for field, child in ast.iter_fields(parent):
            if isinstance(child, ast.AST):
                child_hash = hash(child)
                self.node_records[child_hash] = NodeRecord(
                    node=child,
                    parent=parent,
                    field=field,
                    field_is_list=False,
                    field_idx=None
                )
                self.register_children(child)
            if isinstance(child, list):
                for child_i, list_child in enumerate(child):
                    if isinstance(list_child, ast.AST):
                        child_hash = hash(list_child)
                        self.node_records[child_hash] = NodeRecord(
                            node=list_child,
                            parent=parent,
                            field=field,
                            field_is_list=True,
                            field_idx=child_i
                        )
                        self.register_children(list_child)

    def register_node_to_restore(self, node):
        self.to_restore.append(node)

    def restore_nodes(self):
        while self.to_restore:
            node = self.to_restore.pop()
            record = self.node_records[hash(node)]
            if record.field_is_list:
                getattr(record.parent, record.field)[record.field_idx] = node
            else:
                setattr(record.parent, record.field, node)

def compile_node(node, mode):
    # can this be done faster by compiling node directly?
    return compile(unparse(node), "<string>", mode)

class TemporaryVariableManager:
    def __init__(self):
        self.counts = defaultdict(lambda :0)
    def new_varname(self, prefix):
        self.counts[prefix] += 1
        return f"{prefix}_{self.counts[prefix] -1}"

Context = namedtuple("Context", ["context", "global_context", "score", "status"])
class Context(Context):
    def updated_copy(self, context=None, score=0, status=None):
        if context is None: context = {}
        if status is None: status = self.status
        new_context = Context(
            context={**self.context, **context},
            global_context=self.global_context,
            score=self.score + score,
            status=status,
        )
        return new_context

    def get(self, name):
        return eval(name, self.global_context, self.context)

class ContextsManager:
    def __init__(self, context=None):
        if context is None:
            context = Context(
                context={},
                global_context={},
                score=0,
                status=None
            )
        self.active_contexts = [context]
        self.done_contexts = []
        self.continuing_contexts = []

    def update(self, next_context_func):
        active_contexts = []
        while self.active_contexts:
            context = self.active_contexts.pop()
            for new_context in next_context_func(context):
                if new_context.status is None:
                    active_contexts.append(new_context)
                elif new_context.status == "continue":
                    self.continuing_contexts.append(new_context)
                elif new_context.status == "break":
                    self.done_contexts.append(new_context)
                elif new_context.status == "return":
                    self.done_contexts.append(new_context)
                else:
                    raise ValueError("Unknown context status")
        self.active_contexts = active_contexts

    def all_contexts(self):
        yield from self.active_contexts
        yield from self.done_contexts
        yield from self.continuing_contexts
