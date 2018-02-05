import sys
import ast
from collections import namedtuple

Def = namedtuple('Def', ['name', 'params', 'returns', 'body'])
Param = namedtuple('Param', ['type', 'name'])
Apply = namedtuple('Apply', ['name', 'args', 'kwargs'])
TensorType = namedtuple('TensorType', ['type'])
Ident = namedtuple('Ident', ['id'])
Assign = namedtuple('Assign', ['lhs', 'kind', 'rhs'])


class List(namedtuple('List', ['elems'])):
    def __add__(self, other):
        return List(self.elems + other)


PY2 = sys.version_info[0] == 2
_reserved_prefix = '__jit'
_default_type = TensorType('float')  # TODO: remove


class Builder(object):
    def __call__(self, node):
        try:
            method = getattr(self, 'build_' + node.__class__.__name__)
        except AttributeError:
            raise RuntimeError(node.__class__.__name__ + " isn't a supported Python feature")
        return method(node)


class CountReturns(ast.NodeVisitor):
    def __init__(self):
        self.num_returns = 0

    def visit_Return(self, ret):
        self.num_returns += 1

    @staticmethod
    def get_count(py_def):
        counter = CountReturns()
        counter.visit(py_def)
        return counter.num_returns


_ret_err_msg = ("JIT-ed functions can only have a single return, "
                "and it has to be the last statement in the body")


def build_def(py_def):
    assert len(py_def.decorator_list) == 0
    returns = []
    ret_body = []
    body = py_def.body
    num_returns = CountReturns.get_count(py_def)
    if num_returns == 1:
        ret_stmt, body = body[-1], body[:-1]
        if not isinstance(ret_stmt, ast.Return):
            raise ValueError(_ret_err_msg)
        ret_expr = ret_stmt.value
        ret_vals = ret_expr.elts if isinstance(ret_expr, ast.Tuple) else [ret_expr]
        for i, val in enumerate(ret_vals):
            val_expr = build_expr(val)
            val_name = _reserved_prefix + '_' + str(i)
            returns.append(val_name)
            # TODO[discuss]: get type of return values?
            ret_body.append(Assign([Param(_default_type, Ident(val_name))], '=', val_name))
    elif num_returns > 1:
        raise ValueError(_ret_err_msg)
    return Def(Ident(py_def.name),
               build_param_list(py_def.args),
               returns,
               build_stmt_list(body) + ret_body)


def build_param_list(py_args):
    assert py_args.vararg is None
    assert py_args.kwarg is None
    assert not py_args.defaults
    if PY2:
        # TODO: args are in py_args.args, but are expressions <sigh>
        raise RuntimeError("PY2 not supported")
    else:
        assert not py_args.kwonlyargs
        assert not py_args.kw_defaults
        return List([build_param(arg.arg, arg.annotation) for arg in py_args.args])


def build_param(name, annotation):
    assert annotation is None  # TODO: handle annotations
    return Param(_default_type, Ident(name))


def build_stmt_list(py_stmt_list):
    return List([build_stmt(stmt) for stmt in py_stmt_list])


class StmtBuilder(Builder):
    @staticmethod
    def build_Expr(stmt):
        return build_expr(stmt.value)

build_stmt = StmtBuilder()
_MethodRef = namedtuple('MethodRef', ['self', 'name'])


class ExprBuilder(Builder):

    @staticmethod
    def build_Attribute(expr):
        return _MethodRef(build_expr(expr.value), Ident(expr.attr))

    @staticmethod
    def build_Call(expr):
        ref = build_expr(expr.func, allow_methods=True)
        assert type(ref) is _MethodRef
        args, kwargs = build_args(expr.args, expr.keywords)
        return Apply(ref.name, [ref.self] + args, kwargs)

    @staticmethod
    def build_Name(expr):
        assert not expr.id.startswith(_reserved_prefix)
        return Ident(expr.id)

    def __call__(self, expr, allow_methods=False):
        result = super(ExprBuilder, self).__call__(expr)
        assert type(result) is not _MethodRef or allow_methods
        return result

build_expr = ExprBuilder()

#   BoolOp(boolop op, expr* values)
# | BinOp(expr left, operator op, expr right)
# | UnaryOp(unaryop op, expr operand)
# | Lambda(arguments args, expr body)
# | IfExp(expr test, expr body, expr orelse)
# | Dict(expr* keys, expr* values)
# | Set(expr* elts)
# | ListComp(expr elt, comprehension* generators)
# | SetComp(expr elt, comprehension* generators)
# | DictComp(expr key, expr value, comprehension* generators)
# | GeneratorExp(expr elt, comprehension* generators)
# -- the grammar constrains where yield expressions can occur
# | Await(expr value)
# | Yield(expr? value)
# | YieldFrom(expr value)
# -- need sequences for compare to distinguish between
# -- x < 4 < 3 and (x < 4) < 3
# | Compare(expr left, cmpop* ops, expr* comparators)
# | Call(expr func, expr* args, keyword* keywords)
# | Num(object n) -- a number as a PyObject.
# | Str(string s) -- need to specify raw, unicode, etc?
# | Bytes(bytes s)
# | NameConstant(singleton value)
# | Ellipsis

# -- the following expression can appear in assignment context
# | Attribute(expr value, identifier attr, expr_context ctx)
# | Subscript(expr value, slice slice, expr_context ctx)
# | Starred(expr value, expr_context ctx)
# | Name(identifier id, expr_context ctx)
# | List(expr* elts, expr_context ctx)
# | Tuple(expr* elts, expr_context ctx)


def build_args(py_args, py_kwargs):
    args = [build_expr(py_arg) for py_arg in py_args]
    kwargs = [Attribute(Ident(name), build_expr(value)) for name, value in py_kwargs]
    return args, kwargs


import ast
import inspect
from pprint import pprint


def test(x, y):
    x.add(y)
    return x

py_ast = ast.parse(inspect.getsource(test)).body[0]
print(ast.dump(py_ast))
pprint(build_def(py_ast))
