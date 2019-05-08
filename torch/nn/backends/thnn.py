from .backend import FunctionBackend


class THNNFunctionBackend(FunctionBackend):

    def __reduce__(self):
        return (_get_thnn_function_backend, ())

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def __copy__(self):
        return self


def _get_thnn_function_backend():
    return backend


def _initialize_backend():
    from .._functions.thnn import _all_functions as _thnn_functions
    for cls in _thnn_functions:
        name = cls.__name__
        backend.register_function(name, cls)


backend = THNNFunctionBackend()
_initialize_backend()
