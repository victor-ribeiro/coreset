import numpy as np

WRAPED_ATTRS = (
    "__module__",
    "__name__",
    "__qualname__",
    "__annotations__",
    "__doc__",
    "__defaults__",
    "__kwdefaults__",
)

_UFUNC_METHODS = {"__call__", "reduce", "reduceat", "accumulate", "outer", "inner"}
HANDLED_FUNCS = {}


def np_implements(np_func):
    def decorator(func):
        HANDLED_FUNCS[np_func] = func
        return func

    return decorator


def as_np(fun):
    def deco(arr, *args, **kwargs):
        arr = np.array(arr)
        return fun(arr, *args, **kwargs)

    return deco


def np_factory(fn):
    return np_implements(fn)(as_np(fn))


SUPORT = [
    np_factory(np.sum),
    np_factory(np.mean),
    np_factory(np.multiply),
    np_factory(np.linalg.norm),
    np_factory(np.matmul),
    np_factory(np.transpose),
    np_factory(np.reshape),
    np_factory(np.max),
    np_factory(np.min),
    np_factory(np.stack),
    np_factory(np.hstack),
    np_factory(np.vstack),
    np_factory(np.column_stack),
    # np_factory(np.may_share_memory),
]


def array(obj):

    class Decorator(obj, np.lib.mixins.NDArrayOperatorsMixin):
        def __array__(self, dtype=None):
            buffer = list(self)
            return np.array(buffer, dtype=dtype)

        def __array_function__(self, func, types, args, kwargs):

            if not func in HANDLED_FUNCS:
                return NotImplemented
            return HANDLED_FUNCS[func](*args, **kwargs)

        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            if method == "__call__":
                buffer, *args = inputs
                buffer = np.array(buffer)
                buffer = ufunc(buffer, *args, **kwargs)
                # if self.__meta__:
                #     return self.__class__(buffer, **self.__meta__)
                return self.__class__(
                    buffer,
                )

            return NotImplemented

        def sum(self, *args, **kwargs):
            buffer = np.array(self)
            return np.sum(buffer, *args, **kwargs)

    # Decorator.__class__ = obj.__class__
    Decorator.__module__ = obj.__module__
    Decorator.__qualname__ = obj.__qualname__
    Decorator.__annotations__ = obj.__annotations__

    return Decorator
