from __future__ import annotations
from abc import ABC, ABCMeta, abstractmethod
import logging
from typing import Any, Callable
from collections import deque
import numpy as np
from numpy._typing import _Shape
from numpy.typing import DTypeLike
logger = logging.getLogger(__name__)

class LazyMeta(ABCMeta):

    def __new__(cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs):

        def __getattr__(self, __name: str) -> Any:
            meta_attr = getattr(self._meta, __name)
            if callable(meta_attr):
                return type(self)._wrap_fn(lambda s, *args, **kwargs: getattr(s, __name)(*args, **kwargs), use_self=self)
            elif isinstance(meta_attr, self._tensor_type):
                return type(self)._wrap_fn(lambda s: getattr(s, __name))(self)
            else:
                return meta_attr
        namespace['__getattr__'] = __getattr__

        def mk_wrap(op_name: str, *, meta_noop: bool=False):

            def wrapped_special_op(self, *args, **kwargs):
                return type(self)._wrap_fn(getattr(type(self)._tensor_type, op_name), meta_noop=meta_noop)(self, *args, **kwargs)
            return wrapped_special_op
        for binary_op in ('lt', 'le', 'eq', 'ne', 'ge', 'gt', 'notabs', 'add', 'and', 'floordiv', 'invert', 'lshift', 'mod', 'mul', 'matmul', 'neg', 'or', 'pos', 'pow', 'rshift', 'sub', 'truediv', 'xor', 'iadd', 'iand', 'ifloordiv', 'ilshift', 'imod', 'imul', 'ior', 'irshift', 'isub', 'ixor', 'radd', 'rand', 'rfloordiv', 'rmul', 'ror', 'rpow', 'rsub', 'rtruediv', 'rxor'):
            attr_name = f'__{binary_op}__'
            namespace[attr_name] = mk_wrap(attr_name, meta_noop=True)
        for special_op in ('getitem', 'setitem', 'len'):
            attr_name = f'__{special_op}__'
            namespace[attr_name] = mk_wrap(attr_name, meta_noop=False)
        return super().__new__(cls, name, bases, namespace, **kwargs)

class LazyBase(ABC, metaclass=LazyMeta):
    _tensor_type: type
    _meta: Any
    _data: Any | None
    _lazy: deque[LazyBase]
    _args: tuple
    _func: Callable[[tuple], Any] | None

    def __init__(self, *, meta: Any, data: Any | None=None, lazy: deque[LazyBase] | None=None, args: tuple=(), func: Callable[[tuple], Any] | None=None):
        super().__init__()
        self._meta = meta
        self._data = data
        self._lazy = lazy if lazy is not None else deque()
        self._args = args
        self._func = func
        assert self._func is not None or self._data is not None
        if self._data is None:
            self._lazy.append(self)

    def __init_subclass__(cls) -> None:
        if '_tensor_type' not in cls.__dict__:
            raise TypeError(f"property '_tensor_type' must be defined for {cls!r}")
        return super().__init_subclass__()

    @staticmethod
    def _recurse_apply(o: Any, fn: Callable[[Any], Any]) -> Any:
        if isinstance(o, (list, tuple)):
            L = []
            for item in o:
                L.append(LazyBase._recurse_apply(item, fn))
            if isinstance(o, tuple):
                L = tuple(L)
            return L
        elif isinstance(o, LazyBase):
            return fn(o)
        else:
            return o

    @classmethod
    def _wrap_fn(cls, fn: Callable, *, use_self: LazyBase | None=None, meta_noop: bool | DTypeLike | tuple[DTypeLike, Callable[[tuple[int, ...]], tuple[int, ...]]]=False) -> Callable[[Any], Any]:

        def wrapped_fn(*args, **kwargs):
            if kwargs is None:
                kwargs = {}
            args = ((use_self,) if use_self is not None else ()) + args
            meta_args = LazyBase._recurse_apply(args, lambda t: t._meta)
            if isinstance(meta_noop, bool) and (not meta_noop):
                try:
                    res = fn(*meta_args, **kwargs)
                except NotImplementedError:
                    res = None
            else:
                assert len(args) > 0
                res = args[0]
                assert isinstance(res, cls)
                res = res._meta
                if meta_noop is not True:
                    if isinstance(meta_noop, tuple):
                        dtype, shape = meta_noop
                        assert callable(shape)
                        res = cls.meta_with_dtype_and_shape(dtype, shape(res.shape))
                    else:
                        res = cls.meta_with_dtype_and_shape(meta_noop, res.shape)
            if isinstance(res, cls._tensor_type):

                def collect_replace(t: LazyBase):
                    if collect_replace.shared_lazy is None:
                        collect_replace.shared_lazy = t._lazy
                    else:
                        collect_replace.shared_lazy.extend(t._lazy)
                        t._lazy = collect_replace.shared_lazy
                collect_replace.shared_lazy = None
                LazyBase._recurse_apply(args, collect_replace)
                shared_lazy = collect_replace.shared_lazy
                return cls(meta=cls.eager_to_meta(res), lazy=shared_lazy, args=args, func=lambda a: fn(*a, **kwargs))
            else:
                del res
                eager_args = cls.to_eager(args)
                return fn(*eager_args, **kwargs)
        return wrapped_fn

    @classmethod
    def to_eager(cls, t: Any) -> Any:

        def simple_to_eager(_t: LazyBase) -> Any:

            def already_eager_to_eager(_t: LazyBase) -> Any:
                assert _t._data is not None
                return _t._data
            while _t._data is None:
                lt = _t._lazy.popleft()
                if lt._data is not None:
                    continue
                assert lt._func is not None
                lt._args = cls._recurse_apply(lt._args, already_eager_to_eager)
                lt._data = lt._func(lt._args)
                assert lt._data.dtype == lt._meta.dtype
                assert lt._data.shape == lt._meta.shape
            return _t._data
        return cls._recurse_apply(t, simple_to_eager)

    @classmethod
    def eager_to_meta(cls, t: Any) -> Any:
        return cls.meta_with_dtype_and_shape(t.dtype, t.shape)

    @classmethod
    @abstractmethod
    def meta_with_dtype_and_shape(cls, dtype: Any, shape: Any) -> Any:
        pass

    @classmethod
    def from_eager(cls, t: Any) -> Any:
        if type(t) is cls:
            return t
        elif isinstance(t, cls._tensor_type):
            return cls(meta=cls.eager_to_meta(t), data=t)
        else:
            return TypeError(f'{type(t)!r} is not compatible with {cls._tensor_type!r}')

class LazyNumpyTensor(LazyBase):
    _tensor_type = np.ndarray

    @classmethod
    def meta_with_dtype_and_shape(cls, dtype: DTypeLike, shape: _Shape) -> np.ndarray[Any, Any]:
        cheat = np.zeros(1, dtype)
        return np.lib.stride_tricks.as_strided(cheat, shape, (0 for _ in shape))

    def astype(self, dtype, *args, **kwargs):
        meta = type(self).meta_with_dtype_and_shape(dtype, self._meta.shape)
        full_args = (self, dtype) + args
        return type(self)(meta=meta, args=full_args, lazy=self._lazy, func=lambda a: a[0].astype(*a[1:], **kwargs))

    def tofile(self, *args, **kwargs):
        eager = LazyNumpyTensor.to_eager(self)
        return eager.tofile(*args, **kwargs)