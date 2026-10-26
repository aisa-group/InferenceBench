import contextlib
import importlib
import importlib.machinery
import importlib.util
import inspect
import os
import sys
import types
from importlib.abc import Loader, MetaPathFinder
from pathlib import Path


def _module(name):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _stub_accelerate_bnb():
    if 'accelerate.utils.bnb' in sys.modules or 'accelerate' in sys.modules:
        return
    stub = types.ModuleType('accelerate.utils.bnb')
    stub.__file__ = '<inferencebench_accelerate_bnb_stub>'
    stub.__package__ = 'accelerate.utils'

    def dispatch_model(*a, **kw):
        from accelerate.big_modeling import dispatch_model
        return dispatch_model(*a, **kw)

    def init_empty_weights(*a, **kw):
        from accelerate.big_modeling import init_empty_weights
        return init_empty_weights(*a, **kw)

    def has_4bit_bnb_layers(*a, **kw):
        return False

    def load_and_quantize_model(*a, **kw):
        raise NotImplementedError('bnb stub: load_and_quantize_model')

    stub.dispatch_model = dispatch_model
    stub.init_empty_weights = init_empty_weights
    stub.has_4bit_bnb_layers = has_4bit_bnb_layers
    stub.load_and_quantize_model = load_and_quantize_model
    sys.modules['accelerate.utils.bnb'] = stub


def _triton_default_dir(kind):
    d = os.environ.get(f'TRITON_{kind.upper()}_DIR')
    if d:
        p = Path(d).expanduser()
    else:
        cache = os.environ.get('TRITON_CACHE_DIR')
        xdg = os.environ.get('XDG_CACHE_HOME')
        if cache:
            root = Path(cache).expanduser().parent
        else:
            root = Path(xdg) if xdg else Path.home() / '.cache'
        p = root / ('triton' if kind == 'cache' else f'triton_{kind}')
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _patch_triton_cache_dirs():
    import triton.runtime.cache as triton_cache

    def default_cache_dir():
        return _triton_default_dir('cache')

    def default_dump_dir():
        return _triton_default_dir('dump')

    def __getattr__(name):
        if name == 'default_cache_dir':
            return default_cache_dir
        if name == 'default_dump_dir':
            return default_dump_dir
        if name.startswith('default_') and name.endswith('_dir'):
            kind = name[len('default_'):-len('_dir')]
            return lambda: _triton_default_dir(kind)
        raise AttributeError(name)

    for name, value in (('default_cache_dir', default_cache_dir), ('default_dump_dir', default_dump_dir),
                        ('__getattr__', __getattr__)):
        if not hasattr(triton_cache, name):
            setattr(triton_cache, name, value)


def _patch_transformers_auto_processor():
    import transformers
    if hasattr(transformers, 'AutoProcessor'):
        return

    class _AutoProcessorProxy:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            from transformers.models.auto.processing_auto import AutoProcessor
            return AutoProcessor.from_pretrained(*args, **kwargs)

    real = getattr(_module('transformers.models.auto.processing_auto'), 'AutoProcessor', None)
    transformers.AutoProcessor = real or _AutoProcessorProxy


def _patch_special_tokens_extended():
    import transformers.tokenization_utils_base as tub
    attr = 'all_special_tokens_extended'

    def extended(self):
        try:
            vals = getattr(self, 'all_special_tokens', [])
            return list(vals) if vals is not None else []
        except Exception:
            return []

    base = getattr(tub, 'PreTrainedTokenizerBase', None)
    if base is not None and not hasattr(base, attr):
        base.all_special_tokens_extended = property(extended)

    mixin = getattr(tub, 'SpecialTokensMixin', None)
    if mixin is None or not hasattr(mixin, '__getattr__'):
        return
    orig_getattr = mixin.__getattr__
    if getattr(orig_getattr, '_inferencebench_wrapped_all_special_tokens_extended', False):
        return

    def _getattr(self, key):
        if key == attr:
            return extended(self)
        return orig_getattr(self, key)

    _getattr._inferencebench_wrapped_all_special_tokens_extended = True
    mixin.__getattr__ = _getattr


def _patch_vllm_disabled_tqdm():
    weight_utils = importlib.import_module('vllm.model_executor.model_loader.weight_utils')
    tqdm_cls = getattr(weight_utils, 'DisabledTqdm', None)
    if tqdm_cls is None or getattr(tqdm_cls.__init__, '_inferencebench_disable_kw_fix', False):
        return

    def _init(self, *args, _orig_init=tqdm_cls.__init__, **kwargs):
        kwargs.pop('disable', None)
        return _orig_init(self, *args, **kwargs)

    _init._inferencebench_disable_kw_fix = True
    tqdm_cls.__init__ = _init


def _stub_sgl_kernel():
    if os.environ.get('INFERENCE_BENCH_SGLANG_STUB_SGL_KERNEL', '1').strip() == '0':
        return
    if _module('sgl_kernel') is not None:
        return

    def unavailable(*_a, **_k):
        raise RuntimeError('sgl_kernel extension unavailable in this runtime')

    stub = types.ModuleType('sgl_kernel')
    stub.__file__ = '<inferencebench_sgl_kernel_stub>'
    stub.__getattr__ = lambda name: unavailable
    stub.int8_scaled_mm = unavailable
    sys.modules['sgl_kernel'] = stub


def _patch_triton_constexpr_function():
    import triton.language as tl
    if hasattr(tl, 'constexpr_function'):
        return

    def constexpr_function(fn=None, **_kwargs):
        if fn is None:
            return lambda f: f
        return fn

    tl.constexpr_function = constexpr_function


def _patch_vllm_custom_op_name():
    custom_op = getattr(importlib.import_module('vllm.model_executor.custom_op'), 'CustomOp', None)
    if custom_op is None:
        return

    def _init(self, *args, _orig_init=custom_op.__init__, **kwargs):
        try:
            if not hasattr(self.__class__, 'name'):
                self.__class__.name = self.__class__.__name__
        except (AttributeError, TypeError):
            pass
        return _orig_init(self, *args, **kwargs)

    custom_op.__init__ = _init


def _patch_vllm_graph_capture():
    import torch
    mod = importlib.import_module('vllm.utils')
    fn = getattr(mod, 'graph_capture', None)
    if fn is None or getattr(fn, '_inferencebench_wrapped', False):
        return
    try:
        p = inspect.signature(fn).parameters.get('device')
        needs = p is not None and p.default is inspect.Parameter.empty
    except (TypeError, ValueError):
        needs = True
    if not needs:
        return

    def _wrapped(*args, **kwargs):
        if 'device' not in kwargs and not args:
            try:
                kwargs['device'] = torch.cuda.current_device()
            except Exception:
                pass
        return fn(*args, **kwargs)

    _wrapped._inferencebench_wrapped = True
    mod.graph_capture = _wrapped


def _patch_vllm_get_rope():
    rope = importlib.import_module('vllm.model_executor.layers.rotary_embedding')
    fn = getattr(rope, 'get_rope', None)
    if fn is None or getattr(fn, '_inferencebench_sig_filtered', False):
        return

    def _wrapped(*args, **kwargs):
        try:
            allow = set(inspect.signature(fn).parameters)
            kwargs = {k: v for k, v in kwargs.items() if k in allow}
        except (TypeError, ValueError):
            kwargs.pop('rotary_dim', None)
            kwargs.pop('base', None)
        return fn(*args, **kwargs)

    _wrapped._inferencebench_sig_filtered = True
    rope.get_rope = _wrapped


def _use_soft_filelock():
    if os.environ.get('SOFT_FILELOCK', '').strip() not in {'1', 'true', 'True'}:
        return
    import filelock
    if not hasattr(filelock, 'SoftFileLock'):
        return
    filelock.FileLock = filelock.SoftFileLock
    api = _module('filelock._api')
    if api is not None:
        api.FileLock = filelock.SoftFileLock


def _disable_vllm_hf_locks():
    if os.environ.get('INFERENCE_BENCH_DISABLE_HF_LOCKS', '1').strip() == '0':
        return
    weight_utils = importlib.import_module('vllm.model_executor.model_loader.weight_utils')
    if hasattr(weight_utils, 'get_lock'):
        weight_utils.get_lock = lambda *_args, **_kwargs: contextlib.nullcontext()


def _alias_vllm_model_loader():
    alias_name = 'vllm.model_executor.model_loader.loader'
    if alias_name in sys.modules:
        return
    loader = None
    for name in (alias_name, 'vllm.model_executor.model_loader', 'vllm.model_executor.model_loader.base'):
        loader = getattr(_module(name), 'DefaultModelLoader', None)
        if loader is not None:
            break
    if loader is None:
        class DefaultModelLoader:
            def __init__(self, *args, **kwargs):
                raise RuntimeError('DefaultModelLoader is unavailable in this vLLM build; SGLang LoRA requires it.')
        loader = DefaultModelLoader
    alias = types.ModuleType(alias_name)
    alias.DefaultModelLoader = loader
    sys.modules[alias_name] = alias


_VLLM_QUANT_PREFIX = 'vllm.model_executor.layers.quantization'


class _VllmQuantStubLoader(Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        def __getattr__(name):
            cls = type(name, (), {})
            setattr(module, name, cls)
            return cls

        module.__getattr__ = __getattr__
        module.__all__ = []


class _VllmQuantStubFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _VLLM_QUANT_PREFIX and not fullname.startswith(_VLLM_QUANT_PREFIX + '.'):
            return None
        if importlib.machinery.PathFinder.find_spec(fullname, path) is not None:
            return None
        return importlib.util.spec_from_loader(fullname, _VllmQuantStubLoader())


def _install_vllm_quant_stub():
    if not any(type(f).__name__ == '_VllmQuantStubFinder' for f in sys.meta_path):
        sys.meta_path.insert(0, _VllmQuantStubFinder())


for _patch in (
    _stub_accelerate_bnb,
    _patch_triton_cache_dirs,
    _patch_transformers_auto_processor,
    _patch_special_tokens_extended,
    _patch_vllm_disabled_tqdm,
    _stub_sgl_kernel,
    _patch_triton_constexpr_function,
    _patch_vllm_custom_op_name,
    _patch_vllm_graph_capture,
    _patch_vllm_get_rope,
    _use_soft_filelock,
    _disable_vllm_hf_locks,
    _alias_vllm_model_loader,
    _install_vllm_quant_stub,
):
    try:
        _patch()
    except Exception:
        pass
