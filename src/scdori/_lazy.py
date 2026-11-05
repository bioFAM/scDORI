"""Resolve public exports without importing optional analysis tools at startup."""

from importlib import import_module

_EXTRAS = {
    "gtfparse": "preprocessing",
    "harmonypy": "preprocessing",
    "tangermeme": "preprocessing",
    "leidenalg": "preprocessing",
    "scanpy": "plot",  # also supplied by preprocessing
    "matplotlib": "plot",
    "seaborn": "plot",
    "umap": "plot",
}


def resolve_export(name, exports, namespace):
    """Load a declared public name and cache it after a successful import."""
    if name not in exports:
        raise AttributeError(f"module {namespace['__name__']!r} has no attribute {name!r}")
    module, attribute = exports[name]
    try:
        loaded = import_module(module, namespace["__name__"])
        value = loaded if attribute is None else getattr(loaded, attribute)
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".")[0]
        if missing in _EXTRAS:
            extra = "preprocessing" if ".pp" in namespace["__name__"] or module == ".pp" else _EXTRAS[missing]
            raise ModuleNotFoundError(
                f"{name} requires optional dependencies. Install with: pip install 'scdori[{extra}]'", name=exc.name
            ) from exc
        raise
    namespace[name] = value
    return value
