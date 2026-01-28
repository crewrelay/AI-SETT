"""Provider registry — discover and instantiate model providers."""

from __future__ import annotations

from typing import Optional, Type

from .base import ModelProvider

_REGISTRY: dict[str, Type[ModelProvider]] = {}


def register(name: str):
    """Decorator to register a provider class under a name."""
    def decorator(cls: Type[ModelProvider]):
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_provider(name: str, api_key: str, base_url: Optional[str] = None, **kwargs) -> ModelProvider:
    """Instantiate a registered provider by name."""
    # Lazy-import all provider modules so decorators run
    _ensure_loaded()
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")
    return _REGISTRY[name](api_key=api_key, base_url=base_url, **kwargs)


def list_providers() -> list[str]:
    """Return names of all registered providers."""
    _ensure_loaded()
    return sorted(_REGISTRY)


_loaded = False


def _ensure_loaded():
    global _loaded
    if _loaded:
        return
    _loaded = True
    # Import submodules so their @register decorators execute
    from . import openai_provider  # noqa: F401
    from . import anthropic_provider  # noqa: F401
    from . import google_provider  # noqa: F401
    from . import mistral_provider  # noqa: F401
    from . import cohere_provider  # noqa: F401
