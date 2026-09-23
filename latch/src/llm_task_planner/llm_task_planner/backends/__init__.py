"""Provider registry. `make_backend("groq", ...)` is the only entry point.

Adding a provider means adding a module here and one registry line - the planner
loop does not change, which is the property the plan asked for and the reason
this package exists at all.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

from .base import Decision, LlmBackend, ToolCall, ToolResult

#: name -> "module:class". Imported lazily so that having one provider's SDK
#: missing does not stop the other from running - installing `anthropic` should
#: not be a prerequisite for a Groq-only setup, or vice versa.
BACKENDS: Dict[str, str] = {
    "anthropic": ".anthropic_backend:AnthropicBackend",
    "groq": ".groq_backend:GroqBackend",
    "deepinfra": ".deepinfra_backend:DeepInfraBackend",
}

#: pip package each backend needs, for the error message when it is missing.
#: Guessing this from the backend name was wrong for the third provider -
#: `deepinfra` is reached through the `openai` package, not a `deepinfra` one.
PACKAGES: Dict[str, str] = {
    "anthropic": "anthropic",
    "groq": "groq",
    "deepinfra": "openai",
}


def make_backend(name: str, model: str, system: str,
                 tools: Sequence[Dict[str, Any]], **kwargs: Any) -> LlmBackend:
    key = (name or "anthropic").strip().lower()
    if key not in BACKENDS:
        raise ValueError(
            f"unknown backend {name!r}. Available: {', '.join(sorted(BACKENDS))}")

    import importlib
    module_path, _, cls_name = BACKENDS[key].partition(":")
    try:
        module = importlib.import_module(module_path, package=__name__)
    except ImportError as exc:
        raise RuntimeError(
            f"backend {key!r} needs a package that is not installed: {exc}. "
            f"Try: pip install --user --break-system-packages "
            f"{PACKAGES.get(key, key)}") from exc

    cls = getattr(module, cls_name)
    # Each backend takes its own extras (effort, temperature); pass only what it
    # accepts so one provider's option is not an error for another.
    import inspect
    accepted = set(inspect.signature(cls.__init__).parameters)
    extras = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(model=model, system=system, tools=tools, **extras)


__all__ = ["BACKENDS", "Decision", "LlmBackend", "ToolCall", "ToolResult",
           "make_backend"]
