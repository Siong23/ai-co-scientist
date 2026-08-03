"""Lazy access to the legacy façade used by individual agent modules."""

from importlib import import_module


class _LegacyAgentsProxy:
    def __getattr__(self, name: str):
        """Resolve façade attributes only when an agent method uses them."""
        return getattr(import_module("app.agents"), name)


_legacy = _LegacyAgentsProxy()
