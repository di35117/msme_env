"""
Domain registry for generalized linguistic decoding environments.

Current default domain remains msme_startup to preserve backward compatibility.
"""

from .base import DomainAdapter
from .msme_startup import MSMEStartupAdapter

DOMAIN_REGISTRY = {
    "msme_startup": MSMEStartupAdapter,
}


def get_adapter(domain_id: str = "msme_startup") -> DomainAdapter:
    cls = DOMAIN_REGISTRY.get(domain_id)
    if cls is None:
        raise ValueError(f"Unknown domain '{domain_id}'. Registered: {list(DOMAIN_REGISTRY.keys())}")
    return cls()


__all__ = ["DomainAdapter", "DOMAIN_REGISTRY", "get_adapter"]

