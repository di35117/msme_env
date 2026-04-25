"""
Domain registry for the Linguistic Decoding RL environment.

To add a new domain:
  1. Implement DomainAdapter in domains/<your_domain>/adapter.py
  2. Import it here and add to DOMAIN_REGISTRY.

The server loads adapters by name:
  adapter = get_adapter("msme_startup")
"""

from .base import DomainAdapter
from .msme_startup.adapter import MSMEStartupAdapter

DOMAIN_REGISTRY: dict[str, type[DomainAdapter]] = {
    "msme_startup": MSMEStartupAdapter,
    # "enterprise_support": EnterpriseSupportAdapter,   # future domain
    # "procurement":        ProcurementAdapter,         # future domain
}


def get_adapter(domain_id: str) -> DomainAdapter:
    """
    Instantiate and return the adapter for the given domain.

    Raises:
        ValueError if domain_id is not in the registry.
    """
    cls = DOMAIN_REGISTRY.get(domain_id)
    if cls is None:
        registered = list(DOMAIN_REGISTRY.keys())
        raise ValueError(
            f"Unknown domain '{domain_id}'. "
            f"Registered domains: {registered}"
        )
    return cls()


__all__ = ["DomainAdapter", "get_adapter", "DOMAIN_REGISTRY"]
