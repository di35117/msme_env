"""
Linguistic Decoding RL Environment.

Domain-agnostic engine for training agents to infer hidden intent
from asymmetric speaker communication — understatement, overstatement,
deflection — purely from reward signal.

Demo domain: MSME + startup credit management (Indian banking context).

Quick start:
    from linguistic_decoding.core import LinguisticDecodingEnv, LinguisticDecodingAction

    with LinguisticDecodingEnv(base_url="http://localhost:8000", domain="msme_startup") as env:
        result = env.reset()
        action = LinguisticDecodingAction(
            action_type="verify_gst_returns",
            target_id=7,
            reasoning="DPD-12, OEM delay message. Verify before moratorium.",
        )
        result = env.step(action)

Adding a new domain:
    1. Implement DomainAdapter in domains/<your_domain>/adapter.py
    2. Register in domains/__init__.py DOMAIN_REGISTRY
    3. Pass domain="<your_domain>" to LinguisticDecodingEnv
"""

from .core import (
    LinguisticDecodingEnv,
    LinguisticDecodingAction,
    LinguisticDecodingObservation,
)
from .domains import get_adapter, DOMAIN_REGISTRY

__all__ = [
    "LinguisticDecodingEnv",
    "LinguisticDecodingAction",
    "LinguisticDecodingObservation",
    "get_adapter",
    "DOMAIN_REGISTRY",
]
