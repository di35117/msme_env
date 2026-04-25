"""
Domain adapter interface for generalized linguistic decoding environments.

This layer allows the environment core to remain stable while different
domains (finance, support, compliance, negotiation) provide their own
world generation and reward logic.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class DomainAdapter(ABC):
    """Abstract domain contract used by the generalized architecture."""

    @property
    @abstractmethod
    def domain_id(self) -> str:
        """Unique domain identifier."""

    @property
    @abstractmethod
    def total_entities(self) -> int:
        """Total entities/speakers/accounts per episode."""

    @property
    @abstractmethod
    def time_horizon(self) -> int:
        """Number of time steps in an episode."""

    @property
    @abstractmethod
    def valid_actions(self) -> List[str]:
        """Allowed action types in this domain."""

    @abstractmethod
    def generate_world(
        self,
        episode: int,
        adversarial_weaknesses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Create initial world state for a new episode."""

    @abstractmethod
    def classify_outcome(
        self,
        action_type: str,
        account_type: str,
        hidden_profile: Dict[str, Any],
        current_month: int,
        action_params: Dict[str, Any],
    ) -> str:
        """Map action + hidden state to an outcome label."""

    @abstractmethod
    def compute_step_reward(
        self,
        action_type: str,
        account_type: str,
        outcome: str,
        hidden_profile: Dict[str, Any],
    ) -> float:
        """Compute immediate reward."""

    @abstractmethod
    def compute_episode_reward(
        self,
        hidden_profiles: Dict[int, Dict[str, Any]],
        episode_history: List[Dict[str, Any]],
        episode_num: int,
        final_month: int,
    ) -> Dict[str, float]:
        """Compute episode-level reward breakdown including total."""

    @abstractmethod
    def propagate_effects(
        self,
        hidden_profiles: Dict[int, Dict[str, Any]],
        account_id: int,
        account_type: str,
        action_type: str,
        outcome: str,
        trust_delta: float,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[int]]:
        """Return cluster/ecosystem effects plus cross-contamination accounts."""

