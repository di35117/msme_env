"""
Core data models for the Linguistic Decoding RL environment.

Domain-agnostic. All domain-specific fields live in the
`domain_state` dict, populated by the active DomainAdapter.

The core engine speaks only: speakers, behavioral_signals,
memory_context, and reward. Domains can add whatever they need
inside `domain_state` without touching this file.
"""

from typing import Any, Dict, List, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ---------------------------------------------------------------------------
# ACTION
# ---------------------------------------------------------------------------

class LinguisticDecodingAction(Action):
    """
    Domain-agnostic action for the Linguistic Decoding environment.

    `action_type` and `target_id` are core. `parameters` and `reasoning`
    carry any domain-specific payload the agent needs to pass.

    The DomainAdapter validates whether `action_type` is legal for
    `target_id`'s speaker type and raises a step penalty if not.
    """

    action_type: str = Field(
        ...,
        description=(
            "Action to take. Valid values depend on the active domain. "
            "The adapter rejects invalid actions with a step penalty rather "
            "than raising an exception, so the agent learns from wrong tool use."
        ),
    )
    target_id: int = Field(
        ...,
        description=(
            "ID of the speaker/account to act on. "
            "Range depends on the domain (e.g. 1-30 for msme_startup)."
        ),
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Action-specific parameters. Examples: "
            "{'months': 2} for grant_moratorium, "
            "{'deadline_days': 14} for send_firm_reminder."
        ),
    )
    reasoning: str = Field(
        default="",
        description=(
            "Agent's chain-of-thought before selecting this action. "
            "Logged to episodic memory. Used for reward shaping analysis."
        ),
    )


# ---------------------------------------------------------------------------
# OBSERVATION
# ---------------------------------------------------------------------------

class LinguisticDecodingObservation(Observation):
    """
    Domain-agnostic observation for the Linguistic Decoding environment.

    The agent always sees:
      - `speakers`: list of observable speaker states (no hidden fields)
      - `behavioral_signals`: structured proxy signals per speaker
      - Three-tier memory injected as context strings
      - `last_action_result`: outcome of the previous action
      - `active_network_alerts`: propagated network effect warnings
      - Reward bookkeeping

    Domain-specific data lives in `domain_state`. The agent's prompt
    template should include domain_state for full context, but the
    core reward and memory systems operate without it.
    """

    # Episode / time context
    episode: int = Field(default=1, description="Training episode number")
    step: int = Field(default=0, description="Step within the current episode")
    time_step: int = Field(
        default=1,
        description=(
            "Domain time unit. For msme_startup this is the month (1–36). "
            "For other domains it could be a conversation turn or a quarter."
        ),
    )

    # Speaker portfolio
    speakers: List[Dict] = Field(
        default_factory=list,
        description=(
            "Observable state for every speaker in the current episode. "
            "Each entry has: id, speaker_type, last_message, "
            "behavioral_signals, and observable_metrics. "
            "Hidden state (true_health, runway, etc.) is NEVER included."
        ),
    )

    # Portfolio-level summary (domain-computed)
    portfolio_summary: Dict = Field(
        default_factory=dict,
        description="High-level aggregate stats visible to the agent.",
    )

    # Three-tier memory
    working_memory: str = Field(
        default="",
        description="Compact current time-step state. Target < 2 K tokens.",
    )
    semantic_memory_context: str = Field(
        default="",
        description="Relevant patterns retrieved from semantic memory store.",
    )
    episodic_memory_context: str = Field(
        default="",
        description="Similar past cases retrieved from episodic memory store.",
    )

    # Last action feedback
    last_action_result: Optional[Dict] = Field(
        default=None,
        description=(
            "Structured outcome of the previous action: "
            "{'outcome': str, 'message': str, 'network_effects': list, "
            "'step_reward': float}."
        ),
    )

    # Network effect alerts (both topologies merged)
    active_network_alerts: List[str] = Field(
        default_factory=list,
        description=(
            "Human-readable alerts from active network propagation. "
            "Includes both domain-A cluster alerts and domain-B ecosystem alerts."
        ),
    )

    # Domain-specific state (adapter-populated, not interpreted by core)
    domain_state: Dict = Field(
        default_factory=dict,
        description=(
            "Domain-specific structured state. For msme_startup: "
            "{'msme_accounts': [...], 'startup_accounts': [...]}. "
            "For other domains: whatever the adapter needs to surface."
        ),
    )

    # Reward bookkeeping
    step_reward: float = Field(default=0.0, description="Reward earned by last action")
    episode_reward_so_far: float = Field(
        default=0.0, description="Cumulative episode reward"
    )

    # Terminal
    done: bool = Field(
        default=False,
        description="True when the episode ends (final time step reached).",
    )
    reward: Optional[float] = Field(default=None)
    metadata: Dict = Field(default_factory=dict)
