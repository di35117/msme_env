"""
Core client for the Linguistic Decoding RL environment.

Persistent WebSocket connection to the environment server.
Domain-agnostic: pass `domain` at construction to select the adapter.

Usage:
    >>> env = LinguisticDecodingEnv(
    ...     base_url="http://localhost:8000",
    ...     domain="msme_startup",
    ... )
    >>> result = env.reset()
    >>> print(result.observation.time_step)   # month 1
    >>> print(len(result.observation.speakers))  # 30

    >>> action = LinguisticDecodingAction(
    ...     action_type="verify_gst_returns",
    ...     target_id=7,
    ...     parameters={},
    ...     reasoning="DPD-12, OEM delay message, high cluster centrality. "
    ...               "Verify before moratorium decision.",
    ... )
    >>> result = env.step(action)
    >>> print(result.observation.last_action_result["outcome"])
    >>> print(result.reward)

Docker usage:
    >>> env = LinguisticDecodingEnv.from_docker_image(
    ...     "linguistic-decoding-env:latest",
    ...     domain="msme_startup",
    ... )
"""

from typing import Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import LinguisticDecodingAction, LinguisticDecodingObservation


class LinguisticDecodingEnv(
    EnvClient[LinguisticDecodingAction, LinguisticDecodingObservation, State]
):
    """
    Client for the domain-agnostic Linguistic Decoding RL environment.

    The `domain` parameter selects which DomainAdapter the server loads.
    Current domains: "msme_startup" (default).

    The client is intentionally thin — it does not know about domain-specific
    fields. All domain logic lives server-side in the adapter.
    """

    def __init__(self, base_url: str, domain: str = "msme_startup", **kwargs):
        super().__init__(base_url=base_url, **kwargs)
        self.domain = domain

    def _step_payload(self, action: LinguisticDecodingAction) -> Dict:
        """Convert action to JSON payload, injecting domain."""
        return {
            "domain":      self.domain,
            "action_type": action.action_type,
            "target_id":   action.target_id,
            "parameters":  action.parameters,
            "reasoning":   action.reasoning,
        }

    def _parse_result(
        self, payload: Dict
    ) -> StepResult[LinguisticDecodingObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", {})

        observation = LinguisticDecodingObservation(
            episode=obs_data.get("episode", 1),
            step=obs_data.get("step", 0),
            time_step=obs_data.get("time_step", 1),
            speakers=obs_data.get("speakers", []),
            portfolio_summary=obs_data.get("portfolio_summary", {}),
            working_memory=obs_data.get("working_memory", ""),
            semantic_memory_context=obs_data.get("semantic_memory_context", ""),
            episodic_memory_context=obs_data.get("episodic_memory_context", ""),
            last_action_result=obs_data.get("last_action_result"),
            active_network_alerts=obs_data.get("active_network_alerts", []),
            domain_state=obs_data.get("domain_state", {}),
            step_reward=obs_data.get("step_reward", 0.0),
            episode_reward_so_far=obs_data.get("episode_reward_so_far", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

    def _reset_payload(self) -> Dict:
        """Include domain in reset so the server loads the right adapter."""
        return {"domain": self.domain}
